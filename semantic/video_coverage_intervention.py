"""New disjoint split: equal-size random vs source-query coverage training sets."""
from __future__ import annotations
import gc
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from semantic.ytvis import load_dataset,sample_queries,load_video
from semantic.fresh_video_confirmation import select_clips,bootstrap_video_differences
from semantic.teacher import propagate_one,SAM_CHECKPOINT_SHA256
from semantic.model import extract_query_features
from semantic.head_budget_probe import trajectory
from semantic.background_calibration import logits_for,pick_offset,evaluate,predictions,PROTOCOL as CAL_PROTOCOL
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics
from tools.load_opend4rt import load_model,DEFAULT_VARIANT

ROOT=Path('semantic_artifacts/video_coverage_intervention')
PROTOCOL={'version':1,'split_seed':20261014,'train_videos_each':80,'val_videos':20,'confirmation_videos':40,
          'coverage_minimum_source_queried_videos_per_class':3,'coverage_selection':'greedy unmet class deficit divided by candidate availability; stable shuffled-order ties, fill by original shuffled order',
          'training_sets':['random80','coverage80'],'labels':['source_gt','sam_background'],'heads':['linear'],
          'budgets':[500,2000],'seeds':[0,1,2],'query_seed':42,'frames':8,'resolution':128,
          'optimizer':'AdamW lr=0.001 weight_decay=0.01','batch':'128 foreground + 128 background',
          'checkpoint_selection':'validation macro F1 every 50 steps, only final 2000 budget used for confirmation',
          'calibration_grid':CAL_PROTOCOL['offset_grid'],'calibration_selection':CAL_PROTOCOL['offset_selection'],
          'sam_background_rule':'all prompted instance logits < -2 with 2 pixel erosion; positives unchanged from same SAM pass',
          'heldout_results_never_select_videos_or_parameters':True,
          'teacher_gt_diagnostics_do_not_filter_labels':True}


def choose_coverage(candidates,classes_by_video,count=80,minimum=3,num_classes=25):
    availability={c:sum(c in classes_by_video[int(v['video_id'])] for v in candidates) for c in range(1,num_classes+1)}
    if min(availability.values())<minimum:raise ValueError('Insufficient source query video support')
    covered={c:0 for c in availability};selected=[];remaining=list(candidates)
    while min(covered.values())<minimum:
        best=max(range(len(remaining)),key=lambda i:sum(max(minimum-covered[c],0)/availability[c] for c in classes_by_video[int(remaining[i]['video_id'])]))
        clip=remaining.pop(best);selected.append(clip)
        for c in classes_by_video[int(clip['video_id'])]:covered[c]+=1
        if len(selected)>count:raise ValueError('Coverage target exceeds video budget')
    chosen={int(c['video_id']) for c in selected}
    for clip in candidates:
        if len(selected)>=count:break
        if int(clip['video_id']) not in chosen:
            selected.append(clip);chosen.add(int(clip['video_id']))
    if len(selected)!=count:raise ValueError('Insufficient fill videos')
    return selected


def create_manifest(dataset):
    old=json.loads(Path('semantic_artifacts/manifest.json').read_text());fresh=json.loads(Path('semantic_artifacts/fresh_video_confirmation/manifest.json').read_text())
    excluded={int(c['video_id']) for split in ('train','val','test') for c in old[split]}|{int(c['video_id']) for c in fresh['clips']}
    pool,eligible=select_clips(dataset,excluded,count=len(dataset.videos)-len(excluded),seed=20261014)
    confirm=pool[:40];val=pool[40:60];candidates=pool[60:];classes={}
    for clip in candidates:
        labels=np.full((8,128,128),-1,dtype=np.int16);labels[0]=dataset.gt_label_map(clip,0)
        _,targets,_=sample_queries(labels,per_frame=256,seed=42+clip['video_id'])
        classes[clip['video_id']]=set(int(c) for c in targets.unique().tolist() if c>0)
    coverage=choose_coverage(candidates,classes)
    manifest={'excluded_video_ids':sorted(excluded),'eligible_unused':eligible,'val':val,'confirmation':confirm,
              'training_sets':{'random80':candidates[:80],'coverage80':coverage},
              'candidate_source_query_classes':{str(k):sorted(v) for k,v in classes.items()}}
    for group in manifest['training_sets'].values():
        if {c['video_id'] for c in group}&({c['video_id'] for c in val+confirm}|excluded):raise ValueError('Training/held-out leakage')
    if {c['video_id'] for c in val}&{c['video_id'] for c in confirm}:raise ValueError('Held-out overlap')
    return manifest


def freeze_protocol(dataset):
    ROOT.mkdir(parents=True,exist_ok=True);manifest_path=ROOT/'manifest.json'
    if manifest_path.exists():
        # Preserve the preregistered split even after its videos have been processed.
        manifest=json.loads(manifest_path.read_text())
    else:manifest=create_manifest(dataset);atomic_json(manifest_path,manifest)
    protocol=dict(PROTOCOL,manifest_sha256=file_hash(manifest_path),annotation_sha256=file_hash('semantic_artifacts/ovis/annotations_train.json'),
                  sam_checkpoint_sha256=SAM_CHECKPOINT_SHA256,model_config_sha256=file_hash(DEFAULT_VARIANT/'model.yaml'))
    path=ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Changed protocol/inputs')
    atomic_json(path,protocol)
    return manifest,protocol


def training_union(manifest):
    return sorted({c['video_id']:c for group in manifest['training_sets'].values() for c in group}.values(),key=lambda c:c['video_id'])


def teachers(dataset,manifest):
    from sam2.build_sam import build_sam2_video_predictor
    checkpoint=Path('semantic_artifacts/weights/sam2.1_hiera_tiny.pt')
    if file_hash(checkpoint)!=SAM_CHECKPOINT_SHA256:raise ValueError('SAM weight mismatch')
    folder=ROOT/'teacher';folder.mkdir(exist_ok=True)
    missing=[c for c in training_union(manifest) if not (folder/f"{c['video_id']}.npz").exists()]
    if missing:
        predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),device='cuda')
        for clip in missing:
            labels,metrics=propagate_one(predictor,dataset,clip,include_background=True)
            target=folder/f"{clip['video_id']}.npz";temp=target.with_suffix('.partial.npz')
            np.savez_compressed(temp,labels=labels,metrics=json.dumps(metrics));temp.replace(target)
            print('teacher',clip['video_id'],'positive pixels',metrics['pseudo_pixel_count'],flush=True)
        del predictor;gc.collect();torch.cuda.empty_cache()


def extract(model,clip,labels):
    query,targets,xy=sample_queries(labels,per_frame=256,seed=42+clip['video_id'])
    video=load_video(clip)[None].to('cuda',dtype=torch.bfloat16);chunks=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        memory=model.encode_video(video)
        for start in range(0,len(targets),64):
            small={k:v[:,start:start+64].to('cuda') for k,v in query.items()}
            chunks.append(extract_query_features(model,video,small,memory)['decoded'][0].to(torch.float16).cpu().numpy())
    return {'decoded':np.concatenate(chunks),'labels':targets.numpy().astype(np.int16),'xy':xy.numpy().astype(np.int16),
            't_tgt':query['t_tgt'][0].numpy(),'t_cam':query['t_cam'][0].numpy(),'video_id':np.int64(clip['video_id'])}


def caches(dataset,manifest):
    tasks=[('train',c) for c in training_union(manifest)]+[('val',c) for c in manifest['val']]+[('confirmation',c) for c in manifest['confirmation']]
    missing=[(split,c) for split,c in tasks if not (ROOT/'features'/split/f"{c['video_id']}.npz").exists()]
    if not missing:return
    model=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval()
    for split,clip in missing:
        if split=='train':
            with np.load(ROOT/'teacher'/f"{clip['video_id']}.npz") as p:labels=p['labels'].copy()
            if not np.array_equal(labels[0],dataset.gt_label_map(clip,0)):raise ValueError('Teacher source GT mismatch')
        else:labels=np.stack([dataset.gt_label_map(clip,t) for t in range(8)])
        data=extract(model,clip,labels)
        if not np.isfinite(data['decoded']).all():raise ValueError('Nonfinite features')
        target=ROOT/'features'/split/f"{clip['video_id']}.npz";target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.partial.npz')
        np.savez_compressed(temp,**data);temp.replace(target)
        print('cached',split,clip['video_id'],len(data['labels']),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


def read_clips(split,clips,source_only=False):
    result=[]
    for clip in sorted(clips,key=lambda c:str(c['video_id'])):
        with np.load(ROOT/'features'/split/f"{clip['video_id']}.npz") as p:data={k:p[k].copy() for k in ('decoded','labels','xy','video_id')}
        if source_only:
            mask=data['xy'][:,2]==0
            for k in ('decoded','labels','xy'):data[k]=data[k][mask]
        result.append(data)
    return result


def source_coverage(clips):
    counts={c:0 for c in range(1,26)}
    for clip in clips:
        source=clip['labels'][clip['xy'][:,2]==0]
        for c in np.unique(source[source>0]):counts[int(c)]+=1
    return counts


def main():
    dataset=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    manifest,protocol=freeze_protocol(dataset);teachers(dataset,manifest);caches(dataset,manifest)
    val=read_clips('val',manifest['val']);confirm=read_clips('confirmation',manifest['confirmation'])
    results={};coverage={}
    for group,group_clips in manifest['training_sets'].items():
        sam=read_clips('train',group_clips);coverage[group]=source_coverage(sam)
        if group=='coverage80' and min(coverage[group].values())<3:raise ValueError('Actual cached source queries violate registered coverage')
        for label in PROTOCOL['labels']:
            arm=f'{group}_{label}';train=sam if label=='sam_background' else read_clips('train',group_clips,source_only=True)
            x=np.concatenate([c['decoded'] for c in train]);labels=np.concatenate([c['labels'] for c in train]);times=np.concatenate([c['xy'][:,2] for c in train])
            features=torch.from_numpy(x).float().to('cuda');del x
            runs=[trajectory('linear',seed,features,labels,times,train,val,confirm,dict(protocol,arm=arm),output_root=ROOT/arm) for seed in PROTOCOL['seeds']]
            # Raw confirmation metrics are saved by the common runner but never influence selection/calibration.
            packages=[logits_for(arm,seed,'val',val,checkpoint_path=ROOT/arm/'runs'/f'linear_2000_seed{seed}.pt',output_root=ROOT/arm) for seed in PROTOCOL['seeds']]
            offset,curve=pick_offset(packages,PROTOCOL['calibration_grid']);calibrated=[]
            for seed in PROTOCOL['seeds']:
                package=logits_for(arm,seed,'confirmation',confirm,checkpoint_path=ROOT/arm/'runs'/f'linear_2000_seed{seed}.pt',output_root=ROOT/arm)
                after=evaluate(package,offset);before=evaluate(package,0.0)
                for metric in ('accuracy','macro_f1_present','foreground_macro_f1_present'):
                    if abs(before[metric]-runs[seed]['budgets']['2000']['test'][metric])>1e-12:raise ValueError('Confirmation reconstruction mismatch')
                pred=predictions(package['logits'],offset)
                per_video={str(c['video_id']):classification_metrics(pred[package['video_ids']==c['video_id']],package['labels'][package['video_ids']==c['video_id']],26) for c in manifest['confirmation']}
                calibrated.append({'seed':seed,'raw_confirmation':before,'calibrated_confirmation':after,'calibrated_validation':evaluate(packages[seed],offset),'per_video':per_video})
            results[arm]={'offset':offset,'validation_curve':curve,'runs':calibrated,
                          'trajectory_reports':[str(ROOT/arm/'runs'/f'linear_seed{seed}.json') for seed in PROTOCOL['seeds']]}
            print('completed arm',arm,'offset',offset,flush=True)
            del features
            if label=='source_gt':del train
            gc.collect();torch.cuda.empty_cache()
        del sam;gc.collect()
    contrasts={}
    pairs=[('coverage80_source_gt','random80_source_gt'),('coverage80_sam_background','random80_sam_background'),
           ('random80_sam_background','random80_source_gt'),('coverage80_sam_background','coverage80_source_gt')]
    for treatment,control in pairs:
        entry={}
        for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_accuracy'):
            d=[results[treatment]['runs'][i]['calibrated_confirmation'][metric]-results[control]['runs'][i]['calibrated_confirmation'][metric] for i in range(3)]
            entry[metric]={'paired_differences':d,'mean':float(np.mean(d)),'std':float(np.std(d,ddof=1))}
        video_differences=[float(np.mean([results[treatment]['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present']-
                          results[control]['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present'] for i in range(3)])) for c in manifest['confirmation']]
        entry['mean_per_video_macro_f1_bootstrap']=bootstrap_video_differences(video_differences)
        contrasts[treatment+'_minus_'+control]=entry
    chosen=max(results,key=lambda arm:np.mean([r['calibrated_validation']['macro_f1_present'] for r in results[arm]['runs']]))
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'manifest':manifest,'actual_source_queried_video_coverage':coverage,
                   'results':results,'contrasts':contrasts,'validation_selected_arm':chosen,
                   'verification':{'prior_140_videos_excluded':True,'coverage_target_met_in_actual_features':True,
                                   'raw_confirmation_metrics_reproduce_saved_2000_step_results':True,'no_confirmation_based_selection':True}})
    print('DONE video coverage intervention',chosen,flush=True)

if __name__=='__main__':main()
