"""Round20 frozen resolution x SAM control; only three new source-GT heads."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import numpy as np
import torch
from semantic import dino_resolution_control as resolution
from semantic import dino_sam_control as prior
from semantic.semantic_feature_control import (read_features, train_dino, infer, aggregate_hash,
    save_npz, patch_at_queries, PROTOCOL as FEATURE_PROTOCOL)
from semantic.ytvis import load_dataset, load_video, sample_queries
from semantic.fresh_video_confirmation import select_clips, bootstrap_video_differences
from semantic.background_calibration import evaluate, predictions, pick_offset
from semantic.run import atomic_json, classification_metrics
from semantic.oracle_probe import file_hash

ROOT=Path('semantic_artifacts/dino_resolution_sam_control')
ARMS=('224_gt','224_sam','448_gt','448_sam')
OFFSETS={'224_gt':-1.5,'224_sam':-1.75,'448_sam':-1.5}
PROTOCOL={'round':20,'version':1,'arms':list(ARMS),'seeds':[0,1,2],'confirmation_seed':20261020,
    'training_videos':80,'validation_videos':20,'confirmation_videos':40,'feature_dims':384,
    'new_training':'only three448 GT heads','steps':2000,'optimizer':'AdamW lr.001 wd.01',
    'sampling':'128 foreground +128 background uniform','head_parameters':10010,
    'checkpoint_selection':'raw validation macro F1 every50, ties earlier',
    'calibration_grid':FEATURE_PROTOCOL['calibration_grid'],'calibration_selection':FEATURE_PROTOCOL['calibration_selection'],
    'fixed_offsets':OFFSETS,'source_metadata_previously_inspected':True,'shared_validation_repeatedly_used':True,
    'confirmation_after_selection_freeze':True,'pretraining_ovis_overlap':'unknown','bootstrap_seed':20261020,'bootstrap_repetitions':2000}


def checkpoint(arm,seed):
    if arm=='224_gt':return prior.checkpoint('coverage80_source_gt',seed)
    if arm=='224_sam':return prior.checkpoint('coverage80_sam_background',seed)
    if arm=='448_sam':return resolution.checkpoint('resolution448',seed)
    return ROOT/arm/'dinov2'/'runs'/f'linear_seed{seed}.pt'


def feature_path(arm,split,vid):
    size=arm.split('_')[0]
    if split=='confirmation':return ROOT/'features'/size/split/f'{vid}.npz'
    return resolution.feature_path(f'resolution{size}',split,vid)


def freeze(dataset):
    ROOT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((resolution.ROOT/'summary.json').read_text());verified=json.loads((resolution.ROOT/'verification.json').read_text())
    oldsam=json.loads((prior.ROOT/'summary.json').read_text());samverified=json.loads((prior.ROOT/'verification.json').read_text())
    assert file_hash(resolution.ROOT/'summary.json')==verified['summary_sha256']
    assert file_hash(prior.ROOT/'summary.json')==samverified['summary_sha256']
    old=previous['manifest'];excluded=set(old['excluded_video_ids'])|{c['video_id']for c in old['confirmation']};assert len(excluded)==460
    clips,count=select_clips(dataset,excluded,count=40,seed=20261020);assert count==147
    manifest={'seed':20261020,'excluded_video_ids':sorted(excluded),'eligible_model_unevaluated':count,
              'train':old['train'],'val':old['val'],'confirmation':clips,'source_category_metadata_previously_inspected':True}
    path=ROOT/'manifest.json'
    if path.exists()and json.loads(path.read_text())!=manifest:raise ValueError('Frozen manifest changed')
    atomic_json(path,manifest)
    paths=[resolution.ROOT/'summary.json',resolution.ROOT/'verification.json',prior.ROOT/'summary.json',prior.ROOT/'verification.json',path]
    paths+=[Path('semantic/dino_resolution_sam_control.py'),Path('semantic/OVIS_DINO_RESOLUTION_SAM_PROTOCOL.md'),resolution.WEIGHTS]
    paths+=[checkpoint(a,s)for a in OFFSETS for s in range(3)]
    paths+=[checkpoint(a,s).with_suffix('.json')for a in OFFSETS for s in range(3)]
    paths+=[feature_path(a,split,c['video_id'])for a in ('224_sam','448_sam')for split in ('train','val')for c in manifest[split]]
    paths+=[prior.coverage.PREVIOUS/'teacher'/f"{c['video_id']}.npz"for c in manifest['train']]
    paths+=[Path(frame)for split in manifest if split in ('train','val','confirmation')for c in manifest[split]for frame in c['frames']]
    for seed in range(3):
        assert file_hash(checkpoint('448_sam',seed))==verified['checkpoint_sha256'][f'resolution448_{seed}']
        for arm,label in [('224_gt','coverage80_source_gt'),('224_sam','coverage80_sam_background')]:
            assert file_hash(checkpoint(arm,seed))==samverified['checkpoint_sha256'][f'{label}_{seed}']
    protocol=dict(PROTOCOL,manifest_sha256=file_hash(path),input_sha256={str(p):file_hash(p)for p in paths},inherited_protocol=previous['protocol'])
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=protocol:raise ValueError('Frozen protocol changed')
    atomic_json(path,protocol);return manifest,protocol


def read(arm,split,manifest):
    clips=read_features('dinov2',split,manifest,path_resolver=lambda _,s,v:feature_path(arm,s,v))
    if split=='train'and arm.endswith('_gt'):
        for c in clips:
            mask=c['xy'][:,2]==0
            for key in ('decoded','labels','xy','t_tgt','t_cam'):c[key]=c[key][mask]
    return clips


def feature_hashes(manifest,splits):
    return {f'{size}_{split}':aggregate_hash([feature_path(size+'_sam',split,c['video_id'])for c in manifest[split]])for size in ('224','448')for split in splits}


def infer_arm(arm,seed,split,clips):
    return infer('dinov2',seed,split,clips,checkpoint_path=checkpoint(arm,seed),output_root=ROOT/arm)


def cache_confirmation(dataset,manifest):
    assert (ROOT/'frozen_selection.json').exists()
    model=resolution.load_dino()
    for clip in manifest['confirmation']:
        maps=np.stack([dataset.gt_label_map(clip,t)for t in range(8)])
        q,y,xy=sample_queries(maps,per_frame=256,seed=42+clip['video_id'])
        meta={'labels':y.numpy().astype(np.int16),'xy':xy.numpy().astype(np.int16),
              't_tgt':q['t_tgt'][0].numpy(),'t_cam':q['t_cam'][0].numpy(),'video_id':np.int64(clip['video_id'])}
        video=load_video(clip).cuda()
        for size in (224,448):
            path=feature_path(f'{size}_sam','confirmation',clip['video_id'])
            if path.exists():continue
            tokens=resolution.extract_tokens(model,video,size)
            with torch.inference_mode():features=patch_at_queries(tokens,meta['xy']).half().cpu().numpy()
            assert np.isfinite(features).all();save_npz(path,**meta,decoded=features)
            print('cached',size,clip['video_id'],flush=True);del tokens
        del video
    del model;gc.collect();torch.cuda.empty_cache()


def comparison(terms,results,manifest):
    result={}
    for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_accuracy'):
        d=[sum(sign*results[arm]['runs'][s]['calibrated_confirmation'][metric]for arm,sign in terms)for s in range(3)]
        result[metric]={'paired_differences':d,'mean':float(np.mean(d)),'std':float(np.std(d,ddof=1))}
    d={str(c['video_id']):float(np.mean([sum(sign*results[arm]['runs'][s]['per_video'][str(c['video_id'])]['macro_f1_present']for arm,sign in terms)for s in range(3)]))for c in manifest['confirmation']}
    result['per_video_differences']=d;result['video_bootstrap']=bootstrap_video_differences(list(d.values()),seed=20261020)
    return result


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed round20 exists; verify rather than rerun')
    ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    manifest,protocol=freeze(ds);hashes=feature_hashes(manifest,('train','val'))
    train=read('448_gt','train',manifest);val=read('448_gt','val',manifest)
    assert sum(len(c['labels'])for c in train)==20480
    for seed in range(3):
        report=train_dino(seed,train,val,dict(protocol,arm='448_gt',feature_sha256=hashes),output_root=ROOT/'448_gt')
        old=json.loads(checkpoint('224_gt',seed).with_suffix('.json').read_text())
        assert report['sampled_class_counts']==old['sampled_class_counts']and report['parameter_count']==10010
    del train,val
    validation={};offsets=dict(OFFSETS);curves={}
    previous=json.loads((resolution.ROOT/'summary.json').read_text());oldsam=json.loads((prior.ROOT/'summary.json').read_text())
    for arm in ARMS:
        clips=read(arm,'val',manifest);validation[arm]=[infer_arm(arm,s,'val',clips)for s in range(3)];del clips
        if arm=='448_gt':offsets[arm],curves[arm]=pick_offset(validation[arm],protocol['calibration_grid'])
        else:
            saved=previous['results']['resolution448']if arm=='448_sam'else oldsam['results']['coverage80_source_gt'if arm=='224_gt'else'coverage80_sam_background']
            for s in range(3):assert evaluate(validation[arm][s],offsets[arm])==saved['runs'][s]['calibrated_validation']
    scores={a:float(np.mean([evaluate(p,offsets[a])['macro_f1_present']for p in validation[a]]))for a in ARMS}
    selected=max(ARMS,key=lambda a:scores[a])
    selection={'offsets':offsets,'validation_curves':curves,'mean_validation_macro_f1':scores,'validation_selected_arm':selected,
               'checkpoint_sha256':{f'{a}_{s}':file_hash(checkpoint(a,s))for a in ARMS for s in range(3)}}
    path=ROOT/'frozen_selection.json'
    if path.exists()and json.loads(path.read_text())!=selection:raise ValueError('Selection changed')
    atomic_json(path,selection);print('FROZEN',selected,scores,offsets,flush=True)
    cache_confirmation(ds,manifest);results={}
    for arm in ARMS:
        clips=read(arm,'confirmation',manifest);runs=[]
        for s in range(3):
            p=infer_arm(arm,s,'confirmation',clips);pred=predictions(p['logits'],offsets[arm])
            per_video={str(c['video_id']):classification_metrics(pred[p['video_ids']==c['video_id']],p['labels'][p['video_ids']==c['video_id']],26)for c in manifest['confirmation']}
            runs.append({'seed':s,'raw_validation':evaluate(validation[arm][s],0),'calibrated_validation':evaluate(validation[arm][s],offsets[arm]),
                         'raw_confirmation':evaluate(p,0),'calibrated_confirmation':evaluate(p,offsets[arm]),'per_video':per_video})
            print('confirmed',arm,s,runs[-1]['calibrated_confirmation']['macro_f1_present'],flush=True)
        results[arm]={'offset':offsets[arm],'runs':runs};del clips
    terms={'SAM_at448':[('448_sam',1),('448_gt',-1)],'SAM_at224':[('224_sam',1),('224_gt',-1)],
           'resolution_GT':[('448_gt',1),('224_gt',-1)],'resolution_SAM':[('448_sam',1),('224_sam',-1)],
           'interaction':[('448_sam',1),('448_gt',-1),('224_sam',-1),('224_gt',1)]}
    contrasts={name:comparison(t,results,manifest)for name,t in terms.items()}
    assert feature_hashes(manifest,('train','val'))==hashes
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    assert selection['checkpoint_sha256']=={f'{a}_{s}':file_hash(checkpoint(a,s))for a in ARMS for s in range(3)}
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'manifest':manifest,'feature_sha256':feature_hashes(manifest,('train','val','confirmation')),
        'frozen_selection_sha256':file_hash(path),'validation_selected_arm':selected,'results':results,'contrasts':contrasts,
        'verification':{'only_three448_gt_heads_trained':True,'sampling_matches224_GT':True,'nine_baseline_heads_offsets_unchanged':True,
                        'confirmation_excludes_prior460':True,'all_confirmation_outputs_after_selection_freeze':True}})
    print('DONE round20 SAM448',contrasts['SAM_at448']['macro_f1_present'],contrasts['SAM_at448']['video_bootstrap'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
