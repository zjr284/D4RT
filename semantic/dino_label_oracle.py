"""Diagnostic label correction with fixed query positions and teacher sampling."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_resolution_sam_control as prior
from semantic.semantic_feature_control import infer,save_npz
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.ytvis import load_dataset
from semantic.model import SemanticHead
from semantic.background_calibration import evaluate,predictions,pick_offset
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics

ROOT=Path('semantic_artifacts/dino_label_oracle')
ARMS=('448_gt','448_sam','448_oracle')
PROTOCOL={'version':1,'round':21,'analysis':'diagnostic oracle, extra training GT, already observed confirmation, not new generalization',
    'arms':list(ARMS),'seeds':[0,1,2],'new_training':'only three oracle heads','steps':2000,'optimizer':'AdamW lr.001 wd.01',
    'sampling':'original SAM label128fg+128bg pools and original generator, replace only loss targets',
    'actual_target_balance':'not guaranteed; separately report corrected target histogram','query_count':162560,
    'checkpoint_selection':'raw val macro F1 every50, ties earlier','head_parameters':10010,
    'calibration_grid':prior.PROTOCOL['calibration_grid'],'calibration_selection':prior.PROTOCOL['calibration_selection'],
    'deployment_candidate':False,'new_confirmation_videos':0,'shared_validation_repeatedly_used':True}


def checkpoint(arm,seed):
    if arm!='448_oracle':return prior.checkpoint(arm,seed)
    return ROOT/'runs'/f'linear_seed{seed}.pt'


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((prior.ROOT/'summary.json').read_text());verified=json.loads((prior.ROOT/'verification.json').read_text())
    assert file_hash(prior.ROOT/'summary.json')==verified['summary_sha256']
    manifest=old['manifest'];paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',Path('semantic_artifacts/ovis/annotations_train.json')]
    paths+=[checkpoint(a,s)for a in ('448_gt','448_sam')for s in range(3)]
    paths+=[checkpoint('448_sam',s).with_suffix('.json')for s in range(3)]
    paths+=[prior.feature_path('448_sam',split,c['video_id'])for split in ('train','val','confirmation')for c in manifest[split]]
    paths+=[Path('semantic/dino_label_oracle.py'),Path('semantic/OVIS_DINO_LABEL_ORACLE_PROTOCOL.md')]
    protocol=dict(PROTOCOL,input_sha256={str(p):file_hash(p)for p in paths},inherited_protocol=old['protocol'],
                  fixed_offsets={a:old['results'][a]['offset']for a in ('448_gt','448_sam')})
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=protocol:raise ValueError('Frozen oracle protocol changed')
    atomic_json(path,protocol);return manifest,protocol


def cache_labels(ds,manifest):
    counts=np.zeros((8,26,26),dtype=np.int64)
    for clip in manifest['train']:
        path=ROOT/'labels'/f"{clip['video_id']}.npz"
        with np.load(prior.feature_path('448_sam','train',clip['video_id']))as p:original=p['labels'];xy=p['xy']
        maps=np.stack([ds.gt_label_map(clip,t)for t in range(8)])
        truth=maps[xy[:,2],xy[:,1],xy[:,0]].astype(np.int16)
        assert np.array_equal(original[xy[:,2]==0],truth[xy[:,2]==0])
        if path.exists():
            with np.load(path)as p:assert np.array_equal(p['corrected_labels'],truth)and np.array_equal(p['original_labels'],original)and np.array_equal(p['xy'],xy)
        else:save_npz(path,corrected_labels=truth,original_labels=original,xy=xy)
        np.add.at(counts,(xy[:,2],original,truth),1)
    hist=counts.sum(0);audit={'confusion_original_to_GT_by_source_time':counts.tolist(),'confusion_original_to_GT':hist.tolist(),
        'query_count':int(hist.sum()),'changed_labels':int(hist.sum()-hist.trace()),'source_frame_changes':int(counts[0].sum()-counts[0].trace()),
        'later_queries':int(counts[1:].sum()),'later_changes':int(counts[1:].sum()-sum(m.trace()for m in counts[1:])),
        'teacher_bg_but_true_fg':int(hist[0,1:].sum()),'teacher_fg_but_true_bg':int(hist[1:,0].sum()),
        'teacher_fg_wrong_fg_class':int(hist[1:,1:].sum()-hist[1:,1:].trace())}
    atomic_json(ROOT/'label_audit.json',audit);return audit


def read(arm,split,manifest):
    clips=prior.read('448_sam'if arm=='448_oracle'else arm,split,manifest)
    if arm=='448_oracle'and split=='train':
        for c in clips:
            with np.load(ROOT/'labels'/f"{int(c['video_id'])}.npz")as p:
                assert np.array_equal(c['labels'],p['original_labels'])and np.array_equal(c['xy'],p['xy'])
                c['teacher_labels']=c['labels'];c['labels']=p['corrected_labels'].copy()
    return clips


def train(seed,clips,val,protocol):
    path=checkpoint('448_oracle',seed);report_path=path.with_suffix('.json')
    if report_path.exists():
        report=json.loads(report_path.read_text());assert report['protocol']==protocol and report['checkpoint_sha256']==file_hash(path)
        return report
    torch.manual_seed(seed);model=SemanticHead(hidden_dim=384,num_classes=26).cuda()
    features=torch.from_numpy(np.concatenate([c['decoded']for c in clips])).float().cuda()
    target=torch.from_numpy(np.concatenate([c['labels']for c in clips]).astype(np.int64)).cuda()
    teacher=np.concatenate([c['teacher_labels']for c in clips]);times=np.concatenate([c['xy'][:,2]for c in clips])
    sampler=Sampler(teacher,times,'baseline');generator=torch.Generator(device='cuda').manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best=-1;history=[];old_hist=torch.zeros(26,dtype=torch.long,device='cuda');new_hist=old_hist.clone();index_hash=hashlib.sha256()
    for step in range(1,2001):
        fg,bg=sampler.draw(generator);idx=torch.cat([fg,bg]);index_hash.update(idx.cpu().numpy().tobytes())
        old_hist+=torch.bincount(sampler.labels[idx],minlength=26);new_hist+=torch.bincount(target[idx],minlength=26)
        optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(features[idx]),target[idx]);loss.backward();optimizer.step()
        if step%50==0:
            value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach())})
            if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    baseline=json.loads(checkpoint('448_sam',seed).with_suffix('.json').read_text())
    assert old_hist.cpu().tolist()==baseline['sampled_class_counts']and int(new_hist.sum())==512000
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial.pt')
    torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':protocol},temp);temp.replace(path)
    report={'protocol':protocol,'seed':seed,'history':history,'best_step':best_step,'best_val':best,
        'sampled_teacher_class_counts':old_hist.cpu().tolist(),'sampled_corrected_target_counts':new_hist.cpu().tolist(),
        'sampled_indices_sha256':index_hash.hexdigest(),'parameter_count':10010,'checkpoint_sha256':file_hash(path)}
    atomic_json(report_path,report);del model,features,target;gc.collect();torch.cuda.empty_cache()
    print('trained oracle',seed,best_step,best,flush=True);return report


def infer_arm(arm,seed,split,clips):
    return infer('dinov2',seed,split,clips,checkpoint_path=checkpoint(arm,seed),output_root=ROOT/arm)


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed oracle exists; verify rather than rerun')
    manifest,protocol=freeze();ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    audit=cache_labels(ds,manifest);label_hashes={str(p):file_hash(p)for p in sorted((ROOT/'labels').glob('*.npz'))}
    training_protocol=dict(protocol,corrected_label_sha256=label_hashes)
    clips=read('448_oracle','train',manifest);val=read('448_oracle','val',manifest)
    for seed in range(3):train(seed,clips,val,training_protocol)
    del clips,val
    validation={};offsets=dict(protocol['fixed_offsets']);curves={}
    old=json.loads((prior.ROOT/'summary.json').read_text())
    for arm in ARMS:
        clips=read(arm,'val',manifest);validation[arm]=[infer_arm(arm,s,'val',clips)for s in range(3)];del clips
        if arm=='448_oracle':offsets[arm],curves[arm]=pick_offset(validation[arm],protocol['calibration_grid'])
        else:
            for s in range(3):assert evaluate(validation[arm][s],offsets[arm])==old['results'][arm]['runs'][s]['calibrated_validation']
    selection={'offsets':offsets,'validation_curves':curves,'deployment_candidate':False,
               'checkpoint_sha256':{f'{a}_{s}':file_hash(checkpoint(a,s))for a in ARMS for s in range(3)}}
    path=ROOT/'frozen_selection.json'
    if path.exists()and json.loads(path.read_text())!=selection:raise ValueError('Oracle selection changed')
    atomic_json(path,selection);results={}
    for arm in ARMS:
        clips=read(arm,'confirmation',manifest);runs=[]
        for s in range(3):
            p=infer_arm(arm,s,'confirmation',clips);pred=predictions(p['logits'],offsets[arm])
            metric=evaluate(p,offsets[arm]);per_video={str(c['video_id']):classification_metrics(pred[p['video_ids']==c['video_id']],p['labels'][p['video_ids']==c['video_id']],26)for c in manifest['confirmation']}
            if arm!='448_oracle':assert metric==old['results'][arm]['runs'][s]['calibrated_confirmation']
            runs.append({'seed':s,'raw_validation':evaluate(validation[arm][s],0),'calibrated_validation':evaluate(validation[arm][s],offsets[arm]),
                'raw_confirmation':evaluate(p,0),'calibrated_confirmation':metric,'per_video':per_video})
            print('descriptive',arm,s,metric['macro_f1_present'],flush=True)
        results[arm]={'offset':offsets[arm],'runs':runs};del clips
    contrasts={name:prior.comparison(terms,results,manifest)for name,terms in {'oracle_minus_SAM':[('448_oracle',1),('448_sam',-1)],'oracle_minus_GT':[('448_oracle',1),('448_gt',-1)]}.items()}
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    assert label_hashes=={str(p):file_hash(p)for p in sorted((ROOT/'labels').glob('*.npz'))}
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'manifest':manifest,'corrected_label_sha256':label_hashes,'label_audit':audit,'results':results,'contrasts':contrasts,
        'frozen_selection_sha256':file_hash(path),'deployment_selected_arm_unchanged':'448_sam',
        'verification':{'only_three_oracle_heads_trained':True,'old_baseline_validation_and_confirmation_exactly_reproduced':True,
                        'teacher_sampling_counts_identical':True,'no_new_confirmation_and_no_deployment_change':True}})
    print('DONE oracle diagnostic',audit['changed_labels'],contrasts['oracle_minus_SAM']['macro_f1_present'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
