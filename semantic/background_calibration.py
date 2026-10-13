"""Validation-only shared background logit calibration of fixed weak-label models."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from semantic.model import SemanticHead
from semantic.oracle_probe import file_hash
from semantic.weaklabel_budget_probe import read_split,CACHES
from semantic.run import atomic_json,classification_metrics

ROOT=Path('semantic_artifacts/background_calibration')
ARMS=('positive_only','positive_background')
PROTOCOL={'version':1,'arms':list(ARMS),'head':'linear','training_budget':2000,'seeds':[0,1,2],
          'offset_grid':[float(v) for v in np.arange(-8.0,8.001,0.25)],'offset_applied_to':'background logit only',
          'offset_selection':'one shared offset per label arm, maximize mean validation macro F1 over three seeds',
          'ties':'prefer smallest absolute offset, then negative offset','train_updates':0,
          'test_never_selects_offset':True,'labels_features_checkpoints_unchanged':True}


def predictions(logits,offset):
    if logits.ndim!=2 or logits.shape[1]!=26 or not np.isfinite(logits).all():raise ValueError('Need finite Nx26 logits')
    if not np.isfinite(offset):raise ValueError('Offset must be finite')
    foreground=logits[:,1:].argmax(axis=1)+1
    return np.where(logits[:,0]+offset>=logits[np.arange(len(logits)),foreground],0,foreground)


def pick_offset(validation_packages,grid):
    if not validation_packages or 0.0 not in grid:raise ValueError('Need validation data and zero offset')
    scores=[];best=-1.0;selected=None
    for offset in sorted(grid,key=lambda d:(abs(d),d)):
        per_seed=[classification_metrics(predictions(p['logits'],offset),p['labels'],26)['macro_f1_present'] for p in validation_packages]
        mean=float(np.mean(per_seed));scores.append({'offset':float(offset),'macro_f1_by_seed':per_seed,'mean_macro_f1':mean})
        if mean>best:best=mean;selected=float(offset)
    return selected,scores


def logits_for(arm,seed,split,clips, *, checkpoint_path=None, output_root=None):
    output_root=ROOT if output_root is None else Path(output_root)
    cache=output_root/'logits'/f'{arm}_{seed}_{split}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        with np.load(cache) as p:return {k:p[k].copy() for k in p.files}
    checkpoint=Path('semantic_artifacts/weaklabel_budget_probe')/arm/'runs'/f'linear_2000_seed{seed}.pt' if checkpoint_path is None else Path(checkpoint_path)
    model=SemanticHead(num_classes=26).to('cuda').eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    values=[];labels=[];times=[];videos=[]
    with torch.inference_mode():
        for c in clips:
            values.append(model(torch.from_numpy(c['decoded']).float().to('cuda')).cpu().numpy())
            labels.append(c['labels']);times.append(c['xy'][:,2]);videos.append(np.full(len(c['labels']),int(c['video_id']),dtype=np.int32))
    result={'logits':np.concatenate(values),'labels':np.concatenate(labels),'times':np.concatenate(times),'video_ids':np.concatenate(videos)}
    temp=cache.with_suffix('.partial.npz');np.savez_compressed(temp,**result);temp.replace(cache)
    return result


def evaluate(package,offset):
    y=package['labels'];pred=predictions(package['logits'],offset)
    result=classification_metrics(pred,y,26)
    pos=y>0
    result['foreground_accuracy']=float((pred[pos]==y[pos]).mean())
    result['foreground_to_background_rate']=float((pred[pos]==0).mean())
    result['predicted_class_counts']=np.bincount(pred,minlength=26).tolist()
    result['per_frame']={str(t):classification_metrics(pred[package['times']==t],y[package['times']==t],26) for t in range(8)}
    matrix=np.zeros((26,26),dtype=np.int64);np.add.at(matrix,(y,pred),1);result['confusion_matrix']=matrix.tolist()
    return result


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    checkpoints={f'{arm}_{seed}':file_hash(Path('semantic_artifacts/weaklabel_budget_probe')/arm/'runs'/f'linear_2000_seed{seed}.pt') for arm in ARMS for seed in PROTOCOL['seeds']}
    hashes={}
    for split in ('val','test'):
        h=hashlib.sha256()
        for p in sorted((CACHES['positive_only']/split).glob('*.npz')):h.update(p.name.encode());h.update(bytes.fromhex(file_hash(p)))
        hashes[split]=h.hexdigest()
    protocol=dict(PROTOCOL,checkpoint_sha256=checkpoints,heldout_cache_sha256=hashes,
                  previous_summary_sha256=file_hash('semantic_artifacts/weaklabel_budget_probe/summary.json'))
    file=ROOT/'protocol.json'
    if file.exists() and json.loads(file.read_text())!=protocol:raise ValueError('Protocol/inputs changed')
    atomic_json(file,protocol)
    val=read_split(CACHES['positive_only'],'val');test=read_split(CACHES['positive_only'],'test')
    if len(val)!=20 or len(test)!=20:raise ValueError('Need full held-out splits')
    if {int(c['video_id']) for c in val}&{int(c['video_id']) for c in test}:raise ValueError('Held-out overlap')
    manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text())
    for split,clips in [('val',val),('test',test)]:
        if {int(c['video_id']) for c in clips}!={int(c['video_id']) for c in manifest[split]}:raise ValueError('Manifest mismatch')
    results={};prior=json.loads(Path('semantic_artifacts/weaklabel_budget_probe/summary.json').read_text())
    for arm in ARMS:
        packages=[logits_for(arm,seed,'val',val) for seed in PROTOCOL['seeds']]
        offset,curve=pick_offset(packages,PROTOCOL['offset_grid'])
        # Only after fixing this arm's offset do we evaluate its test packages.
        runs=[]
        for seed in PROTOCOL['seeds']:
            package=logits_for(arm,seed,'test',test)
            before=evaluate(package,0.0);after=evaluate(package,offset)
            for metric in ('accuracy','macro_f1_present','foreground_macro_f1_present'):
                if abs(before[metric]-prior['results'][f'{arm}_linear_2000'][seed]['test'][metric])>1e-12:raise ValueError('Uncalibrated baseline mismatch')
            runs.append({'seed':seed,'before_test':before,'after_test':after,
                         'before_validation':evaluate(packages[seed],0.0),'after_validation':evaluate(packages[seed],offset)})
        deltas={}
        for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_accuracy','foreground_to_background_rate'):
            d=[r['after_test'][metric]-r['before_test'][metric] for r in runs]
            deltas[metric]={'paired_deltas':d,'mean':float(np.mean(d)),'std':float(np.std(d,ddof=1))}
        results[arm]={'selected_offset':offset,'validation_curve':curve,'runs':runs,'paired_changes':deltas}
        print(arm,'offset',offset,'macro F1 change',deltas['macro_f1_present']['mean'],flush=True)
    for arm in ARMS:
        for seed in PROTOCOL['seeds']:
            checkpoint=Path('semantic_artifacts/weaklabel_budget_probe')/arm/'runs'/f'linear_2000_seed{seed}.pt'
            if file_hash(checkpoint)!=checkpoints[f'{arm}_{seed}']:raise ValueError('Checkpoint changed')
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'results':results,
                                  'verification':{'all_six_uncalibrated_baselines_match':True,'checkpoint_hashes_unchanged':True,
                                                  'shared_offsets_selected_from_validation_only':True}})
    print('DONE calibration',flush=True)

if __name__=='__main__':main()
