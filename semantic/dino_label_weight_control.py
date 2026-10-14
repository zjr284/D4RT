"""Validation-only propagation-label weighting with fixed teacher sampler."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_native_rgb_control as prior
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.semantic_feature_control import infer
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/dino_label_weight_control')
WEIGHTS={'lambda0':0.,'quarter':.25,'half':.5,'one':1.}
ARMS=tuple(WEIGHTS)+('source_gt',)


def checkpoint(arm,seed):
    if arm in ('one','source_gt'):return prior.checkpoint('native_sam'if arm=='one'else'native_gt',seed)
    return ROOT/arm/'runs'/f'linear_seed{seed}.pt'


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text())
    assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256']
    manifest={s:old['manifest'][s]for s in ('train','val')}
    paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',Path('semantic/dino_label_weight_control.py'),Path('semantic/OVIS_DINO_LABEL_WEIGHT_PROTOCOL.md')]
    paths+=[checkpoint(a,s)for a in ('one','source_gt')for s in range(3)]
    paths+=[checkpoint(a,s).with_suffix('.json')for a in ('one','source_gt')for s in range(3)]
    paths+=[prior.feature_path('native_sam',split,c['video_id'])for split in manifest for c in manifest[split]]
    p={'round':29,'weights':WEIGHTS,'seeds':[0,1,2],'new_heads':9,'steps':2000,'parameter_count':10010,
       'sampler':'fixed original SAM pools128fg+128bg','loss':'sum(weight*per_query_CE)/sum(weight), source=1 later=lambda',
       'checkpoint':'raw validation macro F1 every50, earlier on tie','optimizer':'AdamW lr.001 wd.01',
       'calibration_grid':prior.PROTOCOL['calibration_grid'],'calibration_selection':prior.PROTOCOL['calibration_selection'],
       'fixed_offsets':{'one':old['results']['native_sam']['offset'],'source_gt':old['results']['native_gt']['offset']},
       'validation_only':True,'deployment_change':False,'new_confirmation_videos':0,'shared_validation_repeatedly_used':True,
       'input_sha256':{str(p):file_hash(p)for p in paths}}
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=p:raise ValueError('Weight control frozen protocol changed')
    atomic_json(path,p);return manifest,p,old


def train(arm,seed,clips,val,protocol):
    path=checkpoint(arm,seed);rp=path.with_suffix('.json')
    if rp.exists():
        r=json.loads(rp.read_text());assert r['protocol']==protocol and r['checkpoint_sha256']==file_hash(path);return r
    torch.manual_seed(seed);model=SemanticHead(hidden_dim=384,num_classes=26).cuda()
    features=torch.from_numpy(np.concatenate([c['decoded']for c in clips])).float().cuda()
    labels=np.concatenate([c['labels']for c in clips]);times=np.concatenate([c['xy'][:,2]for c in clips])
    sampler=Sampler(labels,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed)
    weights=torch.where(sampler.times==0,1.,WEIGHTS[arm]);hist=torch.zeros(26,dtype=torch.long,device='cuda')
    source_hist=hist.clone();later_hist=hist.clone();time_hist=torch.zeros(8,dtype=torch.long,device='cuda');ih=hashlib.sha256()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best=-1;history=[];effective=0.
    for step in range(1,2001):
        fg,bg=sampler.draw(gen);ix=torch.cat([fg,bg]);ih.update(ix.cpu().numpy().tobytes())
        yy=sampler.labels[ix];tt=sampler.times[ix];ww=weights[ix]
        if float(ww.sum())==0:raise ValueError('No nonzero-weight queries in fixed batch')
        hist+=torch.bincount(yy,minlength=26);time_hist+=torch.bincount(tt,minlength=8)
        source_hist+=torch.bincount(yy[tt==0],minlength=26);later_hist+=torch.bincount(yy[tt!=0],minlength=26);effective+=float(ww.sum())
        optimizer.zero_grad(set_to_none=True);loss=(F.cross_entropy(model(features[ix]),yy,reduction='none')*ww).sum()/ww.sum()
        loss.backward();optimizer.step()
        if step%50==0:
            value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach())})
            if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    baseline=json.loads(checkpoint('one',seed).with_suffix('.json').read_text())
    assert hist.cpu().tolist()==baseline['sampled_class_counts'] and int(hist.sum())==512000
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.pt')
    torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':protocol},tmp);tmp.replace(path)
    report={'arm':arm,'seed':seed,'protocol':protocol,'history':history,'best_step':best_step,'best_val':best,
       'sampled_class_counts':hist.cpu().tolist(),'sampled_source_class_counts':source_hist.cpu().tolist(),
       'sampled_later_class_counts':later_hist.cpu().tolist(),'sampled_time_counts':time_hist.cpu().tolist(),
       'sampled_indices_sha256':ih.hexdigest(),'effective_weight_sum':effective,'parameter_count':sum(p.numel()for p in model.parameters()),
       'checkpoint_sha256':file_hash(path)}
    atomic_json(rp,report);del model,features;gc.collect();torch.cuda.empty_cache()
    print('trained weights',arm,seed,best_step,best,'source',int(source_hist.sum()),flush=True);return report


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed weight control exists; verify rather than train')
    manifest,protocol,old=freeze();clips=prior.read('native_sam','train',manifest);val=prior.read('native_sam','val',manifest)
    assert sum(len(c['labels'])for c in clips)==162560
    reports={}
    for arm in ('lambda0','quarter','half'):
        reports[arm]=[train(arm,s,clips,val,dict(protocol,arm=arm))for s in range(3)]
    del clips
    results={};offsets=dict(protocol['fixed_offsets']);curves={}
    for arm in ARMS:
        packages=[infer('dinov2',s,'val',val,checkpoint_path=checkpoint(arm,s),output_root=ROOT/arm)for s in range(3)]
        if arm not in offsets:offsets[arm],curves[arm]=pick_offset(packages,protocol['calibration_grid'])
        runs=[{'seed':s,'raw_validation':evaluate(packages[s],0),'calibrated_validation':evaluate(packages[s],offsets[arm])}for s in range(3)]
        if arm in ('one','source_gt'):
            original='native_sam'if arm=='one'else'native_gt'
            for s in range(3):assert runs[s]['calibrated_validation']==old['results'][original]['runs'][s]['calibrated_validation']
        results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))}
        print('validation only',arm,offsets[arm],results[arm]['mean_calibrated_validation_macro'],flush=True)
    selected=max(WEIGHTS,key=lambda a:results[a]['mean_calibrated_validation_macro'])
    selection={'provisional_validation_weight_choice':selected,'offsets':offsets,'curves':curves,
        'checkpoint_sha256':{f'{a}_{s}':file_hash(checkpoint(a,s))for a in ARMS for s in range(3)},'deployment_unchanged':'native_gt',
        'not_independent_generalization':True}
    atomic_json(ROOT/'frozen_selection.json',selection)
    for path,h in protocol['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'manifest':manifest,'training_reports':reports,'results':results,
        'provisional_validation_weight_choice':selected,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),
        'verification':{'nine_new_heads_only':True,'all_old_inputs_unchanged':True,'all_training_class_histograms_match_original_SAM':True,
            'all_old_validation_metrics_reproduced':True,'no_confirmation_outputs_generated':True,'deployment_unchanged':True}})
    print('DONE propagation label weight exploration',selected,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
