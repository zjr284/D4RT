"""Matched-size supplemental source-GT vs propagated-query validation control."""
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_gt_exposure_control as previous
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.semantic_feature_control import infer
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/dino_supplement_budget_control')
WEIGHTS={'gt_quarter':.25,'gt_one':1.}
ARMS=('source_gt','prop_quarter','prop_one')+tuple(WEIGHTS)


def checkpoint(arm,seed):
    return previous.checkpoint(arm,seed)if arm not in WEIGHTS else ROOT/arm/'runs'/f'linear_seed{seed}.pt'


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((previous.ROOT/'summary.json').read_text());v=json.loads((previous.ROOT/'verification.json').read_text())
    assert file_hash(previous.ROOT/'summary.json')==v['summary_sha256'];m=old['manifest']
    paths=[previous.ROOT/'summary.json',previous.ROOT/'verification.json',Path('semantic/dino_supplement_budget_control.py'),Path('semantic/OVIS_DINO_SUPPLEMENT_BUDGET_PROTOCOL.md')]
    paths+=[checkpoint(a,s)for a in ARMS[:3]for s in range(3)]+[checkpoint(a,s).with_suffix('.json')for a in ARMS[:3]for s in range(3)]
    paths+=[previous.prior.feature_path('native_sam',split,c['video_id'])for split in m for c in m[split]]
    p={'round':31,'weights':WEIGHTS,'seeds':[0,1,2],'new_heads':6,'steps':2000,'parameter_count':10010,
       'batch':'256sourceGT+256extraGT; each128fg+128bg','source_seed':'s','extra_seed':'20261030+s',
       'loss':'(meanCE_source+lambda*meanCE_extra)/(1+lambda)','exposure_per_new_arm':1024000,
       'calibration_grid':old['protocol']['calibration_grid'],'calibration_selection':old['protocol']['calibration_selection'],
       'fixed_offsets':{a:old['results'][a]['offset']for a in ARMS[:3]},'validation_only':True,'deployment_change':False,
       'new_confirmation_videos':0,'shared_validation_repeatedly_used':True,'input_sha256':{str(p):file_hash(p)for p in paths}}
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=p:raise ValueError('Supplement budget protocol changed')
    atomic_json(path,p);return m,p,old


def train(arm,seed,source,val,protocol):
    path=checkpoint(arm,seed);rp=path.with_suffix('.json')
    if rp.exists():
        r=json.loads(rp.read_text());assert r['protocol']==protocol and r['checkpoint_sha256']==file_hash(path);return r
    torch.manual_seed(seed);model=SemanticHead(hidden_dim=384,num_classes=26).cuda()
    features=torch.from_numpy(np.concatenate([c['decoded']for c in source])).float().cuda()
    sampler=Sampler(np.concatenate([c['labels']for c in source]),np.concatenate([c['xy'][:,2]for c in source]),'baseline')
    generators=[torch.Generator(device='cuda').manual_seed(seed),torch.Generator(device='cuda').manual_seed(20261030+seed)]
    hist=[torch.zeros(26,dtype=torch.long,device='cuda')for _ in range(2)];ih=[hashlib.sha256(),hashlib.sha256()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);best=-1;history=[]
    for step in range(1,2001):
        indices=[]
        for j in range(2):
            fg,bg=sampler.draw(generators[j]);ix=torch.cat([fg,bg]);indices.append(ix);ih[j].update(ix.cpu().numpy().tobytes())
            hist[j]+=torch.bincount(sampler.labels[ix],minlength=26)
        optimizer.zero_grad(set_to_none=True)
        losses=[F.cross_entropy(model(features[indices[j]]),sampler.labels[indices[j]])for j in range(2)]
        loss=(losses[0]+WEIGHTS[arm]*losses[1])/(1+WEIGHTS[arm]);loss.backward();optimizer.step()
        if step%50==0:
            value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach()),'source_loss':float(losses[0].detach()),'extra_loss':float(losses[1].detach())})
            if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    old=json.loads(previous.checkpoint('prop_one',seed).with_suffix('.json').read_text())
    assert hist[0].cpu().tolist()==old['source_class_counts']and ih[0].hexdigest()==old['source_indices_sha256']
    assert all(int(h.sum())==512000 and int(h[0])==256000 for h in hist)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.pt')
    torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':protocol},tmp);tmp.replace(path)
    r={'arm':arm,'seed':seed,'protocol':protocol,'history':history,'best_step':best_step,'best_val':best,'parameter_count':10010,
       'source_class_counts':hist[0].cpu().tolist(),'extra_class_counts':hist[1].cpu().tolist(),
       'source_time_counts':[512000,0,0,0,0,0,0,0],'extra_time_counts':[512000,0,0,0,0,0,0,0],
       'source_indices_sha256':ih[0].hexdigest(),'extra_indices_sha256':ih[1].hexdigest(),'checkpoint_sha256':file_hash(path)}
    atomic_json(rp,r);del model,features;gc.collect();torch.cuda.empty_cache();print('trained extraGT',arm,seed,best_step,best,flush=True);return r


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed supplement budget exists')
    m,p,old=freeze();source=previous.prior.read('native_gt','train',m);val=previous.prior.read('native_sam','val',m)
    assert sum(len(c['labels'])for c in source)==20480
    reports={a:[train(a,s,source,val,dict(p,arm=a))for s in range(3)]for a in WEIGHTS};del source
    offsets=dict(p['fixed_offsets']);curves={};results={}
    for arm in ARMS:
        packages=[infer('dinov2',s,'val',val,checkpoint_path=checkpoint(arm,s),output_root=ROOT/arm)for s in range(3)]
        if arm in WEIGHTS:offsets[arm],curves[arm]=pick_offset(packages,p['calibration_grid'])
        runs=[{'seed':s,'raw_validation':evaluate(packages[s],0),'calibrated_validation':evaluate(packages[s],offsets[arm])}for s in range(3)]
        if arm not in WEIGHTS:
            for s in range(3):assert runs[s]['calibrated_validation']==old['results'][arm]['runs'][s]['calibrated_validation']
        results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))}
        print('validation only supplement budget',arm,offsets[arm],results[arm]['mean_calibrated_validation_macro'],flush=True)
    selected=max(ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro'])
    contrasts={}
    for lhs,rhs in [('prop_quarter','gt_quarter'),('prop_one','gt_one')]+[(a,'source_gt')for a in ARMS[1:]]:
        d=[results[lhs]['runs'][i]['calibrated_validation']['macro_f1_present']-results[rhs]['runs'][i]['calibrated_validation']['macro_f1_present']for i in range(3)]
        contrasts[lhs+'_minus_'+rhs]={'per_seed':d,'mean':float(np.mean(d))}
    selection={'provisional_validation_choice':selected,'offsets':offsets,'curves':curves,'deployment_unchanged':'native_gt',
        'checkpoint_sha256':{f'{a}_{s}':file_hash(checkpoint(a,s))for a in ARMS for s in range(3)}}
    atomic_json(ROOT/'frozen_selection.json',selection)
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'manifest':m,'training_reports':reports,'results':results,'contrasts':contrasts,
        'provisional_validation_choice':selected,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),
        'verification':{'base_source_GT_indices_and_counts_match_previous':True,'query_budget_matches_propagation_arms':True,'only_six_new_heads':True,
        'no_new_confirmation_or_deployment_change':True,'all_old_inputs_unchanged':True}})
    print('DONE matched supplement budget',selected,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
