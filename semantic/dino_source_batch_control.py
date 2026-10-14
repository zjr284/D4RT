"""Validation-only smaller source-GT batch; preserve all frozen references."""
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_label_weight_control as weighted
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.semantic_feature_control import infer
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/dino_source_batch_control')
ARMS=('source_gt256','lambda0','source_gt32')


def checkpoint(arm,seed):
    if arm=='source_gt256':return weighted.checkpoint('source_gt',seed)
    if arm=='lambda0':return weighted.checkpoint('lambda0',seed)
    return ROOT/arm/'runs'/f'linear_seed{seed}.pt'


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((weighted.ROOT/'summary.json').read_text());v=json.loads((weighted.ROOT/'verification.json').read_text())
    assert file_hash(weighted.ROOT/'summary.json')==v['summary_sha256'];m=old['manifest']
    paths=[weighted.ROOT/'summary.json',weighted.ROOT/'verification.json',Path('semantic/dino_source_batch_control.py'),Path('semantic/OVIS_DINO_SOURCE_BATCH_PROTOCOL.md')]
    paths+=[checkpoint(a,s)for a in ARMS[:2]for s in range(3)]+[checkpoint(a,s).with_suffix('.json')for a in ARMS[:2]for s in range(3)]
    paths+=[weighted.prior.feature_path('native_sam',split,c['video_id'])for split in m for c in m[split]]
    p={'round':32,'seeds':[0,1,2],'new_heads':3,'steps':2000,'parameter_count':10010,'source_batch':32,'foreground_batch':16,'background_batch':16,
        'new_arm_exposure':64000,'optimizer':'AdamW lr.001 wd.01','checkpoint':'raw val macro every50, earlier ties',
        'calibration_grid':old['protocol']['calibration_grid'],'calibration_selection':old['protocol']['calibration_selection'],
        'fixed_offsets':{'source_gt256':old['results']['source_gt']['offset'],'lambda0':old['results']['lambda0']['offset']},
        'validation_only':True,'deployment_change':False,'new_confirmation_videos':0,'shared_validation_repeatedly_used':True,
        'input_sha256':{str(p):file_hash(p)for p in paths}}
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=p:raise ValueError('Source batch protocol changed')
    atomic_json(path,p);return m,p,old


def train(seed,clips,val,protocol):
    path=checkpoint('source_gt32',seed);rp=path.with_suffix('.json')
    if rp.exists():
        r=json.loads(rp.read_text());assert r['protocol']==protocol and r['checkpoint_sha256']==file_hash(path);return r
    torch.manual_seed(seed);model=SemanticHead(hidden_dim=384,num_classes=26).cuda()
    features=torch.from_numpy(np.concatenate([c['decoded']for c in clips])).float().cuda()
    labels=np.concatenate([c['labels']for c in clips]);times=np.concatenate([c['xy'][:,2]for c in clips]);assert np.all(times==0)
    sampler=Sampler(labels,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);hist=torch.zeros(26,dtype=torch.long,device='cuda');ih=hashlib.sha256()
    best=-1;history=[]
    for step in range(1,2001):
        fg,bg=sampler.draw(gen,count=16);ix=torch.cat([fg,bg]);ih.update(ix.cpu().numpy().tobytes());hist+=torch.bincount(sampler.labels[ix],minlength=26)
        optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(features[ix]),sampler.labels[ix]);loss.backward();optimizer.step()
        if step%50==0:
            value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach())})
            if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    assert int(hist.sum())==64000 and int(hist[0])==32000
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.pt')
    torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':protocol},tmp);tmp.replace(path)
    r={'seed':seed,'protocol':protocol,'history':history,'best_step':best_step,'best_val':best,'parameter_count':10010,
        'sampled_class_counts':hist.cpu().tolist(),'sampled_source_time_counts':[64000,0,0,0,0,0,0,0],
        'sampled_indices_sha256':ih.hexdigest(),'checkpoint_sha256':file_hash(path)}
    atomic_json(rp,r);del model,features;gc.collect();torch.cuda.empty_cache();print('trained source32',seed,best_step,best,flush=True);return r


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed source batch control exists')
    m,p,old=freeze();source=weighted.prior.read('native_gt','train',m);val=weighted.prior.read('native_sam','val',m)
    assert sum(len(c['labels'])for c in source)==20480
    reports=[train(s,source,val,p)for s in range(3)];del source
    offsets=dict(p['fixed_offsets']);curves={};results={}
    for arm in ARMS:
        packages=[infer('dinov2',s,'val',val,checkpoint_path=checkpoint(arm,s),output_root=ROOT/arm)for s in range(3)]
        if arm=='source_gt32':offsets[arm],curves[arm]=pick_offset(packages,p['calibration_grid'])
        runs=[{'seed':s,'raw_validation':evaluate(packages[s],0),'calibrated_validation':evaluate(packages[s],offsets[arm])}for s in range(3)]
        if arm!='source_gt32':
            a='source_gt'if arm=='source_gt256'else'lambda0'
            for s in range(3):assert runs[s]['calibrated_validation']==old['results'][a]['runs'][s]['calibrated_validation']
        results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))}
        print('validation only source batch',arm,offsets[arm],results[arm]['mean_calibrated_validation_macro'],flush=True)
    selected=max(ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro'])
    selection={'provisional_validation_choice':selected,'offsets':offsets,'curves':curves,'deployment_unchanged':'native_gt',
        'checkpoint_sha256':{f'{a}_{s}':file_hash(checkpoint(a,s))for a in ARMS for s in range(3)}}
    atomic_json(ROOT/'frozen_selection.json',selection)
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'manifest':m,'training_reports':reports,'results':results,
        'provisional_validation_choice':selected,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),
        'verification':{'only_three_new_heads':True,'source_time_always_zero':True,'new_exposure64000':True,'old_reference_metrics_and_hashes_unchanged':True,
        'no_new_confirmation_or_deployment_change':True}})
    print('DONE smaller source batch exploration',selected,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
