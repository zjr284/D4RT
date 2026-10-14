"""Supplement propagation targets while preserving historical source-GT draws."""
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

ROOT=Path('semantic_artifacts/dino_gt_exposure_control')
WEIGHTS={'prop_quarter':.25,'prop_one':1.}
ARMS=('source_gt',)+tuple(WEIGHTS)


def checkpoint(arm,seed):
    return prior.checkpoint('native_gt',seed)if arm=='source_gt'else ROOT/arm/'runs'/f'linear_seed{seed}.pt'


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text())
    assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256']
    manifest={s:old['manifest'][s]for s in ('train','val')}
    paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',Path('semantic/dino_gt_exposure_control.py'),Path('semantic/OVIS_DINO_GT_EXPOSURE_PROTOCOL.md')]
    paths+=[checkpoint('source_gt',s)for s in range(3)]+[checkpoint('source_gt',s).with_suffix('.json')for s in range(3)]
    paths+=[prior.feature_path('native_sam',split,c['video_id'])for split in manifest for c in manifest[split]]
    p={'round':30,'seeds':[0,1,2],'new_heads':6,'weights':WEIGHTS,'steps':2000,'parameter_count':10010,
       'batch':'256GT+256prop, each128fg+128bg','source_seed':'s','prop_seed':'20261030+s',
       'loss':'(meanCE_source+lambda*meanCE_later)/(1+lambda)','GT_exposure':512000,'total_new_arm_exposure':1024000,
       'optimizer':'AdamW lr.001 wd.01','checkpoint':'raw val macro every50, earlier ties',
       'calibration_grid':prior.PROTOCOL['calibration_grid'],'calibration_selection':prior.PROTOCOL['calibration_selection'],
       'source_gt_fixed_offset':old['results']['native_gt']['offset'],'validation_only':True,'deployment_change':False,
       'new_confirmation_videos':0,'shared_validation_repeatedly_used':True,'input_sha256':{str(p):file_hash(p)for p in paths}}
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=p:raise ValueError('GT exposure protocol changed')
    atomic_json(path,p);return manifest,p,old


def pools(manifest):
    source=prior.read('native_gt','train',manifest);later=prior.read('native_sam','train',manifest)
    for c in later:
        mask=c['xy'][:,2]>0
        for k in ('decoded','labels','xy','t_tgt','t_cam'):c[k]=c[k][mask]
    assert sum(len(c['labels'])for c in source)==20480 and sum(len(c['labels'])for c in later)==142080
    return source,later


def train(arm,seed,source,later,val,protocol):
    path=checkpoint(arm,seed);rp=path.with_suffix('.json')
    if rp.exists():
        r=json.loads(rp.read_text());assert r['protocol']==protocol and r['checkpoint_sha256']==file_hash(path);return r
    torch.manual_seed(seed);model=SemanticHead(hidden_dim=384,num_classes=26).cuda()
    fs=[torch.from_numpy(np.concatenate([c['decoded']for c in clips])).float().cuda()for clips in (source,later)]
    samplers=[Sampler(np.concatenate([c['labels']for c in clips]),np.concatenate([c['xy'][:,2]for c in clips]),'baseline')for clips in (source,later)]
    generators=[torch.Generator(device='cuda').manual_seed(seed),torch.Generator(device='cuda').manual_seed(20261030+seed)]
    hist=[torch.zeros(26,dtype=torch.long,device='cuda')for _ in range(2)];th=[torch.zeros(8,dtype=torch.long,device='cuda')for _ in range(2)];ih=[hashlib.sha256(),hashlib.sha256()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);best=-1;history=[]
    for step in range(1,2001):
        indices=[]
        for j in range(2):
            fg,bg=samplers[j].draw(generators[j]);ix=torch.cat([fg,bg]);indices.append(ix);ih[j].update(ix.cpu().numpy().tobytes())
            hist[j]+=torch.bincount(samplers[j].labels[ix],minlength=26);th[j]+=torch.bincount(samplers[j].times[ix],minlength=8)
        optimizer.zero_grad(set_to_none=True)
        losses=[F.cross_entropy(model(fs[j][indices[j]]),samplers[j].labels[indices[j]])for j in range(2)]
        loss=(losses[0]+WEIGHTS[arm]*losses[1])/(1+WEIGHTS[arm]);loss.backward();optimizer.step()
        if step%50==0:
            value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach()),'source_loss':float(losses[0].detach()),'prop_loss':float(losses[1].detach())})
            if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    baseline=json.loads(checkpoint('source_gt',seed).with_suffix('.json').read_text())
    assert hist[0].cpu().tolist()==baseline['sampled_class_counts']and int(th[0][0])==512000 and int(th[1][0])==0
    assert all(int(h[0])==256000 and int(h[1:].sum())==256000 for h in hist)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.pt')
    torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':protocol},tmp);tmp.replace(path)
    r={'arm':arm,'seed':seed,'protocol':protocol,'history':history,'best_step':best_step,'best_val':best,'parameter_count':10010,
       'source_class_counts':hist[0].cpu().tolist(),'prop_class_counts':hist[1].cpu().tolist(),'source_time_counts':th[0].cpu().tolist(),'prop_time_counts':th[1].cpu().tolist(),
       'source_indices_sha256':ih[0].hexdigest(),'prop_indices_sha256':ih[1].hexdigest(),'checkpoint_sha256':file_hash(path)}
    atomic_json(rp,r);del model,fs;gc.collect();torch.cuda.empty_cache();print('trained GT exposure',arm,seed,best_step,best,flush=True);return r


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed GT exposure control exists')
    manifest,p,old=freeze();source,later=pools(manifest);val=prior.read('native_sam','val',manifest)
    reports={a:[train(a,s,source,later,val,dict(p,arm=a))for s in range(3)]for a in WEIGHTS};del source,later
    offsets={'source_gt':p['source_gt_fixed_offset']};curves={};results={}
    for arm in ARMS:
        packages=[infer('dinov2',s,'val',val,checkpoint_path=checkpoint(arm,s),output_root=ROOT/arm)for s in range(3)]
        if arm!='source_gt':offsets[arm],curves[arm]=pick_offset(packages,p['calibration_grid'])
        runs=[{'seed':s,'raw_validation':evaluate(packages[s],0),'calibrated_validation':evaluate(packages[s],offsets[arm])}for s in range(3)]
        if arm=='source_gt':
            for s in range(3):assert runs[s]['calibrated_validation']==old['results']['native_gt']['runs'][s]['calibrated_validation']
        results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))}
        print('validation only GT exposure',arm,offsets[arm],results[arm]['mean_calibrated_validation_macro'],flush=True)
    selected=max(ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro'])
    selection={'provisional_validation_choice':selected,'offsets':offsets,'curves':curves,'deployment_unchanged':'native_gt',
        'checkpoint_sha256':{f'{a}_{s}':file_hash(checkpoint(a,s))for a in ARMS for s in range(3)}}
    atomic_json(ROOT/'frozen_selection.json',selection)
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'manifest':manifest,'training_reports':reports,'results':results,
        'provisional_validation_choice':selected,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),
        'verification':{'source_class_exposure_matches_frozen_GT':True,'only_six_new_heads':True,'no_new_confirmation_or_deployment_change':True,'old_inputs_unchanged':True}})
    print('DONE GT exposure supplement exploration',selected,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
