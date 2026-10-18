"""Matched-width complementary frozen features; all choices from source validation."""
import fcntl,gc,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import feature_standardization_control as prior
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz
from semantic.run import atomic_json
from semantic.oracle_probe import file_hash
ROOT=Path('semantic_artifacts/feature_fusion_control')
REFERENCE='z_b14_gt32'
NEW=('b14_zero','b14_mit','b14_ade_encoder','b14_ade_decoder')
ARMS=(REFERENCE,)+NEW
EXTRA={'b14_zero':None,'b14_mit':'z_mit_encoder_fp32','b14_ade_encoder':'z_ade_encoder_fp32','b14_ade_decoder':'z_ade_decoder_fp32'}
SPLITS=prior.SPLITS

def ckpt(arm,seed):
    return prior.ckpt(arm,seed) if arm==REFERENCE else ROOT/arm/'runs'/f'linear_seed{seed}.pt'

def join_features(base,extra):
    if base['decoded'].ndim!=2 or base['decoded'].shape[1]!=768 or base['decoded'].dtype!=np.float32:
        raise ValueError('Expected FP32 standardized B14 features')
    if extra is None:
        right=np.zeros((len(base['decoded']),256),np.float32)
    else:
        for key in ('labels','xy','video_id'):
            if not np.array_equal(base[key],extra[key]):raise ValueError('Unaligned query provenance: '+key)
        right=extra['decoded']
        if right.shape!=(len(base['decoded']),256) or right.dtype!=np.float32:
            raise ValueError('Expected aligned FP32 standardized256 features')
    value=np.concatenate([base['decoded'],right],axis=1)
    if not np.isfinite(value).all():raise ValueError('Nonfinite fusion input')
    return {**base,'decoded':value}

def read(arm,split,m,stats):
    base=prior.read(REFERENCE,split,m,stats)
    if arm==REFERENCE:return base
    extra=EXTRA[arm]
    if extra is None:return [join_features(a,None)for a in base]
    right=prior.read(extra,split,m,stats)
    assert len(base)==len(right)
    return [join_features(a,b)for a,b in zip(base,right)]

def freeze():
    source=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text())
    assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256']
    paths=[Path(x)for x in source['protocol']['input_sha256']]
    paths.extend([prior.ROOT/name for name in ('summary.json','verification.json','frozen_selection.json','statistics_manifest.json','manifest.json')])
    paths.extend(Path(row['path'])for row in source['statistics'].values())
    paths.extend([Path('semantic/feature_fusion_control.py'),Path('semantic/OVIS_FEATURE_FUSION_PROTOCOL.md'),Path('tests/test_feature_fusion.py'),Path('semantic/feature_standardization_control.py')])
    for seed in range(3):
        paths.extend([prior.ckpt(REFERENCE,seed),prior.ckpt(REFERENCE,seed).with_suffix('.json')])
        paths.extend(prior.ROOT/REFERENCE/'logits'/f'seed{seed}_{split}.npz'for split in SPLITS)
    test=subprocess.run([str(prior.prior.resource_features.ROOT/'env/bin/python'),'tests/test_feature_fusion.py'],capture_output=True,text=True,check=True)
    (ROOT/'unit_checks.log').write_text(test.stdout+test.stderr);assert 'PASS'in test.stdout
    p={'round':84,'new_heads':12,'new_arms':list(NEW),'reference':REFERENCE,'dimension':1024,'head_parameters':26650,'steps':2000,'batch':32,'exposures':64000,'learning_rate':.001,'weight_decay':.01,'calibration_grid':source['protocol']['calibration_grid'],'fixed_reference_offsets':source['results'][REFERENCE]['offsets'],'statistics_manifest_sha256':source['statistics_manifest_sha256'],'source_validation_only':True,'new_confirmation_groups':0,'unit_checks_sha256':file_hash(ROOT/'unit_checks.log'),'input_sha256':{str(path):file_hash(path)for path in sorted(set(paths))}}
    if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
    else:atomic_json(ROOT/'protocol.json',p)
    m=json.loads((prior.ROOT/'manifest.json').read_text());atomic_json(ROOT/'manifest.json',m)
    return m,p,source['statistics']

def train(arm,seed,base,val,p):
    path=ckpt(arm,seed);report=path.with_suffix('.json')
    if report.exists():
        r=json.loads(report.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
    torch.manual_seed(seed);model=SemanticHead(hidden_dim=1024,num_classes=26).cuda()
    x=torch.from_numpy(np.concatenate([a['decoded']for a in base])).cuda();labels=np.concatenate([a['labels']for a in base]);times=np.concatenate([a['xy'][:,2]for a in base])
    assert len(labels)==20480 and (times==0).all() and x.dtype==torch.float32
    sampler=Sampler(labels,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);ih=hashlib.sha256();hist=torch.zeros(26,dtype=torch.long,device='cuda');best=-1;history=[]
    for step in range(1,2001):
        fg,bg=sampler.draw(gen,count=16);ix=torch.cat([fg,bg]);ih.update(ix.cpu().numpy().tobytes());hist+=torch.bincount(sampler.labels[ix],minlength=26)
        model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ix]),sampler.labels[ix]);assert torch.isfinite(loss);loss.backward();optimizer.step()
        if step%50==0:
            value=score(model,val);history.append({'step':step,'balanced_raw_val_macro':value,'loss':float(loss.detach())})
            if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    reference=json.loads(ckpt(REFERENCE,seed).with_suffix('.json').read_text())
    assert reference['sampled_indices_sha256']==ih.hexdigest() and reference['sampled_class_counts']==hist.cpu().tolist()
    path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},partial);partial.replace(path)
    r={'arm':arm,'seed':seed,'protocol':p,'best_step':best_step,'best_balanced_raw_val':best,'history':history,'head_parameters':26650,'sampled_indices_sha256':ih.hexdigest(),'sampled_class_counts':hist.cpu().tolist(),'checkpoint_sha256':file_hash(path)}
    atomic_json(report,r);print('trained matched fusion',arm,seed,best_step,best,flush=True);del model,x;gc.collect();torch.cuda.empty_cache();return r

def infer(arm,seed,split,data):
    path=ROOT/arm/'logits'/f'seed{seed}_{split}.npz';h=file_hash(ckpt(arm,seed))
    if path.exists():
        with np.load(path)as a:out={k:a[k].copy()for k in a.files}
        assert str(out['checkpoint_sha256'])==h;return out
    if arm==REFERENCE:
        with np.load(prior.ROOT/REFERENCE/'logits'/f'seed{seed}_{split}.npz')as a:out={k:a[k].copy()for k in a.files}
    else:
        state=torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model']
        with torch.inference_mode():values=[F.linear(torch.from_numpy(a['decoded']).cuda(),state['classifier.weight'].cuda(),state['classifier.bias'].cuda()).cpu().numpy()for a in data]
        out={'logits':np.concatenate(values),'labels':np.concatenate([a['labels']for a in data]),'times':np.concatenate([a['xy'][:,2]for a in data]),'video_ids':np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in data]),'checkpoint_sha256':np.array(h)}
    save_npz(path,**out);return out

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with(ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(ROOT/'summary.json').exists():raise RuntimeError('Completed feature fusion control exists')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        m,p,stats=freeze();reports={};results={};curves={};offsets={}
        for arm in ARMS:
            datasets={split:read(arm,split,m,stats)for split in ('train',)+SPLITS}
            if arm in NEW:reports[arm]=[train(arm,seed,datasets['train'],datasets['balanced_val'],p)for seed in range(3)]
            outputs={split:[infer(arm,seed,split,datasets[split])for seed in range(3)]for split in SPLITS}
            offsets[arm]={'raw':0.};curves[arm]={}
            for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
                if arm==REFERENCE:offsets[arm][profile]=p['fixed_reference_offsets'][profile]
                else:offsets[arm][profile],curves[arm][profile]=pick_offset(outputs[split],p['calibration_grid'])
            runs=[{'seed':seed,**{split:{profile:evaluate(outputs[split][seed],offset)for profile,offset in offsets[arm].items()}for split in SPLITS}}for seed in range(3)]
            results[arm]={'offsets':offsets[arm],'runs':runs,'mean_balanced_profile_macro':float(np.mean([a['balanced_val']['balanced_profile']['macro_f1_present']for a in runs]))}
            print('matched fusion source score',arm,results[arm]['mean_balanced_profile_macro'],flush=True);del datasets,outputs
        pairs=[('b14_ade_encoder','b14_mit')]+[(arm,'b14_zero')for arm in NEW[1:]]+[(arm,REFERENCE)for arm in NEW]
        comparisons={a+'_minus_'+b:prior.prior.contrasts(results,a,b)for a,b in pairs}
        selected=max(ARMS,key=lambda a:results[a]['mean_balanced_profile_macro'])
        eligible=[]
        for arm in NEW[1:]:
            diff=comparisons[arm+'_minus_'+REFERENCE]['balanced_val']['balanced_profile']['per_seed_macro_difference']
            if results[arm]['mean_balanced_profile_macro']>max(results[REFERENCE]['mean_balanced_profile_macro'],results['b14_zero']['mean_balanced_profile_macro']) and min(diff)>0:eligible.append(arm)
        selection={'validation_selected_arm':selected,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{arm}_{seed}':file_hash(ckpt(arm,seed))for arm in ARMS for seed in range(3)},'eligible_for_separately_preregistered_confirmation':eligible,'deployment_unchanged':'native_gt'}
        atomic_json(ROOT/'frozen_selection.json',selection)
        for path,h in p['input_sha256'].items():assert file_hash(path)==h
        atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'contrasts':comparisons,'validation_selected_arm':selected,'eligible_for_separately_preregistered_confirmation':eligible,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'new_confirmation_groups':0,'deployment_unchanged':'native_gt'})
        print('DONE matched complementary feature control',selected,'eligible',eligible,flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
