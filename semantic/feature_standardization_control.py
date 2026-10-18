"""Train-only affine parameterization control without new feature information."""
import fcntl,gc,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import segformer_feature_control as prior
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz
from semantic.run import atomic_json,classification_metrics
from semantic.oracle_probe import file_hash
ROOT=Path('semantic_artifacts/feature_standardization_control')
RAW=('b14_gt32','mit_encoder_fp32','ade_encoder_fp32','ade_decoder_fp32')
NEW=tuple('z_'+arm for arm in RAW)
ARMS=RAW+NEW
SPLITS=prior.SPLITS

def family(arm):
    return arm[2:] if arm.startswith('z_') else arm

def ckpt(arm,seed):
    return ROOT/arm/'runs'/f'linear_seed{seed}.pt' if arm in NEW else prior.ckpt(arm,seed)

def raw_path(arm,split,vid):
    if arm=='b14_gt32':
        if split=='spatial_val':
            return prior.spatial.fp('B14',vid)
        return prior.capacity.fp('train' if split=='train' else 'val',vid)
    return prior.fp(arm,split,vid)

def raw_data(arm,split,m):
    result=[]
    for clip in sorted(m['train'] if split=='train' else m['val'],key=lambda c:str(c['video_id'])):
        with np.load(raw_path(arm,split,clip['video_id']))as a:
            package={key:a[key].copy()for key in a.files}
        if arm=='b14_gt32' and split=='spatial_val':
            with np.load(prior.spatial.qp(clip['video_id']))as a:
                package['labels']=a['labels'].copy()
        result.append(package)
    return result

def fit_statistics(x):
    if x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError('Finite training feature matrix required')
    values=x.astype(np.float64)
    mean=values.mean(0).astype(np.float32)
    std=values.std(0,ddof=0)
    scale=np.where(std<=1e-6,1.,std).astype(np.float32)
    return mean,scale,std.astype(np.float32)

def transform(x,mean,scale):
    if x.shape[-1]!=len(mean) or mean.shape!=scale.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Matching finite affine statistics required')
    value=(x.astype(np.float32)-mean)/scale
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite standardized features')
    return value

def freeze():
    source=json.loads((prior.ROOT/'summary.json').read_text())
    assert file_hash(prior.ROOT/'summary.json')==json.loads((prior.ROOT/'verification.json').read_text())['summary_sha256']
    m=json.loads((prior.ROOT/'manifest.json').read_text())
    paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'frozen_selection.json',prior.ROOT/'manifest.json',Path('semantic/feature_standardization_control.py'),Path('semantic/OVIS_FEATURE_STANDARDIZATION_PROTOCOL.md'),Path('tests/test_feature_standardization.py'),Path('semantic/model.py'),Path('semantic/head_budget_probe.py'),Path('semantic/sampling_ablation.py'),Path('semantic/background_calibration.py')]
    for arm in RAW:
        for split in ('train',)+SPLITS:
            for clip in m['train'] if split=='train' else m['val']:
                paths.append(raw_path(arm,split,clip['video_id']))
                if arm=='b14_gt32' and split=='spatial_val':
                    paths.append(prior.spatial.qp(clip['video_id']))
        for seed in range(3):
            paths.extend([ckpt(arm,seed),prior.ROOT/arm/'logits'/f'seed{seed}_balanced_val.npz',prior.ROOT/arm/'logits'/f'seed{seed}_spatial_val.npz'])
    train_labels=np.concatenate([a['labels']for a in raw_data('b14_gt32','train',m)])
    val_labels=np.concatenate([a['labels']for a in raw_data('b14_gt32','balanced_val',m)])
    common=sorted(set(map(int,train_labels[train_labels>0]))&set(map(int,val_labels[val_labels>0])))
    test=subprocess.run([str(prior.resource_features.ROOT/'env/bin/python'),'tests/test_feature_standardization.py'],capture_output=True,text=True,check=True)
    (ROOT/'unit_checks.log').write_text(test.stdout+test.stderr)
    assert 'PASS'in test.stdout
    p={'round':83,'raw_arms':list(RAW),'new_arms':list(NEW),'new_heads':12,'steps':2000,'batch':32,'exposures':64000,'statistics':'training only FP64 population moments, FP32 parameters','small_std_threshold':1e-6,'small_std_scale':1.,'second_FP16_quantization':False,'common_foreground_classes':common,'calibration_grid':source['protocol']['calibration_grid'],'fixed_raw_offsets':{arm:source['results'][arm]['offsets']for arm in RAW},'source_validation_only':True,'unit_checks_sha256':file_hash(ROOT/'unit_checks.log'),'input_sha256':{str(path):file_hash(path)for path in paths}}
    if(ROOT/'protocol.json').exists():
        assert json.loads((ROOT/'protocol.json').read_text())==p
    else:
        atomic_json(ROOT/'protocol.json',p)
    atomic_json(ROOT/'manifest.json',m)
    return m,p

def statistics(m):
    report={}
    for arm in RAW:
        base=raw_data(arm,'train',m)
        x=np.concatenate([a['decoded']for a in base])
        assert len(x)==20480
        mean,scale,std=fit_statistics(x)
        path=ROOT/'statistics'/f'{arm}.npz'
        if path.exists():
            with np.load(path)as a:
                assert np.array_equal(mean,a['mean']) and np.array_equal(scale,a['scale']) and np.array_equal(std,a['std'])
        else:
            save_npz(path,mean=mean,scale=scale,std=std)
        report[arm]={'path':str(path),'sha256':file_hash(path),'samples':len(x),'dimension':x.shape[1],'fallback_scale_channels':int((std<=1e-6).sum()),'std_min':float(std.min()),'std_median':float(np.median(std)),'std_max':float(std.max())}
    path=ROOT/'statistics_manifest.json'
    if path.exists():
        assert json.loads(path.read_text())==report
    else:
        atomic_json(path,report)
    return report

def read(arm,split,m,stats):
    data=raw_data(family(arm),split,m)
    if arm in NEW:
        with np.load(stats[family(arm)]['path'])as a:
            mean,scale=a['mean'].copy(),a['scale'].copy()
        data=[{**x,'decoded':transform(x['decoded'],mean,scale)}for x in data]
    return data

def train(arm,seed,base,val,p,stats):
    path=ckpt(arm,seed)
    report=path.with_suffix('.json')
    if report.exists():
        r=json.loads(report.read_text())
        assert r['protocol']==p and r['statistics_sha256']==stats[family(arm)]['sha256'] and file_hash(path)==r['checkpoint_sha256']
        return r
    torch.manual_seed(seed)
    dimension=base[0]['decoded'].shape[1]
    model=SemanticHead(hidden_dim=dimension,num_classes=26).cuda()
    x=torch.from_numpy(np.concatenate([a['decoded']for a in base])).cuda()
    y=np.concatenate([a['labels']for a in base])
    times=np.concatenate([a['xy'][:,2]for a in base])
    assert (times==0).all() and len(y)==20480 and x.dtype==torch.float32
    sampler=Sampler(y,times,'baseline')
    gen=torch.Generator(device='cuda').manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    ih=hashlib.sha256()
    hist=torch.zeros(26,dtype=torch.long,device='cuda')
    best=-1
    history=[]
    for step in range(1,2001):
        fg,bg=sampler.draw(gen,count=16)
        ix=torch.cat([fg,bg])
        ih.update(ix.cpu().numpy().tobytes())
        hist+=torch.bincount(sampler.labels[ix],minlength=26)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[ix]),sampler.labels[ix])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if step%50==0:
            value=score(model,val)
            history.append({'step':step,'balanced_raw_val_macro':value,'loss':float(loss.detach())})
            if value>best:
                best=value
                best_step=step
                state={key:value.detach().cpu().clone()for key,value in model.state_dict().items()}
    reference=json.loads(ckpt(family(arm),seed).with_suffix('.json').read_text())
    assert ih.hexdigest()==reference['sampled_indices_sha256'] and hist.cpu().tolist()==reference['sampled_class_counts']
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.partial.pt')
    torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p,'statistics_sha256':stats[family(arm)]['sha256']},partial)
    partial.replace(path)
    r={'arm':arm,'seed':seed,'protocol':p,'statistics_sha256':stats[family(arm)]['sha256'],'best_step':best_step,'best_balanced_raw_val':best,'history':history,'sampled_indices_sha256':ih.hexdigest(),'sampled_class_counts':hist.cpu().tolist(),'head_parameters':26*dimension+26,'checkpoint_sha256':file_hash(path)}
    atomic_json(report,r)
    print('trained fixed standardization',arm,seed,best_step,best,flush=True)
    del model,x
    gc.collect()
    torch.cuda.empty_cache()
    return r

def infer(arm,seed,split,data):
    path=ROOT/arm/'logits'/f'seed{seed}_{split}.npz'
    checkpoint_hash=file_hash(ckpt(arm,seed))
    if path.exists():
        with np.load(path)as a:
            out={key:a[key].copy()for key in a.files}
        assert str(out['checkpoint_sha256'])==checkpoint_hash
        return out
    if arm in RAW:
        with np.load(prior.ROOT/arm/'logits'/f'seed{seed}_{split}.npz')as a:
            out={key:a[key].copy()for key in a.files}
    else:
        model=SemanticHead(hidden_dim=data[0]['decoded'].shape[1],num_classes=26).cuda().eval()
        model.load_state_dict(torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model'])
        with torch.inference_mode():
            values=[model(torch.from_numpy(a['decoded']).cuda()).cpu().numpy()for a in data]
        out={'logits':np.concatenate(values),'labels':np.concatenate([a['labels']for a in data]),'times':np.concatenate([a['xy'][:,2]for a in data]),'video_ids':np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in data]),'checkpoint_sha256':np.array(checkpoint_hash)}
        del model
    assert str(out['checkpoint_sha256'])==checkpoint_hash
    save_npz(path,**out)
    return out

def training_fit(arm,seed,base,balanced_out,p):
    path=ROOT/'training_logits'/f'{arm}_{seed}.npz'
    state=torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model']
    weight,bias=state['classifier.weight'].cuda(),state['classifier.bias'].cuda()
    with torch.inference_mode():
        values=np.concatenate([F.linear(torch.from_numpy(a['decoded']).float().cuda(),weight,bias).cpu().numpy()for a in base])
    labels=np.concatenate([a['labels']for a in base])
    pred=values.argmax(-1)
    valpred=balanced_out['logits'].argmax(-1)
    valtarget=balanced_out['labels']
    classes=p['common_foreground_classes']
    recall={str(c):float((pred[labels==c]==c).mean())for c in classes}
    valrecall={str(c):float((valpred[valtarget==c]==c).mean())for c in classes}
    save_npz(path,logits=values,labels=labels)
    return {'raw_source_training':classification_metrics(pred,labels,26),'common_FG_train_recall':recall,'common_FG_balanced_val_recall':valrecall,'common_FG_macro_train_recall':float(np.mean(list(recall.values()))),'common_FG_macro_balanced_val_recall':float(np.mean(list(valrecall.values()))),'output_path':str(path),'output_sha256':file_hash(path)}

def folded_equivalence(arm,seed,raw_val,standardized_out,stats):
    state=torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model']
    with np.load(stats[family(arm)]['path'])as a:
        mean=torch.from_numpy(a['mean'].copy()).cuda()
        scale=torch.from_numpy(a['scale'].copy()).cuda()
    weight=state['classifier.weight'].cuda()/scale
    bias=state['classifier.bias'].cuda()-F.linear(mean[None],weight)[0]
    with torch.inference_mode():
        values=np.concatenate([F.linear(torch.from_numpy(a['decoded']).float().cuda(),weight,bias).cpu().numpy()for a in raw_val])
    expected=standardized_out['logits']
    maximum=float(np.abs(values-expected).max())
    tolerance=2e-4+2e-5*float(np.abs(expected).max())
    assert maximum<=tolerance
    return {'maximum_raw_fold_logit_difference':maximum,'preregistered_tolerance':tolerance,'label_changes_due_to_numerics':int((values.argmax(-1)!=expected.argmax(-1)).sum())}

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with(ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(ROOT/'summary.json').exists():
            raise RuntimeError('Completed train-only standardization control exists')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        m,p=freeze()
        stats=statistics(m)
        reports={}
        results={}
        curves={}
        offsets={}
        fits={}
        folds={}
        for arm in ARMS:
            base=read(arm,'train',m,stats)
            datasets={split:read(arm,split,m,stats)for split in SPLITS}
            if arm in NEW:
                reports[arm]=[train(arm,seed,base,datasets['balanced_val'],p,stats)for seed in range(3)]
            outputs={split:[infer(arm,seed,split,datasets[split])for seed in range(3)]for split in SPLITS}
            offsets[arm]={'raw':0.}
            curves[arm]={}
            for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
                if arm in RAW:
                    offsets[arm][profile]=p['fixed_raw_offsets'][arm][profile]
                else:
                    offsets[arm][profile],curves[arm][profile]=pick_offset(outputs[split],p['calibration_grid'])
            runs=[{'seed':seed,**{split:{profile:evaluate(outputs[split][seed],offset)for profile,offset in offsets[arm].items()}for split in SPLITS}}for seed in range(3)]
            results[arm]={'offsets':offsets[arm],'runs':runs,'mean_balanced_profile_macro':float(np.mean([r['balanced_val']['balanced_profile']['macro_f1_present']for r in runs]))}
            fits[arm]=[training_fit(arm,seed,base,outputs['balanced_val'][seed],p)for seed in range(3)]
            if arm in NEW:
                raw_val=raw_data(family(arm),'balanced_val',m)
                folds[arm]=[folded_equivalence(arm,seed,raw_val,outputs['balanced_val'][seed],stats)for seed in range(3)]
            print('standardization balanced score',arm,results[arm]['mean_balanced_profile_macro'],flush=True)
            del base,datasets,outputs
        comparisons={arm+'_minus_'+family(arm):prior.contrasts(results,arm,family(arm))for arm in NEW}
        selected=max(ARMS,key=lambda arm:results[arm]['mean_balanced_profile_macro'])
        selection={'validation_selected_arm':selected,'offsets':offsets,'curves':curves,'statistics_manifest_sha256':file_hash(ROOT/'statistics_manifest.json'),'checkpoint_sha256':{f'{arm}_{seed}':file_hash(ckpt(arm,seed))for arm in ARMS for seed in range(3)},'deployment_unchanged':'native_gt'}
        atomic_json(ROOT/'frozen_selection.json',selection)
        for path,h in p['input_sha256'].items():
            assert file_hash(path)==h
        atomic_json(ROOT/'summary.json',{'protocol':p,'statistics':stats,'statistics_manifest_sha256':file_hash(ROOT/'statistics_manifest.json'),'training_reports':reports,'results':results,'training_fits_and_common_class_transfer':fits,'affine_fold_equivalence':folds,'within_branch_standardization_contrasts':comparisons,'validation_selected_arm':selected,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'new_confirmation_groups':0,'deployment_unchanged':'native_gt'})
        print('DONE train-only affine standardization control',selected,flush=True)
if __name__=='__main__':
    try:
        main()
    except Exception as e:
        ROOT.mkdir(parents=True,exist_ok=True)
        atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)})
        raise
