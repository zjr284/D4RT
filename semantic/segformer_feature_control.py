"""Fixed query/head budget comparison of official segmentation feature branches."""
import fcntl,gc,hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch
from semantic import segformer_features as resource_features
from semantic import dino_backbone_training as capacity
from semantic import dino_spatial_calibration as spatial
from semantic.segformer_precision_features import feature_maps,at_queries,load_official
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/segformer_feature_control')
NEW=tuple(branch+'_'+precision for precision in ('fp32','bf16')for branch in ('mit_encoder','ade_encoder','ade_decoder'))
ARMS=('b14_gt32',)+NEW
SPLITS=('balanced_val','spatial_val')

def ckpt(arm,seed):
    if arm=='b14_gt32':
        return capacity.ckpt('B14_gt32',seed)
    return ROOT/arm/'runs'/f'linear_seed{seed}.pt'

def fp(arm,split,vid):
    return ROOT/'features'/arm/split/f'{vid}.npz'

def meta(split,vid):
    path=spatial.qp(vid) if split=='spatial_val' else capacity.fp('train' if split=='train' else 'val',vid)
    with np.load(path)as data:
        return {k:data[k].copy()for k in data.files if k!='decoded'}

def reference_file(split,seed):
    if split=='balanced_val':
        return capacity.ROOT/'B14_gt32'/'logits'/f'dinov2_{seed}_val.npz'
    return spatial.ROOT/'logits'/f'B14_gt32_{seed}.npz'

def freeze():
    source=json.loads((resource_features.ROOT/'summary.json').read_text())
    assert file_hash(resource_features.ROOT/'summary.json')==json.loads((resource_features.ROOT/'verification.json').read_text())['summary_sha256']
    assert file_hash(spatial.ROOT/'summary.json')==json.loads((spatial.ROOT/'verification.json').read_text())['summary_sha256']
    m=json.loads((capacity.ROOT/'manifest.json').read_text())
    paths=[resource_features.ROOT/'summary.json',resource_features.ROOT/'verification.json',resource_features.ROOT/'model_specs.json',resource_features.ROOT/'preflight.json',resource_features.ROOT/'env_install.json',capacity.ROOT/'manifest.json',capacity.ROOT/'summary.json',capacity.ROOT/'verification.json',spatial.ROOT/'summary.json',spatial.ROOT/'verification.json',spatial.ROOT/'frozen_selection.json',Path('semantic/segformer_feature_control.py'),Path('semantic/segformer_precision_features.py'),Path('semantic/OVIS_SEGFORMER_FEATURE_CONTROL_PROTOCOL.md'),Path('semantic/segformer_features.py'),Path('semantic/model.py'),Path('semantic/sampling_ablation.py'),Path('semantic/head_budget_probe.py'),Path('semantic/background_calibration.py')]
    import transformers
    from transformers.models.segformer import modeling_segformer,configuration_segformer
    assert transformers.__version__=='4.57.6'
    paths.extend([Path(modeling_segformer.__file__),Path(configuration_segformer.__file__)])
    for spec in source['model_specs'].values():
        assert file_hash(spec['weight_path'])==spec['expected_sha256']
        paths.extend(Path(spec[k])for k in ('weight_path','config_path','preprocessor_path'))
    for split in ('train','val'):
        for clip in m[split]:
            paths.append(capacity.fp(split,clip['video_id']))
            paths.extend(Path(x)for x in (clip['frames'][:1] if split=='train' else clip['frames']))
            if split=='val':
                paths.append(spatial.qp(clip['video_id']))
                paths.append(spatial.fp('B14',clip['video_id']))
    for seed in range(3):
        paths.extend([ckpt('b14_gt32',seed),ckpt('b14_gt32',seed).with_suffix('.json')])
        paths.extend(reference_file(split,seed)for split in SPLITS)
    legacy=json.loads((capacity.ROOT/'summary.json').read_text())
    natural=json.loads((spatial.ROOT/'frozen_selection.json').read_text())
    p={'round':82,'arms':list(ARMS),'new_heads':18,'feature_dim_new':256,'head_parameters_new':6682,'head_parameters_reference':19994,'steps':2000,'batch_size':32,'exposures':64000,'primary_raw_balanced_contrast':['ade_encoder_fp32','mit_encoder_fp32'],'calibration_grid':legacy['protocol']['calibration_grid'],'fixed_reference_offsets':{'raw':0.,'balanced_profile':legacy['results']['B14_gt32']['offset'],'spatial_profile':natural['spatial_offsets']['B14_gt32']},'matmul_and_cudnn_TF32_disabled':True,'OVIS_source_supervision_same':True,'external_ADE_dense_supervision_extra':True,'source_validation_only':True,'input_sha256':{str(path):file_hash(path)for path in paths}}
    if(ROOT/'protocol.json').exists():
        assert json.loads((ROOT/'protocol.json').read_text())==p
    else:
        atomic_json(ROOT/'protocol.json',p)
    atomic_json(ROOT/'manifest.json',m)
    return m,p

def cache(m):
    costs=json.loads((ROOT/'extraction.json').read_text()) if(ROOT/'extraction.json').exists() else []
    for alias in ('mit','ade'):
        branches=(alias+'_encoder',) if alias=='mit' else ('ade_encoder','ade_decoder')
        model=None
        for precision in ('fp32','bf16'):
            for split in ('train','val'):
                for clip in m[split]:
                    output_splits=('train',) if split=='train' else SPLITS
                    needed=[fp(branch+'_'+precision,outsplit,clip['video_id'])for branch in branches for outsplit in output_splits]
                    if all(path.exists()for path in needed):
                        continue
                    if model is None:
                        model=load_official(alias)
                    metadata={key:meta(key,clip['video_id'])for key in output_splits}
                    points=np.concatenate([metadata[key]['xy']for key in output_splits])
                    video=capacity.rgb.load_native_video({'frames':clip['frames'][:1] if split=='train' else clip['frames']}).cuda()
                    torch.cuda.synchronize()
                    started=time.perf_counter()
                    maps=feature_maps(model,alias,video,precision)
                    for branch in branches:
                        values=at_queries(maps[branch],points).half().cpu().numpy()
                        start=0
                        for output_split in output_splits:
                            n=len(metadata[output_split]['labels'])
                            package={**metadata[output_split],'decoded':values[start:start+n]}
                            start+=n
                            path=fp(branch+'_'+precision,output_split,clip['video_id'])
                            if path.exists():
                                with np.load(path)as a:
                                    assert all(np.array_equal(a[k],v)for k,v in package.items())
                            else:
                                save_npz(path,**package)
                    torch.cuda.synchronize()
                    costs.append({'alias':alias,'precision':precision,'split':split,'video_id':clip['video_id'],'seconds':time.perf_counter()-started,'map_shapes':{k:list(v.shape)for k,v in maps.items()}})
                    atomic_json(ROOT/'extraction.json',costs)
                    print('SegFormer feature cache',alias,precision,split,clip['video_id'],flush=True)
                    del video,maps
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    for branch in ('mit_encoder','ade_encoder','ade_decoder'):
        with np.load(fp(branch+'_bf16','balanced_val',258))as a:
            with np.load(resource_features.ROOT/'preflight_features'/f'{branch}.npz')as old:
                assert np.array_equal(a['xy'][::32],old['xy']) and np.array_equal(a['decoded'][::32],old['decoded'])

def read(arm,split,m):
    clips=m['train'] if split=='train' else m['val']
    data=[]
    for clip in sorted(clips,key=lambda c:str(c['video_id'])):
        with np.load(fp(arm,split,clip['video_id']))as a:
            data.append({k:a[k].copy()for k in a.files})
    return data

def train(arm,seed,base,val,p):
    path=ckpt(arm,seed)
    report=path.with_suffix('.json')
    if report.exists():
        r=json.loads(report.read_text())
        assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256']
        return r
    torch.manual_seed(seed)
    model=SemanticHead(hidden_dim=256,num_classes=26).cuda()
    assert sum(a.numel()for a in model.parameters())==6682
    x=torch.from_numpy(np.concatenate([a['decoded']for a in base])).float().cuda()
    y=np.concatenate([a['labels']for a in base])
    times=np.concatenate([a['xy'][:,2]for a in base])
    assert len(y)==20480 and (times==0).all()
    sampler=Sampler(y,times,'baseline')
    generator=torch.Generator(device='cuda').manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    hist=torch.zeros(26,dtype=torch.long,device='cuda')
    indices_hash=hashlib.sha256()
    history=[]
    best=-1
    for step in range(1,2001):
        foreground,background=sampler.draw(generator,count=16)
        ix=torch.cat([foreground,background])
        indices_hash.update(ix.cpu().numpy().tobytes())
        hist+=torch.bincount(sampler.labels[ix],minlength=26)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(x[ix]),sampler.labels[ix])
        loss.backward()
        optimizer.step()
        if step%50==0:
            value=score(model,val)
            history.append({'step':step,'balanced_raw_val_macro':value,'loss':float(loss.detach())})
            if value>best:
                best=value
                best_step=step
                state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    reference=json.loads(ckpt('b14_gt32',seed).with_suffix('.json').read_text())
    assert indices_hash.hexdigest()==reference['sampled_indices_sha256'] and hist.cpu().tolist()==reference['sampled_class_counts']
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.partial.pt')
    torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},partial)
    partial.replace(path)
    r={'arm':arm,'seed':seed,'protocol':p,'best_step':best_step,'best_balanced_raw_val':best,'history':history,'sampled_indices_sha256':indices_hash.hexdigest(),'sampled_class_counts':hist.cpu().tolist(),'head_parameters':6682,'checkpoint_sha256':file_hash(path)}
    atomic_json(report,r)
    print('trained SegFormer head',arm,seed,best_step,best,flush=True)
    del model,x
    gc.collect()
    torch.cuda.empty_cache()
    return r

def infer(arm,seed,split,data,m):
    path=ROOT/arm/'logits'/f'seed{seed}_{split}.npz'
    h=file_hash(ckpt(arm,seed))
    if path.exists():
        with np.load(path)as a:
            out={k:a[k].copy()for k in a.files}
        assert str(out['checkpoint_sha256'])==h
        return out
    if arm=='b14_gt32':
        with np.load(reference_file(split,seed))as a:
            source={k:a[k].copy()for k in a.files}
        packages=[]
        for clip in sorted(m['val'],key=lambda c:str(c['video_id'])):
            rows=source['video_ids']==clip['video_id']
            packages.append({key:source[key][rows]for key in ('logits','labels','times','video_ids')})
        out={key:np.concatenate([a[key]for a in packages])for key in packages[0]}
    else:
        model=SemanticHead(hidden_dim=256,num_classes=26).cuda().eval()
        model.load_state_dict(torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model'])
        with torch.inference_mode():
            values=[model(torch.from_numpy(a['decoded']).float().cuda()).cpu().numpy()for a in data]
        out={'logits':np.concatenate(values),'labels':np.concatenate([a['labels']for a in data]),'times':np.concatenate([a['xy'][:,2]for a in data]),'video_ids':np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in data])}
        del model
    out['checkpoint_sha256']=np.array(h)
    assert (out['labels']>=0).all() and len(out['labels'])==40960
    save_npz(path,**out)
    return out

def contrasts(results,left,right):
    output={}
    for split in SPLITS:
        output[split]={}
        for profile in ('raw','balanced_profile','spatial_profile'):
            differences=[results[left]['runs'][seed][split][profile]['macro_f1_present']-results[right]['runs'][seed][split][profile]['macro_f1_present']for seed in range(3)]
            output[split][profile]={'per_seed_macro_difference':differences,'mean':float(np.mean(differences))}
    return output

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with(ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(ROOT/'summary.json').exists():
            raise RuntimeError('Completed segmentation feature control exists')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        m,p=freeze()
        cache(m)
        reports={}
        results={}
        curves={}
        offsets={}
        for arm in ARMS:
            datasets={key:read(arm,key,m)for key in SPLITS} if arm in NEW else {key:None for key in SPLITS}
            if arm in NEW:
                base=read(arm,'train',m)
                reports[arm]=[train(arm,seed,base,datasets['balanced_val'],p)for seed in range(3)]
                del base
            outputs={key:[infer(arm,seed,key,datasets[key],m)for seed in range(3)]for key in SPLITS}
            offsets[arm]={'raw':0.}
            curves[arm]={}
            for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
                if arm=='b14_gt32':
                    offsets[arm][profile]=p['fixed_reference_offsets'][profile]
                else:
                    offsets[arm][profile],curves[arm][profile]=pick_offset(outputs[split],p['calibration_grid'])
            runs=[]
            for seed in range(3):
                runs.append({'seed':seed,**{key:{profile:evaluate(outputs[key][seed],offset)for profile,offset in offsets[arm].items()}for key in SPLITS}})
            results[arm]={'offsets':offsets[arm],'runs':runs,'mean_balanced_profile_macro':float(np.mean([r['balanced_val']['balanced_profile']['macro_f1_present']for r in runs]))}
            print('feature-control balanced selection score',arm,results[arm]['mean_balanced_profile_macro'],flush=True)
            del datasets,outputs
        main_contrast=contrasts(results,'ade_encoder_fp32','mit_encoder_fp32')
        pairs=[('ade_decoder_fp32','ade_encoder_fp32')]+[(branch+'_bf16',branch+'_fp32')for branch in ('mit_encoder','ade_encoder','ade_decoder')]+[(arm,'b14_gt32')for arm in NEW]
        secondary={left+'_minus_'+right:contrasts(results,left,right)for left,right in pairs}
        selected=max(ARMS,key=lambda arm:results[arm]['mean_balanced_profile_macro'])
        selection={'validation_selected_arm':selected,'criterion':'balanced_profile mean macro on original balanced val only','offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{arm}_{seed}':file_hash(ckpt(arm,seed))for arm in ARMS for seed in range(3)},'new_candidate_exceeds_B14':selected!='b14_gt32','deployment_unchanged':'native_gt'}
        atomic_json(ROOT/'frozen_selection.json',selection)
        for path,h in p['input_sha256'].items():
            assert file_hash(path)==h
        atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'primary_pretraining_contrast':main_contrast,'secondary_contrasts':secondary,'validation_selected_arm':selected,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'features').rglob('*.npz'))},'extraction_sha256':file_hash(ROOT/'extraction.json'),'new_confirmation_groups':0,'deployment_unchanged':'native_gt'})
        print('DONE segmentation pretraining and precision feature control',selected,flush=True)
if __name__=='__main__':
    try:
        main()
    except Exception as e:
        ROOT.mkdir(parents=True,exist_ok=True)
        atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)})
        raise
