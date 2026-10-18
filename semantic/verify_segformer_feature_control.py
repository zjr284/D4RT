"""Independent precision/feature reconstruction, budgets and all42 logits."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from semantic import segformer_feature_control as work
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def manual_maps(model,alias,video,precision):
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
        hidden=model.segformer(pixel_values=(video-mean)/std,output_hidden_states=True,return_dict=True).hidden_states
        last=hidden[-1]
        if last.ndim==3:
            last=last.transpose(1,2).reshape(len(video),256,14,14)
        result={alias+'_encoder':last.float()}
        if alias=='ade':
            projected=[]
            for state,layer in zip(hidden,model.decode_head.linear_c):
                h,w=state.shape[-2:]
                value=layer(state).permute(0,2,1).reshape(len(video),256,h,w)
                projected.append(F.interpolate(value,size=hidden[0].shape[-2:],mode='bilinear',align_corners=False))
            fused=model.decode_head.linear_fuse(torch.cat(list(reversed(projected)),dim=1))
            fused=model.decode_head.batch_norm(fused)
            fused=model.decode_head.activation(fused)
            fused=model.decode_head.dropout(fused)
            result['ade_decoder']=fused.float()
    return result

def sample(maps,xy):
    out=torch.empty(len(xy),256,device='cuda')
    with torch.inference_mode():
        for t in sorted(set(map(int,xy[:,2]))):
            rows=np.flatnonzero(xy[:,2]==t)
            grid=(((torch.from_numpy(xy[rows,:2]).float().cuda()+.5)/128)*2-1)[None,None]
            out[torch.from_numpy(rows).cuda()]=F.grid_sample(maps[t:t+1],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
    return out.half().cpu().numpy()

def original_meta(split,vid):
    path=work.spatial.qp(vid) if split=='spatial_val' else work.capacity.fp('train' if split=='train' else 'val',vid)
    with np.load(path)as a:
        return {key:a[key].copy()for key in a.files if key!='decoded'}

def main():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    s=json.loads((ROOT/'summary.json').read_text())
    p=s['protocol']
    assert p==json.loads((ROOT/'protocol.json').read_text()) and p['round']==82 and p['new_heads']==18
    for path,h in p['input_sha256'].items():
        assert file_hash(path)==h
    m=json.loads((ROOT/'manifest.json').read_text())
    assert m==json.loads((work.capacity.ROOT/'manifest.json').read_text())
    data={arm:{split:[]for split in ('train',)+work.SPLITS}for arm in work.NEW}
    for arm in work.NEW:
        for split in ('train',)+work.SPLITS:
            clips=m['train'] if split=='train' else m['val']
            for clip in sorted(clips,key=lambda c:str(c['video_id'])):
                path=work.fp(arm,split,clip['video_id'])
                assert file_hash(path)==s['feature_hashes'][str(path)]
                with np.load(path)as a:
                    package={key:a[key].copy()for key in a.files}
                expected=original_meta(split,clip['video_id'])
                assert package.keys()==expected.keys()|{'decoded'}
                for key,value in expected.items():
                    assert np.array_equal(package[key],value)
                assert package['decoded'].shape==(256 if split=='train' else 2048,256)
                assert package['decoded'].dtype==np.float16 and np.isfinite(package['decoded']).all()
                if split=='train':
                    assert (package['xy'][:,2]==0).all()
                data[arm][split].append(package)
    assert len(s['feature_hashes'])==720 and set(s['feature_hashes'])=={str(path)for path in (ROOT/'features').rglob('*.npz')}
    cost=json.loads((ROOT/'extraction.json').read_text())
    assert len(cost)==400 and file_hash(ROOT/'extraction.json')==s['extraction_sha256']
    assert len({(row['alias'],row['precision'],row['split'],row['video_id'])for row in cost})==400
    feature_replays=[]
    for alias in ('mit','ade'):
        model=work.load_official(alias)
        for precision in ('fp32','bf16'):
            for split in ('train','val'):
                clip=m[split][0]
                frames=clip['frames'][:1] if split=='train' else clip['frames']
                images=[]
                for path in frames:
                    with Image.open(path)as im:
                        array=np.array(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32)
                    images.append(torch.from_numpy(array).permute(2,0,1)/255)
                video=torch.stack(images).cuda()
                maps=manual_maps(model,alias,video,precision)
                for branch,value in maps.items():
                    arm=branch+'_'+precision
                    for key in (('train',) if split=='train' else work.SPLITS):
                        metadata=original_meta(key,clip['video_id'])
                        features=sample(value,metadata['xy'])
                        with np.load(work.fp(arm,key,clip['video_id']))as a:
                            assert np.array_equal(features,a['decoded'])
                        feature_replays.append({'arm':arm,'split':key,'video_id':clip['video_id']})
        del model
        torch.cuda.empty_cache()
    for branch in ('mit_encoder','ade_encoder','ade_decoder'):
        with np.load(work.fp(branch+'_bf16','balanced_val',258))as a:
            with np.load(work.resource_features.ROOT/'preflight_features'/f'{branch}.npz')as old:
                assert np.array_equal(a['xy'][::32],old['xy']) and np.array_equal(a['decoded'][::32],old['decoded'])
    print('verified720 metadata packages and independent precision feature replays',flush=True)
    base=data[work.NEW[0]]['train']
    labels=np.concatenate([a['labels']for a in base])
    y=torch.tensor(labels,device='cuda')
    foreground=torch.where(y>0)[0]
    background=torch.where(y==0)[0]
    sampler_hashes={}
    for seed in range(3):
        generator=torch.Generator(device='cuda').manual_seed(seed)
        ih=hashlib.sha256()
        hist=np.zeros(26,np.int64)
        for step in range(2000):
            f=foreground[torch.randint(len(foreground),(16,),generator=generator,device='cuda')]
            b=background[torch.randint(len(background),(16,),generator=generator,device='cuda')]
            ix=torch.cat([f,b]).cpu().numpy()
            ih.update(ix.tobytes())
            hist+=np.bincount(labels[ix],minlength=26)
        baseline=json.loads(work.ckpt('b14_gt32',seed).with_suffix('.json').read_text())
        assert ih.hexdigest()==baseline['sampled_indices_sha256'] and hist.tolist()==baseline['sampled_class_counts'] and hist.sum()==64000
        sampler_hashes[str(seed)]=ih.hexdigest()
        for arm in work.NEW:
            path=work.ckpt(arm,seed)
            r=json.loads(path.with_suffix('.json').read_text())
            assert r==s['training_reports'][arm][seed] and r['protocol']==p
            assert r['sampled_indices_sha256']==ih.hexdigest() and r['sampled_class_counts']==hist.tolist()
            assert [a['step']for a in r['history']]==list(range(50,2001,50))
            best=max(r['history'],key=lambda a:a['balanced_raw_val_macro'])
            assert best['step']==r['best_step'] and best['balanced_raw_val_macro']==r['best_balanced_raw_val']
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            assert checkpoint['seed']==seed and checkpoint['steps']==2000 and checkpoint['best_step']==best['step'] and checkpoint['protocol']==p
            assert sum(value.numel()for value in checkpoint['model'].values())==r['head_parameters']==6682 and file_hash(path)==r['checkpoint_sha256']
    frozen=json.loads((ROOT/'frozen_selection.json').read_text())
    assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256']
    results={}
    offsets={}
    for arm in work.ARMS:
        outputs={}
        for split in work.SPLITS:
            if arm in work.NEW:
                values_data=data[arm][split]
            else:
                values_data=[]
                for clip in sorted(m['val'],key=lambda c:str(c['video_id'])):
                    path=work.capacity.fp('val',clip['video_id']) if split=='balanced_val' else work.spatial.fp('B14',clip['video_id'])
                    with np.load(path)as a:
                        values_data.append({key:a[key].copy()for key in a.files})
                    if split=='spatial_val':
                        values_data[-1]['labels']=original_meta(split,clip['video_id'])['labels']
            target=np.concatenate([a['labels']for a in values_data])
            times=np.concatenate([a['xy'][:,2]for a in values_data])
            ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in values_data])
            outputs[split]=[]
            for seed in range(3):
                path=work.ckpt(arm,seed)
                assert file_hash(path)==frozen['checkpoint_sha256'][f'{arm}_{seed}']
                state=torch.load(path,map_location='cpu',weights_only=True)['model']
                logits=np.concatenate([linear(a['decoded'],state)for a in values_data])
                with np.load(ROOT/arm/'logits'/f'seed{seed}_{split}.npz')as a:
                    assert np.array_equal(logits,a['logits']) and np.array_equal(target,a['labels']) and np.array_equal(times,a['times']) and np.array_equal(ids,a['video_ids']) and str(a['checkpoint_sha256'])==file_hash(path)
                outputs[split].append({'logits':logits,'labels':target,'times':times})
                if arm in work.NEW and split=='balanced_val':
                    assert abs(independent_eval(logits,target,times,0)['macro_f1_present']-s['training_reports'][arm][seed]['best_balanced_raw_val'])<1e-12
        offsets[arm]={'raw':0.}
        for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
            if arm=='b14_gt32':
                offsets[arm][profile]=p['fixed_reference_offsets'][profile]
                continue
            curve=[]
            best=-1
            choice=None
            for offset in sorted(p['calibration_grid'],key=lambda v:(abs(v),v)):
                values=[independent_eval(a['logits'],a['labels'],a['times'],offset)['macro_f1_present']for a in outputs[split]]
                mean=float(np.mean(values))
                curve.append({'offset':float(offset),'macro_f1_by_seed':values,'mean_macro_f1':mean})
                if mean>best:
                    best=mean
                    choice=float(offset)
            equal(curve,frozen['curves'][arm][profile])
            offsets[arm][profile]=choice
        runs=[]
        for seed in range(3):
            row={'seed':seed}
            for split in work.SPLITS:
                a=outputs[split][seed]
                row[split]={profile:independent_eval(a['logits'],a['labels'],a['times'],offset)for profile,offset in offsets[arm].items()}
            runs.append(row)
        result={'offsets':offsets[arm],'runs':runs,'mean_balanced_profile_macro':float(np.mean([r['balanced_val']['balanced_profile']['macro_f1_present']for r in runs]))}
        equal(result,s['results'][arm])
        results[arm]=result
        print('verified two validation sets/raw42 logits and profiles',arm,flush=True)
    def check_contrast(left,right,expected):
        rebuilt={}
        for split in work.SPLITS:
            rebuilt[split]={}
            for profile in ('raw','balanced_profile','spatial_profile'):
                values=[results[left]['runs'][seed][split][profile]['macro_f1_present']-results[right]['runs'][seed][split][profile]['macro_f1_present']for seed in range(3)]
                rebuilt[split][profile]={'per_seed_macro_difference':values,'mean':float(np.mean(values))}
        equal(rebuilt,expected)
    check_contrast('ade_encoder_fp32','mit_encoder_fp32',s['primary_pretraining_contrast'])
    pairs=[('ade_decoder_fp32','ade_encoder_fp32')]+[(branch+'_bf16',branch+'_fp32')for branch in ('mit_encoder','ade_encoder','ade_decoder')]+[(arm,'b14_gt32')for arm in work.NEW]
    assert set(s['secondary_contrasts'])=={left+'_minus_'+right for left,right in pairs}
    for left,right in pairs:
        check_contrast(left,right,s['secondary_contrasts'][left+'_minus_'+right])
    selected=max(work.ARMS,key=lambda arm:results[arm]['mean_balanced_profile_macro'])
    assert selected==s['validation_selected_arm']==frozen['validation_selected_arm'] and offsets==frozen['offsets'] and frozen['new_candidate_exceeds_B14']==(selected!='b14_gt32')
    for path,h in p['input_sha256'].items():
        assert file_hash(path)==h
    assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt'
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all720_feature_metadata_packages_and18_head_budgets_verified':True,'independent_feature_replays':feature_replays,'three_shared_full_sampler_replays':sampler_hashes,'all18_checkpoint_selection_rules_verified':True,'all42_raw_logits_and12_calibration_curves_recomputed':True,'all_profile_metrics_primary_secondary_contrasts_and_source_selection_recomputed':True,'all3_BF16_branches_reproduce_round81_points':True,'all_frozen_inputs_unchanged':True,'no_new_confirmation_or_deployment_change':True})
    print('DONE independent SegFormer pretraining/precision feature control',selected,flush=True)
if __name__=='__main__':
    main()
