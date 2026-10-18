"""Independent aligned feature assembly, sampling, complete val prediction and selection."""
import hashlib,json
import numpy as np
import torch
from semantic import feature_fusion_control as work
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def standardized(raw,split,m,stats):
    with np.load(stats[raw]['path'])as a:mu=a['mean'].copy();scale=a['scale'].copy()
    rows=[]
    for clip in sorted(m['train'] if split=='train' else m['val'],key=lambda a:str(a['video_id'])):
        with np.load(work.prior.raw_path(raw,split,clip['video_id']))as a:row={k:a[k].copy()for k in a.files}
        if raw=='b14_gt32' and split=='spatial_val':
            with np.load(work.prior.prior.spatial.qp(clip['video_id']))as a:row['labels']=a['labels'].copy()
        value=row['decoded'].astype(np.float32);value-=mu;value/=scale
        assert np.isfinite(value).all() and value.dtype==np.float32
        row['decoded']=value;rows.append(row)
    return rows

def assemble(arm,split,m,stats):
    base=standardized('b14_gt32',split,m,stats)
    if arm==work.REFERENCE:return base
    raw={'b14_mit':'mit_encoder_fp32','b14_ade_encoder':'ade_encoder_fp32','b14_ade_decoder':'ade_decoder_fp32'}.get(arm)
    extra=standardized(raw,split,m,stats)if raw else None
    for i,row in enumerate(base):
        if extra is None:right=np.zeros((len(row['decoded']),256),np.float32)
        else:
            peer=extra[i]
            for key in ('labels','xy','video_id'):assert np.array_equal(peer[key],row[key])
            right=peer['decoded']
        assert row['decoded'].shape[1]==768 and right.shape==(len(row['decoded']),256)
        row['decoded']=np.column_stack((row['decoded'],right));assert row['decoded'].shape[1]==1024 and row['decoded'].dtype==np.float32
    return base

def main():
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text()) and p['round']==84 and p['new_heads']==12
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    previous=json.loads((work.prior.ROOT/'summary.json').read_text());assert json.loads((work.prior.ROOT/'verification.json').read_text())['summary_sha256']==file_hash(work.prior.ROOT/'summary.json')
    stats=previous['statistics'];assert file_hash(work.prior.ROOT/'statistics_manifest.json')==p['statistics_manifest_sha256']
    for record in stats.values():assert file_hash(record['path'])==record['sha256']
    m=json.loads((ROOT/'manifest.json').read_text());assert m==json.loads((work.prior.ROOT/'manifest.json').read_text())
    assert file_hash(ROOT/'unit_checks.log')==p['unit_checks_sha256'] and 'PASS'in(ROOT/'unit_checks.log').read_text()
    training=standardized('b14_gt32','train',m,stats);labels=np.concatenate([a['labels']for a in training]);assert len(labels)==20480 and all((a['xy'][:,2]==0).all()for a in training)
    foreground=torch.tensor(np.flatnonzero(labels>0),device='cuda');background=torch.tensor(np.flatnonzero(labels==0),device='cuda');hashes={}
    for seed in range(3):
        gen=torch.Generator(device='cuda').manual_seed(seed);digest=hashlib.sha256();hist=np.zeros(26,np.int64)
        for _ in range(2000):
            f=foreground[torch.randint(len(foreground),(16,),generator=gen,device='cuda')];b=background[torch.randint(len(background),(16,),generator=gen,device='cuda')];ix=torch.cat([f,b]).cpu().numpy();digest.update(ix.tobytes());hist+=np.bincount(labels[ix],minlength=26)
        hashes[str(seed)]=digest.hexdigest();assert hist.sum()==64000
        ref=json.loads(work.ckpt(work.REFERENCE,seed).with_suffix('.json').read_text());assert ref['sampled_indices_sha256']==digest.hexdigest() and ref['sampled_class_counts']==hist.tolist()
        for arm in work.NEW:
            r=json.loads(work.ckpt(arm,seed).with_suffix('.json').read_text());assert r==s['training_reports'][arm][seed] and r['protocol']==p and r['sampled_indices_sha256']==digest.hexdigest() and r['sampled_class_counts']==hist.tolist()
            assert [row['step']for row in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda a:a['balanced_raw_val_macro'])
            assert best['step']==r['best_step'] and best['balanced_raw_val_macro']==r['best_balanced_raw_val']
            cp=torch.load(work.ckpt(arm,seed),map_location='cpu',weights_only=True)
            assert cp['seed']==seed and cp['steps']==2000 and cp['best_step']==best['step'] and cp['protocol']==p
            assert cp['model']['classifier.weight'].shape==(26,1024) and sum(v.numel()for v in cp['model'].values())==r['head_parameters']==26650 and file_hash(work.ckpt(arm,seed))==r['checkpoint_sha256']
    print('verified shared sampler, matched1024 input and12 budgets',flush=True)
    frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];results={}
    for arm in work.ARMS:
        base=assemble(arm,'train',m,stats);assert np.array_equal(np.concatenate([a['labels']for a in base]),labels)
        if arm=='b14_zero':assert all(np.count_nonzero(a['decoded'][:,768:])==0 for a in base)
        outputs={}
        for split in work.SPLITS:
            rows=assemble(arm,split,m,stats);target=np.concatenate([a['labels']for a in rows]);times=np.concatenate([a['xy'][:,2]for a in rows]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in rows]);outputs[split]=[]
            for seed in range(3):
                path=work.ckpt(arm,seed);assert file_hash(path)==frozen['checkpoint_sha256'][f'{arm}_{seed}'];state=torch.load(path,map_location='cpu',weights_only=True)['model'];logits=np.concatenate([linear(a['decoded'],state)for a in rows])
                with np.load(ROOT/arm/'logits'/f'seed{seed}_{split}.npz')as a:
                    for key,value in [('logits',logits),('labels',target),('times',times),('video_ids',ids)]:assert np.array_equal(a[key],value),(arm,split,seed,key)
                    assert str(a['checkpoint_sha256'])==file_hash(path)
                outputs[split].append({'logits':logits,'labels':target,'times':times})
                if arm in work.NEW and split=='balanced_val':assert abs(independent_eval(logits,target,times,0)['macro_f1_present']-s['training_reports'][arm][seed]['best_balanced_raw_val'])<1e-12
        offsets={'raw':0.}
        for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
            if arm==work.REFERENCE:offsets[profile]=previous['results'][arm]['offsets'][profile];continue
            curve=[];best=-1;choice=None
            for offset in sorted(p['calibration_grid'],key=lambda a:(abs(a),a)):
                values=[independent_eval(a['logits'],a['labels'],a['times'],offset)['macro_f1_present']for a in outputs[split]];mean=float(np.mean(values));curve.append({'offset':float(offset),'macro_f1_by_seed':values,'mean_macro_f1':mean})
                if mean>best:best=mean;choice=float(offset)
            equal(curve,frozen['curves'][arm][profile]);offsets[profile]=choice
        runs=[{'seed':seed,**{split:{profile:independent_eval(outputs[split][seed]['logits'],outputs[split][seed]['labels'],outputs[split][seed]['times'],offset)for profile,offset in offsets.items()}for split in work.SPLITS}}for seed in range(3)]
        results[arm]={'offsets':offsets,'runs':runs,'mean_balanced_profile_macro':float(np.mean([r['balanced_val']['balanced_profile']['macro_f1_present']for r in runs]))};equal(results[arm],s['results'][arm]);print('verified30 val output groups and profiles',arm,flush=True)
    pairs=[('b14_ade_encoder','b14_mit')]+[(arm,'b14_zero')for arm in work.NEW[1:]]+[(arm,work.REFERENCE)for arm in work.NEW]
    assert set(s['contrasts'])=={a+'_minus_'+b for a,b in pairs}
    for left,right in pairs:
        c={}
        for split in work.SPLITS:
            c[split]={}
            for profile in ('raw','balanced_profile','spatial_profile'):
                values=[results[left]['runs'][seed][split][profile]['macro_f1_present']-results[right]['runs'][seed][split][profile]['macro_f1_present']for seed in range(3)];c[split][profile]={'per_seed_macro_difference':values,'mean':float(np.mean(values))}
        equal(c,s['contrasts'][left+'_minus_'+right])
    selected=max(work.ARMS,key=lambda a:results[a]['mean_balanced_profile_macro']);assert selected==s['validation_selected_arm']==frozen['validation_selected_arm']
    eligible=[]
    for arm in work.NEW[1:]:
        diff=[results[arm]['runs'][seed]['balanced_val']['balanced_profile']['macro_f1_present']-results[work.REFERENCE]['runs'][seed]['balanced_val']['balanced_profile']['macro_f1_present']for seed in range(3)]
        if min(diff)>0 and results[arm]['mean_balanced_profile_macro']>max(results[work.REFERENCE]['mean_balanced_profile_macro'],results['b14_zero']['mean_balanced_profile_macro']):eligible.append(arm)
    assert eligible==s['eligible_for_separately_preregistered_confirmation']==frozen['eligible_for_separately_preregistered_confirmation'] and frozen['offsets']=={a:r['offsets']for a,r in results.items()}
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt'
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'independent_FP32_assembly_all_training_and_two_validation_sets':True,'all_frozen_raw_packages_training_statistics_metadata_unchanged':True,'three_complete_sampler_replays':hashes,'all12_matched_width_head_budgets_and_checkpoint_selection':True,'all30_validation_logits_recomputed_bitwise':True,'eight_shared_calibration_curves_all_profile_metrics_and_contrasts_recomputed':True,'selection_and_confirmation_eligibility_independently_recomputed':True,'no_new_confirmation_or_deployment_change':True})
    print('DONE independent complementary feature verification',selected,eligible,flush=True)
if __name__=='__main__':main()
