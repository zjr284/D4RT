"""Independent train-only statistics, complete index replay and affine checks."""
import hashlib,json
import numpy as np
import torch
from torch.nn import functional as F
from semantic import feature_standardization_control as work
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.run import atomic_json,classification_metrics
from semantic.oracle_probe import file_hash
ROOT=work.ROOT

def load_data(raw,split,manifest):
    out=[]
    for clip in sorted(manifest['train'] if split=='train' else manifest['val'],key=lambda a:str(a['video_id'])):
        with np.load(work.raw_path(raw,split,clip['video_id']))as a:
            row={k:a[k].copy()for k in a.files}
        if raw=='b14_gt32' and split=='spatial_val':
            with np.load(work.prior.spatial.qp(clip['video_id']))as a:
                row['labels']=a['labels'].copy()
        out.append(row)
    return out

def main():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol']
    assert p==json.loads((ROOT/'protocol.json').read_text()) and p['round']==83 and p['new_heads']==12
    previous=json.loads((work.prior.ROOT/'summary.json').read_text())
    for path,h in {**previous['protocol']['input_sha256'],**p['input_sha256']}.items():
        assert file_hash(path)==h
    assert file_hash(ROOT/'unit_checks.log')==p['unit_checks_sha256'] and 'PASS'in(ROOT/'unit_checks.log').read_text()
    m=json.loads((ROOT/'manifest.json').read_text())
    data={};stats={};package_count=0
    for raw in work.RAW:
        data[raw]={split:load_data(raw,split,m)for split in ('train',)+work.SPLITS}
        for split,rows in data[raw].items():
            for row,reference in zip(rows,data['b14_gt32'][split]):
                assert row['decoded'].dtype==np.float16 and np.isfinite(row['decoded']).all()
                assert np.array_equal(row['xy'],reference['xy']) and np.array_equal(row['labels'],reference['labels'])
                package_count+=1
        training=np.concatenate([a['decoded']for a in data[raw]['train']]).astype(np.float64)
        n=len(training);assert n==20480
        mean64=np.sum(training,axis=0,dtype=np.float64)/n
        variance=np.sum(np.square(training-mean64),axis=0,dtype=np.float64)/n
        std64=np.sqrt(variance);mean=mean64.astype(np.float32)
        scale=np.ones(len(mean),np.float32);active=std64>1e-6;scale[active]=std64[active].astype(np.float32)
        record=s['statistics'][raw]
        with np.load(record['path'])as a:
            assert np.array_equal(mean,a['mean']) and np.array_equal(scale,a['scale']) and np.array_equal(std64.astype(np.float32),a['std'])
        assert file_hash(record['path'])==record['sha256'] and record['samples']==n and record['dimension']==training.shape[1]
        assert record['fallback_scale_channels']==int((~active).sum())
        equal([float(std64.astype(np.float32).min()),float(np.median(std64.astype(np.float32))),float(std64.astype(np.float32).max())],[record['std_min'],record['std_median'],record['std_max']])
        stats[raw]=(mean,scale)
        data['z_'+raw]={}
        for split,rows in data[raw].items():
            z=[]
            for row in rows:
                value=row['decoded'].astype(np.float32)
                np.subtract(value,mean,out=value);np.divide(value,scale,out=value)
                assert value.dtype==np.float32 and np.isfinite(value).all()
                z.append({**row,'decoded':value})
            data['z_'+raw][split]=z
    assert package_count==480
    equal(json.loads((ROOT/'statistics_manifest.json').read_text()),s['statistics'])
    assert file_hash(ROOT/'statistics_manifest.json')==s['statistics_manifest_sha256']
    labels=np.concatenate([a['labels']for a in data['b14_gt32']['train']]);val_labels=np.concatenate([a['labels']for a in data['b14_gt32']['balanced_val']])
    common=sorted(set(map(int,labels[labels>0]))&set(map(int,val_labels[val_labels>0])))
    assert common==p['common_foreground_classes'] and all((a['xy'][:,2]==0).all()for a in data['b14_gt32']['train'])
    foreground=torch.tensor(np.flatnonzero(labels>0),device='cuda');background=torch.tensor(np.flatnonzero(labels==0),device='cuda')
    hashes={}
    for seed in range(3):
        gen=torch.Generator(device='cuda').manual_seed(seed);digest=hashlib.sha256();hist=np.zeros(26,np.int64)
        for _ in range(2000):
            fg=foreground[torch.randint(len(foreground),(16,),generator=gen,device='cuda')]
            bg=background[torch.randint(len(background),(16,),generator=gen,device='cuda')]
            ix=torch.cat([fg,bg]).cpu().numpy();digest.update(ix.tobytes());hist+=np.bincount(labels[ix],minlength=26)
        hashes[str(seed)]=digest.hexdigest();assert hist.sum()==64000
        for arm in work.ARMS:
            report=json.loads(work.ckpt(arm,seed).with_suffix('.json').read_text())
            assert report['sampled_indices_sha256']==digest.hexdigest() and report['sampled_class_counts']==hist.tolist()
            if arm in work.RAW:
                if arm!='b14_gt32':assert report==previous['training_reports'][arm][seed]
                continue
            assert report==s['training_reports'][arm][seed] and report['protocol']==p and report['statistics_sha256']==s['statistics'][arm[2:]]['sha256']
            history=report['history'];assert [a['step']for a in history]==list(range(50,2001,50))
            best=max(history,key=lambda a:a['balanced_raw_val_macro'])
            assert best['step']==report['best_step'] and best['balanced_raw_val_macro']==report['best_balanced_raw_val']
            checkpoint=torch.load(work.ckpt(arm,seed),map_location='cpu',weights_only=True)
            dimension=len(stats[arm[2:]][0])
            assert sum(a.numel()for a in checkpoint['model'].values())==report['head_parameters']==26*(dimension+1)
            assert checkpoint['steps']==2000 and checkpoint['seed']==seed and checkpoint['best_step']==best['step'] and checkpoint['protocol']==p and checkpoint['statistics_sha256']==report['statistics_sha256']
            assert file_hash(work.ckpt(arm,seed))==report['checkpoint_sha256']
    print('verified480 raw packages, independent train moments and12 sampling/budget records',flush=True)
    selection=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256']
    results={};fold_checks={}
    for arm in work.ARMS:
        outputs={split:[]for split in work.SPLITS}
        for seed in range(3):
            checkpoint=work.ckpt(arm,seed);state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model']
            assert file_hash(checkpoint)==selection['checkpoint_sha256'][f'{arm}_{seed}']
            for split in work.SPLITS:
                rows=data[arm][split];logits=np.concatenate([linear(a['decoded'],state)for a in rows]);target=np.concatenate([a['labels']for a in rows]);times=np.concatenate([a['xy'][:,2]for a in rows]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in rows])
                with np.load(ROOT/arm/'logits'/f'seed{seed}_{split}.npz')as a:
                    for k,value in [('logits',logits),('labels',target),('times',times),('video_ids',ids)]:assert np.array_equal(a[k],value),(arm,seed,split,k)
                    assert str(a['checkpoint_sha256'])==file_hash(checkpoint)
                outputs[split].append({'logits':logits,'labels':target,'times':times})
                if arm in work.NEW and split=='balanced_val':assert abs(independent_eval(logits,target,times,0)['macro_f1_present']-s['training_reports'][arm][seed]['best_balanced_raw_val'])<1e-12
            train_logits=np.concatenate([linear(a['decoded'],state)for a in data[arm]['train']]);pred=train_logits.argmax(-1);valpred=outputs['balanced_val'][seed]['logits'].argmax(-1)
            fit=s['training_fits_and_common_class_transfer'][arm][seed]
            with np.load(fit['output_path'])as a:assert np.array_equal(train_logits,a['logits']) and np.array_equal(labels,a['labels'])
            assert file_hash(fit['output_path'])==fit['output_sha256']
            equal(classification_metrics(pred,labels,26),fit['raw_source_training'])
            tr={str(c):float(np.mean(pred[labels==c]==c))for c in common};vr={str(c):float(np.mean(valpred[val_labels==c]==c))for c in common}
            equal(tr,fit['common_FG_train_recall']);equal(vr,fit['common_FG_balanced_val_recall']);equal(float(np.mean(list(tr.values()))),fit['common_FG_macro_train_recall']);equal(float(np.mean(list(vr.values()))),fit['common_FG_macro_balanced_val_recall'])
            if arm in work.NEW:
                mu,scale=stats[arm[2:]];w=state['classifier.weight'].cuda()/torch.tensor(scale,device='cuda');bias=state['classifier.bias'].cuda()-F.linear(torch.tensor(mu,device='cuda')[None],w)[0]
                with torch.inference_mode():folded=np.concatenate([F.linear(torch.from_numpy(a['decoded']).float().cuda(),w,bias).cpu().numpy()for a in data[arm[2:]]['balanced_val']])
                direct=outputs['balanced_val'][seed]['logits'];maximum=float(np.abs(folded-direct).max());tolerance=2e-4+2e-5*float(np.abs(direct).max())
                record={'maximum_raw_fold_logit_difference':maximum,'preregistered_tolerance':tolerance,'label_changes_due_to_numerics':int(np.sum(folded.argmax(-1)!=direct.argmax(-1)))}
                equal(record,s['affine_fold_equivalence'][arm][seed]);assert maximum<=tolerance
                x=data[arm[2:]]['balanced_val'][0]['decoded'][:64].astype(np.float64);w64=state['classifier.weight'].numpy().astype(np.float64);b64=state['classifier.bias'].numpy().astype(np.float64)
                direct64=((x-mu.astype(np.float64))/scale.astype(np.float64))@w64.T+b64
                folded_w=w64/scale.astype(np.float64);folded64=x@folded_w.T+(b64-folded_w@mu.astype(np.float64))
                algebra_error=float(np.abs(direct64-folded64).max());assert algebra_error<1e-9
                fold_checks[f'{arm}_{seed}']={'full_FP32':record,'independent_FP64_first64_maximum_error':algebra_error}
        offsets={'raw':0.}
        for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
            if arm in work.RAW:offsets[profile]=previous['results'][arm]['offsets'][profile];continue
            curve=[];best=-1;choice=None
            for offset in sorted(p['calibration_grid'],key=lambda v:(abs(v),v)):
                values=[independent_eval(a['logits'],a['labels'],a['times'],offset)['macro_f1_present']for a in outputs[split]];mean=float(np.mean(values));curve.append({'offset':float(offset),'macro_f1_by_seed':values,'mean_macro_f1':mean})
                if mean>best:best=mean;choice=float(offset)
            equal(curve,selection['curves'][arm][profile]);offsets[profile]=choice
        runs=[{'seed':seed,**{split:{profile:independent_eval(outputs[split][seed]['logits'],outputs[split][seed]['labels'],outputs[split][seed]['times'],offset)for profile,offset in offsets.items()}for split in work.SPLITS}}for seed in range(3)]
        results[arm]={'offsets':offsets,'runs':runs,'mean_balanced_profile_macro':float(np.mean([a['balanced_val']['balanced_profile']['macro_f1_present']for a in runs]))};equal(results[arm],s['results'][arm])
        print('verified full val/train logits, metrics and fixed offsets',arm,flush=True)
    for arm in work.NEW:
        raw=arm[2:];comparison={}
        for split in work.SPLITS:
            comparison[split]={}
            for profile in ('raw','balanced_profile','spatial_profile'):
                diff=[results[arm]['runs'][seed][split][profile]['macro_f1_present']-results[raw]['runs'][seed][split][profile]['macro_f1_present']for seed in range(3)]
                comparison[split][profile]={'per_seed_macro_difference':diff,'mean':float(np.mean(diff))}
        equal(comparison,s['within_branch_standardization_contrasts'][arm+'_minus_'+raw])
    selected=max(work.ARMS,key=lambda a:results[a]['mean_balanced_profile_macro']);assert selected==s['validation_selected_arm']==selection['validation_selected_arm']
    assert selection['offsets']=={a:r['offsets']for a,r in results.items()} and selection['statistics_manifest_sha256']==s['statistics_manifest_sha256']
    assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt'
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all480_original_feature_packages_metadata_verified':True,'four_independent_training_only_moment_reconstructions':True,'shared_three_full_sampler_replays':hashes,'all12_new_head_budgets_checkpoint_selection_and_statistics_provenance':True,'all48_validation_logits_and24_training_logits_recomputed':True,'eight_new_shared_calibration_curves_all_profile_metrics_and_selection_recomputed':True,'all_common_class_training_validation_recall_recomputed':True,'fold_checks':fold_checks,'all_frozen_inputs_unchanged':True,'no_new_confirmation_or_deployment_change':True})
    print('DONE independent train-only standardization verification',selected,flush=True)
if __name__=='__main__':main()
