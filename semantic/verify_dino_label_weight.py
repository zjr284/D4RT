"""Replay sampling, recompute every validation logit and calibration selection."""
import hashlib
import json
import numpy as np
import torch
from semantic.dino_label_weight_control import ROOT,ARMS,WEIGHTS,checkpoint,prior
from semantic.sampling_ablation import Sampler
from semantic.model import SemanticHead
from semantic.background_calibration import evaluate,pick_offset,predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];m=s['manifest']
    assert p==json.loads((ROOT/'protocol.json').read_text())and set(m)=={'train','val'}
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    clips=prior.read('native_sam','train',m);val=prior.read('native_sam','val',m)
    labels=np.concatenate([c['labels']for c in clips]);times=np.concatenate([c['xy'][:,2]for c in clips]);assert len(labels)==162560
    sampler=Sampler(labels,times,'baseline');replay={}
    for seed in range(3):
        gen=torch.Generator(device='cuda').manual_seed(seed);h=hashlib.sha256()
        hist=np.zeros(26,np.int64);source=hist.copy();later=hist.copy();th=np.zeros(8,np.int64)
        for step in range(2000):
            fg,bg=sampler.draw(gen);ix=torch.cat([fg,bg]).cpu().numpy();h.update(ix.tobytes())
            yy=labels[ix];tt=times[ix];hist+=np.bincount(yy,minlength=26);th+=np.bincount(tt,minlength=8)
            source+=np.bincount(yy[tt==0],minlength=26);later+=np.bincount(yy[tt!=0],minlength=26)
        for arm in ('lambda0','quarter','half'):
            r=s['training_reports'][arm][seed]
            for key,value in [('sampled_class_counts',hist),('sampled_source_class_counts',source),('sampled_later_class_counts',later),('sampled_time_counts',th)]:assert r[key]==value.tolist()
            assert r['sampled_indices_sha256']==h.hexdigest()
            assert r['effective_weight_sum']==int(source.sum())+WEIGHTS[arm]*int(later.sum())
            assert r['parameter_count']==10010 and len(r['history'])==40
            assert [x['step']for x in r['history']]==list(range(50,2001,50))
            best=max(r['history'],key=lambda x:x['val_macro_f1']);assert best['step']==r['best_step']and best['val_macro_f1']==r['best_val']
        replay[str(seed)]={'index_sha256':h.hexdigest(),'source_queries':int(source.sum()),'later_queries':int(later.sum())}
    all_hashes={};curves={};count=0
    for arm in ARMS:
        packages=[]
        for seed in range(3):
            path=ROOT/arm/'logits'/f'dinov2_{seed}_val.npz'
            with np.load(path)as z:pack={k:z[k].copy()for k in z.files}
            sha=file_hash(checkpoint(arm,seed));assert str(pack['checkpoint_sha256'])==sha
            model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval();payload=torch.load(checkpoint(arm,seed),map_location='cpu',weights_only=True)
            model.load_state_dict(payload['model'])
            if arm in WEIGHTS and arm!='one':assert payload['best_step']==s['training_reports'][arm][seed]['best_step']and payload['protocol']==s['training_reports'][arm][seed]['protocol']
            with torch.inference_mode():logits=np.concatenate([model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in val])
            assert np.array_equal(logits,pack['logits'])and np.array_equal(pack['labels'],np.concatenate([c['labels']for c in val]))
            assert np.array_equal(pack['times'],np.concatenate([c['xy'][:,2]for c in val]))
            assert np.array_equal(pack['video_ids'],np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int32)for c in val]))
            runs=s['results'][arm]['runs'][seed];offset=s['results'][arm]['offset']
            assert evaluate(pack,0)==runs['raw_validation']and evaluate(pack,offset)==runs['calibrated_validation']
            pred=predictions(logits,offset);matrix=np.zeros((26,26),np.int64);np.add.at(matrix,(pack['labels'],pred),1)
            support=matrix.sum(1);den=matrix.sum(0)+support;f1=2*np.diag(matrix)/np.maximum(den,1)
            assert np.isclose(f1[support>0].mean(),runs['calibrated_validation']['macro_f1_present'],rtol=0,atol=1e-14)
            packages.append(pack);all_hashes[str(path)]=file_hash(path);count+=1
        if arm in ('lambda0','quarter','half'):
            offset,curve=pick_offset(packages,p['calibration_grid']);assert offset==s['results'][arm]['offset'];curves[arm]=curve
        else:assert s['results'][arm]['offset']==p['fixed_offsets'][arm]
        assert s['results'][arm]['mean_calibrated_validation_macro']==float(np.mean([r['calibrated_validation']['macro_f1_present']for r in s['results'][arm]['runs']]))
    selection=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256']
    assert selection['curves']==curves
    assert max(WEIGHTS,key=lambda a:s['results'][a]['mean_calibrated_validation_macro'])==s['provisional_validation_weight_choice']==selection['provisional_validation_weight_choice']
    assert selection['deployment_unchanged']=='native_gt'
    assert not list(ROOT.glob('**/*confirmation*.npz'))and not list(ROOT.glob('**/*final*.npz'))
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'sampling_replay':replay,'output_sha256':all_hashes,
        'counts':{'new_heads':9,'sampler_replays':3,'validation_outputs_recomputed':count,'training_trajectories_checked':9,'confirmation_outputs':0},
        'verification':{'sampling_indices_counts_and_effective_weights_replayed':True,'all_best_checkpoints_and_calibration_verified':True,
        'all_logits_recomputed_bitwise':True,'metrics_independently_recomputed_from_confusion':True,'all_old_input_hashes_unchanged':True,'no_deployment_change':True}})
    print('DONE label weight independent audit',count)


if __name__=='__main__':main()
