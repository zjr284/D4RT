"""Replay both generators and recompute supplementary-label validation outputs."""
import hashlib
import json
import numpy as np
import torch
from semantic.dino_gt_exposure_control import ROOT,ARMS,WEIGHTS,checkpoint,prior,pools
from semantic.sampling_ablation import Sampler
from semantic.model import SemanticHead
from semantic.background_calibration import evaluate,pick_offset,predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];m=s['manifest']
    assert p==json.loads((ROOT/'protocol.json').read_text())and set(m)=={'train','val'}
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    groups=pools(m);val=prior.read('native_sam','val',m)
    labels=[np.concatenate([c['labels']for c in group])for group in groups]
    times=[np.concatenate([c['xy'][:,2]for c in group])for group in groups]
    samplers=[Sampler(y,t,'baseline')for y,t in zip(labels,times)];replay={}
    for seed in range(3):
        values=[]
        for j in range(2):
            gen=torch.Generator(device='cuda').manual_seed(seed if j==0 else 20261030+seed)
            h=hashlib.sha256();hist=np.zeros(26,np.int64);th=np.zeros(8,np.int64)
            for step in range(2000):
                fg,bg=samplers[j].draw(gen);ix=torch.cat([fg,bg]).cpu().numpy();h.update(ix.tobytes())
                hist+=np.bincount(labels[j][ix],minlength=26);th+=np.bincount(times[j][ix],minlength=8)
            assert hist.sum()==512000 and hist[0]==256000
            prefix='source'if j==0 else'prop'
            for arm in WEIGHTS:
                r=s['training_reports'][arm][seed]
                assert r[prefix+'_indices_sha256']==h.hexdigest()and r[prefix+'_class_counts']==hist.tolist()and r[prefix+'_time_counts']==th.tolist()
            values.append({'group':prefix,'index_sha256':h.hexdigest(),'time_counts':th.tolist()})
        assert values[0]['time_counts']==[512000,0,0,0,0,0,0,0]and values[1]['time_counts'][0]==0
        baseline=json.loads(checkpoint('source_gt',seed).with_suffix('.json').read_text())
        for arm in WEIGHTS:
            r=s['training_reports'][arm][seed];assert r['source_class_counts']==baseline['sampled_class_counts']
            assert r['parameter_count']==10010 and [x['step']for x in r['history']]==list(range(50,2001,50))
            best=max(r['history'],key=lambda x:x['val_macro_f1']);assert r['best_step']==best['step']and r['best_val']==best['val_macro_f1']
        replay[str(seed)]=values
    selection=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256']
    curves={};outputs={}
    for arm in ARMS:
        packages=[]
        for seed in range(3):
            path=ROOT/arm/'logits'/f'dinov2_{seed}_val.npz'
            with np.load(path)as z:pack={k:z[k].copy()for k in z.files}
            sha=file_hash(checkpoint(arm,seed));assert str(pack['checkpoint_sha256'])==sha==selection['checkpoint_sha256'][f'{arm}_{seed}']
            model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval();payload=torch.load(checkpoint(arm,seed),map_location='cpu',weights_only=True);model.load_state_dict(payload['model'])
            if arm in WEIGHTS:assert payload['best_step']==s['training_reports'][arm][seed]['best_step']and payload['protocol']==s['training_reports'][arm][seed]['protocol']
            with torch.inference_mode():logits=np.concatenate([model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in val])
            assert np.array_equal(logits,pack['logits'])and np.array_equal(pack['labels'],np.concatenate([c['labels']for c in val]))
            assert np.array_equal(pack['times'],np.concatenate([c['xy'][:,2]for c in val]))
            assert np.array_equal(pack['video_ids'],np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int32)for c in val]))
            r=s['results'][arm]['runs'][seed];off=s['results'][arm]['offset']
            assert evaluate(pack,0)==r['raw_validation']and evaluate(pack,off)==r['calibrated_validation']
            pred=predictions(logits,off);cm=np.zeros((26,26),np.int64);np.add.at(cm,(pack['labels'],pred),1)
            support=cm.sum(1);f1=2*np.diag(cm)/np.maximum(cm.sum(0)+support,1)
            assert np.isclose(f1[support>0].mean(),r['calibrated_validation']['macro_f1_present'],rtol=0,atol=1e-14)
            packages.append(pack);outputs[str(path)]=file_hash(path)
        if arm in WEIGHTS:
            off,curve=pick_offset(packages,p['calibration_grid']);assert off==s['results'][arm]['offset'];curves[arm]=curve
        else:assert s['results'][arm]['offset']==p['source_gt_fixed_offset']
        assert s['results'][arm]['mean_calibrated_validation_macro']==float(np.mean([r['calibrated_validation']['macro_f1_present']for r in s['results'][arm]['runs']]))
    assert curves==selection['curves']
    assert max(ARMS,key=lambda a:s['results'][a]['mean_calibrated_validation_macro'])==s['provisional_validation_choice']==selection['provisional_validation_choice']
    assert selection['deployment_unchanged']=='native_gt'and not list(ROOT.glob('**/*confirmation*.npz'))
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'sampling_replay':replay,'output_sha256':outputs,
        'counts':{'new_heads':6,'generator_replays':6,'validation_outputs_recomputed':9,'confirmation_outputs':0},
        'verification':{'both_sampling_generators_replayed':True,'source_GT_exposure_matches_history':True,'best_checkpoints_and_calibration_verified':True,
        'all_logits_recomputed_bitwise':True,'metrics_recomputed_from_confusion':True,'old_inputs_unchanged':True,'no_deployment_change':True}})
    print('DONE GT exposure independent audit',len(outputs))


if __name__=='__main__':main()
