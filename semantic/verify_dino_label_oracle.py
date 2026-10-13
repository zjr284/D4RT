"""Independently replay oracle labels, teacher sampling and all frozen outputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from semantic.dino_label_oracle import ROOT,ARMS,prior,freeze,read,checkpoint
from semantic.ytvis import load_dataset
from semantic.sampling_ablation import Sampler
from semantic.model import SemanticHead
from semantic.background_calibration import evaluate,predictions,pick_offset
from semantic.run import atomic_json,classification_metrics
from semantic.oracle_probe import file_hash


def main():
    summary=json.loads((ROOT/'summary.json').read_text());selection=json.loads((ROOT/'frozen_selection.json').read_text())
    manifest,protocol=freeze();assert summary['protocol']==protocol and summary['manifest']==manifest
    assert summary['frozen_selection_sha256']==file_hash(ROOT/'frozen_selection.json')and not selection['deployment_candidate']
    ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    changed=0;source_changed=0
    for clip in manifest['train']:
        path=ROOT/'labels'/f"{clip['video_id']}.npz";assert file_hash(path)==summary['corrected_label_sha256'][str(path)]
        with np.load(path)as p:xy=p['xy'];teacher=p['original_labels'];corrected=p['corrected_labels']
        maps=np.stack([ds.gt_label_map(clip,t)for t in range(8)])
        assert np.array_equal(corrected,maps[xy[:,2],xy[:,1],xy[:,0]])
        with np.load(prior.feature_path('448_sam','train',clip['video_id']))as p:
            assert np.array_equal(xy,p['xy'])and np.array_equal(teacher,p['labels'])
        changed+=int((teacher!=corrected).sum());source_changed+=int((teacher[xy[:,2]==0]!=corrected[xy[:,2]==0]).sum())
    assert changed==summary['label_audit']['changed_labels']and source_changed==0
    clips=read('448_oracle','train',manifest)
    teacher=np.concatenate([c['teacher_labels']for c in clips]);times=np.concatenate([c['xy'][:,2]for c in clips]);targets=torch.as_tensor(np.concatenate([c['labels']for c in clips]).astype(np.int64),device='cuda')
    for seed in range(3):
        report=json.loads(checkpoint('448_oracle',seed).with_suffix('.json').read_text());sampler=Sampler(teacher,times,'baseline')
        generator=torch.Generator(device='cuda').manual_seed(seed);h=hashlib.sha256();a=torch.zeros(26,dtype=torch.long,device='cuda');b=a.clone()
        for step in range(2000):
            fg,bg=sampler.draw(generator);idx=torch.cat([fg,bg]);h.update(idx.cpu().numpy().tobytes())
            a+=torch.bincount(sampler.labels[idx],minlength=26);b+=torch.bincount(targets[idx],minlength=26)
        assert h.hexdigest()==report['sampled_indices_sha256']and a.cpu().tolist()==report['sampled_teacher_class_counts']and b.cpu().tolist()==report['sampled_corrected_target_counts']
        baseline=json.loads(checkpoint('448_sam',seed).with_suffix('.json').read_text());assert a.cpu().tolist()==baseline['sampled_class_counts']
        best=max(report['history'],key=lambda x:x['val_macro_f1']);assert best['step']==report['best_step']and best['val_macro_f1']==report['best_val']
        assert report['protocol']==dict(protocol,corrected_label_sha256=summary['corrected_label_sha256'])and report['parameter_count']==10010
    del clips,targets
    hashes={};diagnostics={}
    for arm in ARMS:
        validation=[];runs=[]
        for seed in range(3):
            path=checkpoint(arm,seed);sha=file_hash(path);hashes[f'{arm}_{seed}']=sha;assert sha==selection['checkpoint_sha256'][f'{arm}_{seed}']
            model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval();state=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(state['model'])
            run={'seed':seed,'subsets':{}}
            for split in ('val','confirmation'):
                clips=read(arm,split,manifest)
                with torch.inference_mode():logits=np.concatenate([model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in clips])
                package={'logits':logits,'labels':np.concatenate([c['labels']for c in clips]),'times':np.concatenate([c['xy'][:,2]for c in clips]),
                    'video_ids':np.concatenate([np.full(len(c['labels']),int(c['video_id']),dtype=np.int32)for c in clips])}
                with np.load(ROOT/arm/'logits'/f'dinov2_{seed}_{split}.npz')as p:
                    assert np.array_equal(logits,p['logits'])and str(p['checkpoint_sha256'])==sha
                    for k in ('labels','times','video_ids'):assert np.array_equal(package[k],p[k])
                saved=summary['results'][arm]['runs'][seed];offset=selection['offsets'][arm]
                assert evaluate(package,0)==saved['raw_validation'if split=='val'else'raw_confirmation']
                assert evaluate(package,offset)==saved['calibrated_validation'if split=='val'else'calibrated_confirmation']
                if split=='val':validation.append(package)
                else:
                    pred=predictions(logits,offset)
                    for c in manifest['confirmation']:
                        mask=package['video_ids']==c['video_id'];assert classification_metrics(pred[mask],package['labels'][mask],26)==saved['per_video'][str(c['video_id'])]
                for name,mask in [('source_frame',package['times']==0),('later_frames',package['times']>0)]:
                    run['subsets'][f'{split}_{name}']=classification_metrics(predictions(logits[mask],offset),package['labels'][mask],26)
                del clips
            if arm=='448_oracle':
                report=json.loads(path.with_suffix('.json').read_text());assert report['best_val']==evaluate(validation[-1],0)['macro_f1_present']
            runs.append(run);del model
        if arm=='448_oracle':
            offset,curve=pick_offset(validation,protocol['calibration_grid']);assert offset==selection['offsets'][arm]and curve==selection['validation_curves'][arm]
        else:assert selection['offsets'][arm]==protocol['fixed_offsets'][arm]
        diagnostics[arm]={'runs':runs};print('verified oracle control',arm,flush=True)
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'checkpoint_sha256':hashes,'diagnostics':diagnostics,
        'verification':{'all80_labels_match_GT_and_source_unchanged':True,'all_three_sampling_index_hashes_and_teacher_target_counts_replayed':True,
                        'all18_output_packages_recomputed_bitwise':True,'all_metrics_checkpoint_and_calibration_valid':True,
                        'all_old_inputs_and_models_unchanged':True,'oracle_not_deployment_or_new_confirmation':True}})
    print('DONE oracle independent verification',flush=True)


if __name__=='__main__':main()
