"""Audit frozen resolution x supervision comparison without selecting on confirmation."""
from __future__ import annotations
import gc
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from semantic.dino_resolution_sam_control import (ROOT,ARMS,OFFSETS,resolution,prior,freeze,read,checkpoint,feature_path,feature_hashes,comparison)
from semantic.verify_dino_resolution import independent_features
from semantic.ytvis import load_dataset,load_video,sample_queries
from semantic.background_calibration import evaluate,predictions,pick_offset
from semantic.run import atomic_json,classification_metrics
from semantic.model import SemanticHead
from semantic.oracle_probe import file_hash
from semantic.frozen_error_audit import confusion_counts,decompose


def main():
    summary=json.loads((ROOT/'summary.json').read_text());selection=json.loads((ROOT/'frozen_selection.json').read_text())
    ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    manifest,protocol=freeze(ds);assert manifest==summary['manifest']and protocol==summary['protocol']
    assert file_hash(ROOT/'frozen_selection.json')==summary['frozen_selection_sha256']
    assert feature_hashes(manifest,('train','val','confirmation'))==summary['feature_sha256']
    for split in ('train','val','confirmation'):
        a=read('224_sam',split,manifest);b=read('448_sam',split,manifest)
        for x,y in zip(a,b):
            for k in ('labels','xy','t_tgt','t_cam','video_id'):assert np.array_equal(x[k],y[k])
            if split=='train':
                vid=int(x['video_id']);clip=next(c for c in manifest['train']if c['video_id']==vid)
                with np.load(prior.coverage.PREVIOUS/'teacher'/f'{vid}.npz')as p:maps=p['labels']
                assert np.array_equal(maps[0],ds.gt_label_map(clip,0))
                xy=x['xy'];assert np.array_equal(x['labels'],maps[xy[:,2],xy[:,1],xy[:,0]])
        if split=='train':
            for size,sam in [('224',a),('448',b)]:
                gt=read(size+'_gt',split,manifest)
                assert sum(len(c['labels'])for c in gt)==20480 and sum(len(c['labels'])for c in sam)==162560
                for x,y in zip(gt,sam):
                    mask=y['xy'][:,2]==0
                    for k in ('decoded','labels','xy','t_tgt','t_cam'):assert np.array_equal(x[k],y[k][mask])
                del gt
        if split=='confirmation':
            lookup={int(c['video_id']):c for c in a}
            for clip in manifest[split]:
                maps=np.stack([ds.gt_label_map(clip,t)for t in range(8)])
                q,y,xy=sample_queries(maps,per_frame=256,seed=42+clip['video_id']);cached=lookup[clip['video_id']]
                assert np.array_equal(y.numpy(),cached['labels'])and np.array_equal(xy.numpy(),cached['xy'])
                for k in ('t_tgt','t_cam'):assert np.array_equal(q[k][0].numpy(),cached[k])
        del a,b
    hashes={};diagnostics={}
    for arm in ARMS:
        runs=[];validation=[]
        for seed in range(3):
            path=checkpoint(arm,seed);sha=file_hash(path);hashes[f'{arm}_{seed}']=sha
            assert sha==selection['checkpoint_sha256'][f'{arm}_{seed}']
            report=json.loads(path.with_suffix('.json').read_text());state=torch.load(path,map_location='cpu',weights_only=True)
            if arm=='448_gt':
                assert report['protocol']==dict(protocol,arm=arm,feature_sha256=feature_hashes(manifest,('train','val')))
                best=max(report['history'],key=lambda x:x['val_macro_f1'])
                assert report['best_step']==state['best_step']==best['step']and report['best_val']==best['val_macro_f1']
                old=json.loads(checkpoint('224_gt',seed).with_suffix('.json').read_text())
                assert report['sampled_class_counts']==old['sampled_class_counts']
            assert report['parameter_count']==10010 and report['sampled_class_counts'][0]==256000
            assert sum(report['sampled_class_counts'][1:])==256000
            model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval();model.load_state_dict(state['model'])
            run={'seed':seed,'subsets':{}}
            for split in ('val','confirmation'):
                clips=read(arm,split,manifest)
                with torch.inference_mode():logits=np.concatenate([model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in clips])
                package={'logits':logits,'labels':np.concatenate([c['labels']for c in clips]),'times':np.concatenate([c['xy'][:,2]for c in clips]),
                    'video_ids':np.concatenate([np.full(len(c['labels']),int(c['video_id']),dtype=np.int32)for c in clips])}
                with np.load(ROOT/arm/'logits'/f'dinov2_{seed}_{split}.npz')as p:
                    assert np.array_equal(logits,p['logits'])and str(p['checkpoint_sha256'])==sha
                    for k in ('labels','times','video_ids'):assert np.array_equal(package[k],p[k])
                offset=selection['offsets'][arm];saved=summary['results'][arm]['runs'][seed]
                assert evaluate(package,0)==saved['raw_validation'if split=='val'else'raw_confirmation']
                assert evaluate(package,offset)==saved['calibrated_validation'if split=='val'else'calibrated_confirmation']
                if split=='val':validation.append(package)
                else:
                    pred=predictions(logits,offset)
                    for c in manifest['confirmation']:
                        mask=package['video_ids']==c['video_id'];assert classification_metrics(pred[mask],package['labels'][mask],26)==saved['per_video'][str(c['video_id'])]
                for name,mask in [('all_frames',np.ones(len(logits),dtype=bool)),('source_frame',package['times']==0),('later_frames',package['times']>0)]:
                    pred=predictions(logits[mask],offset);truth=package['labels'][mask];m=classification_metrics(pred,truth,26)
                    m['error_partition']=decompose(confusion_counts(truth,pred));run['subsets'][f'{split}_{name}']=m
                del clips
            if arm=='448_gt':assert evaluate(validation[-1],0)['macro_f1_present']==report['best_val']
            runs.append(run);del model
        if arm=='448_gt':
            offset,curve=pick_offset(validation,protocol['calibration_grid'])
            assert offset==selection['offsets'][arm]and curve==selection['validation_curves'][arm]
        else:assert selection['offsets'][arm]==OFFSETS[arm]
        diagnostics[arm]={'runs':runs};print('verified',arm,flush=True)
    selected=max(ARMS,key=lambda a:np.mean([r['calibrated_validation']['macro_f1_present']for r in summary['results'][a]['runs']]))
    assert selected==selection['validation_selected_arm']==summary['validation_selected_arm']
    terms={'SAM_at448':[('448_sam',1),('448_gt',-1)],'SAM_at224':[('224_sam',1),('224_gt',-1)],
           'resolution_GT':[('448_gt',1),('224_gt',-1)],'resolution_SAM':[('448_sam',1),('224_sam',-1)],
           'interaction':[('448_sam',1),('448_gt',-1),('224_sam',-1),('224_gt',1)]}
    for name,t in terms.items():assert comparison(t,summary['results'],manifest)==summary['contrasts'][name]
    previous=json.loads((resolution.ROOT/'summary.json').read_text())
    assert file_hash(resolution.WEIGHTS)==previous['protocol']['input_sha256'][str(resolution.WEIGHTS)]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=resolution.VENDOR,text=True).strip()==previous['protocol']['dinov2_source_commit']
    model=resolution.load_dino();clip=manifest['confirmation'][0];video=load_video(clip).cuda();feature_checks=[]
    for size in (224,448):
        with np.load(feature_path(f'{size}_sam','confirmation',clip['video_id']))as p:xy=p['xy'];cached=p['decoded']
        assert np.array_equal(independent_features(model,video,xy,size),cached)
        feature_checks.append({'size':size,'video_id':clip['video_id'],'all2048_features_recomputed_bitwise':True})
    del model,video;gc.collect();torch.cuda.empty_cache()
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    assert feature_hashes(manifest,('train','val','confirmation'))==summary['feature_sha256']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'checkpoint_sha256':hashes,'diagnostics':diagnostics,
        'direct_feature_checks':feature_checks,'verification':{'SAM_full_metadata_shared_between_resolutions':True,'GT_prefixes_identical_to_SAM_per_resolution':True,
        'all80_teacher_and_GT_source_labels_match':True,'all40_new_queries_reproduced':True,'all24_logit_packages_recomputed_bitwise':True,
        'all_metrics_sampling_checkpoint_calibration_and_contrasts_valid':True,'all_old_weights_inputs_and_features_unchanged':True},
        'environment':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':torch.cuda.get_device_name(0)}})
    print('DONE round20 independent audit',flush=True)


if __name__=='__main__':main()
