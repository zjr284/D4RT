"""Independent frozen outputs, metadata, feature values and selection audit."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic.dino_resolution_control import (ROOT, ARMS, BASELINE, prior, freeze, checkpoint,
    feature_path, read, feature_hashes, load_dino)
from semantic.ytvis import load_dataset, load_video, sample_queries
from semantic.background_calibration import evaluate, predictions, pick_offset
from semantic.run import atomic_json, classification_metrics
from semantic.model import SemanticHead
from semantic.oracle_probe import file_hash
from semantic.fresh_video_confirmation import bootstrap_video_differences
from semantic.frozen_error_audit import confusion_counts, decompose


@torch.inference_mode()
def independent_features(model, video, xy, size):
    mean = torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std = torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    images=(F.interpolate(video,size=(size,size),mode='bicubic',align_corners=False)-mean)/std
    with torch.autocast('cuda',dtype=torch.bfloat16): patches=model.forward_features(images)['x_norm_patchtokens']
    grid_size=size//14
    patches=patches.transpose(1,2).reshape(len(video),384,grid_size,grid_size).float()
    result=torch.empty(len(xy),384,device='cuda')
    for frame in np.unique(xy[:,2]):
        rows=np.flatnonzero(xy[:,2]==frame)
        positions=torch.as_tensor(xy[rows,:2].astype(np.float32),device='cuda')
        grid=((positions+.5)/128*2-1).reshape(1,1,len(rows),2)
        result[rows]=F.grid_sample(patches[int(frame):int(frame)+1],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
    return result.half().cpu().numpy()


def main():
    summary=json.loads((ROOT/'summary.json').read_text()); selection=json.loads((ROOT/'frozen_selection.json').read_text())
    dataset=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    manifest,protocol=freeze(dataset); assert summary['manifest']==manifest and summary['protocol']==protocol
    assert summary['frozen_selection_sha256']==file_hash(ROOT/'frozen_selection.json')
    assert feature_hashes(manifest,('train','val','confirmation'))==summary['feature_sha256']
    for split in ('train','val','confirmation'):
        a=read('resolution224',split,manifest); b=read('resolution448',split,manifest)
        for x,y in zip(a,b):
            for key in ('labels','xy','t_tgt','t_cam','video_id'): assert np.array_equal(x[key],y[key])
            if split=='train':
                with np.load(prior.coverage.PREVIOUS/'teacher'/f"{int(x['video_id'])}.npz") as p: maps=p['labels']
                xy=x['xy']; assert np.array_equal(x['labels'],maps[xy[:,2],xy[:,1],xy[:,0]])
        if split=='confirmation':
            lookup={int(c['video_id']):c for c in a}
            for clip in manifest[split]:
                maps=np.stack([dataset.gt_label_map(clip,t) for t in range(8)])
                query,labels,xy=sample_queries(maps,per_frame=256,seed=42+clip['video_id'])
                cached=lookup[clip['video_id']]
                assert np.array_equal(labels.numpy(),cached['labels']) and np.array_equal(xy.numpy(),cached['xy'])
                for key in ('t_tgt','t_cam'): assert np.array_equal(cached[key],query[key][0].numpy())
        del a,b
    checkpoint_hashes={}; diagnostics={}; packages={}
    for arm in ARMS:
        runs=[]; validation=[]
        for seed in range(3):
            path=checkpoint(arm,seed); sha=file_hash(path); checkpoint_hashes[f'{arm}_{seed}']=sha
            assert sha==selection['checkpoint_sha256'][f'{arm}_{seed}']
            model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval()
            state=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(state['model'])
            report=json.loads(path.with_suffix('.json').read_text())
            if arm=='resolution448':
                assert report['protocol']==dict(protocol,arm=arm,feature_sha256=json.loads((ROOT/'training_feature_hashes.json').read_text()))
                best=max(report['history'],key=lambda x:x['val_macro_f1'])
                assert report['best_step']==best['step'] and report['best_val']==best['val_macro_f1']
                assert state['best_step']==best['step'] and state['steps']==2000
                old=json.loads(checkpoint('resolution224',seed).with_suffix('.json').read_text())
                assert report['sampled_class_counts']==old['sampled_class_counts']
            assert report['parameter_count']==10010 and report['sampled_class_counts'][0]==256000
            assert sum(report['sampled_class_counts'][1:])==256000
            run={'seed':seed,'subsets':{}}
            for split in ('val','confirmation'):
                clips=read(arm,split,manifest)
                with torch.inference_mode(): values=np.concatenate([model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy() for c in clips])
                package={'logits':values,'labels':np.concatenate([c['labels']for c in clips]),
                    'times':np.concatenate([c['xy'][:,2]for c in clips]),
                    'video_ids':np.concatenate([np.full(len(c['labels']),int(c['video_id']),dtype=np.int32)for c in clips])}
                with np.load(ROOT/arm/'logits'/f'dinov2_{seed}_{split}.npz') as p:
                    assert np.array_equal(values,p['logits']) and str(p['checkpoint_sha256'])==sha
                    for k in ('labels','times','video_ids'):assert np.array_equal(package[k],p[k])
                saved=summary['results'][arm]['runs'][seed];offset=selection['offsets'][arm]
                assert evaluate(package,0)==saved['raw_validation'if split=='val'else'raw_confirmation']
                assert evaluate(package,offset)==saved['calibrated_validation'if split=='val'else'calibrated_confirmation']
                if split=='val': validation.append(package)
                else:
                    packages[f'{arm}_{seed}']=package
                    pred=predictions(values,offset)
                    for c in manifest['confirmation']:
                        mask=package['video_ids']==c['video_id']
                        assert classification_metrics(pred[mask],package['labels'][mask],26)==saved['per_video'][str(c['video_id'])]
                for name,mask in [('all_frames',np.ones(len(values),dtype=bool)),('source_frame',package['times']==0),('later_frames',package['times']>0)]:
                    pred=predictions(values[mask],offset); truth=package['labels'][mask]
                    metric=classification_metrics(pred,truth,26);metric['error_partition']=decompose(confusion_counts(truth,pred))
                    run['subsets'][f'{split}_{name}']=metric
                del clips
            if arm=='resolution448':assert evaluate(validation[-1],0)['macro_f1_present']==report['best_val']
            runs.append(run);del model
        if arm=='resolution448':
            offset,curve=pick_offset(validation,protocol['calibration_grid'])
            assert offset==selection['offsets'][arm] and curve==selection['validation_curves'][arm]
        else:assert selection['offsets'][arm]==-1.75
        diagnostics[arm]={'runs':runs};print('verified frozen logits/selection',arm,flush=True)
    selected=max(ARMS,key=lambda a:np.mean([r['calibrated_validation']['macro_f1_present']for r in summary['results'][a]['runs']]))
    assert selected==selection['validation_selected_arm']==summary['validation_selected_arm']
    diffs={str(c['video_id']):float(np.mean([summary['results']['resolution448']['runs'][s]['per_video'][str(c['video_id'])]['macro_f1_present']-summary['results']['resolution224']['runs'][s]['per_video'][str(c['video_id'])]['macro_f1_present']for s in range(3)]))for c in manifest['confirmation']}
    assert diffs==summary['contrasts']['per_video_differences']
    assert bootstrap_video_differences(list(diffs.values()),seed=20261018)==summary['contrasts']['video_bootstrap']
    model=load_dino();feature_checks=[]
    for split in ('train','val','confirmation'):
        clip=manifest[split][0]
        with np.load(feature_path('resolution448',split,clip['video_id']))as p: xy=p['xy'];cached=p['decoded']
        video=load_video(clip).cuda();features=independent_features(model,video,xy,448)
        if split=='train':
            source=xy[:,2]==0;features[source]=independent_features(model,video[:1],xy[source],448)
        assert np.array_equal(features,cached)
        feature_checks.append({'split':split,'video_id':clip['video_id'],'queries':len(xy),'all448_values_recomputed_bitwise_equal':True})
        if split=='confirmation':
            with np.load(feature_path('resolution224',split,clip['video_id']))as p: old=p['decoded']
            assert np.array_equal(independent_features(model,video,xy,224),old)
        del video
    del model;gc.collect();torch.cuda.empty_cache()
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    assert feature_hashes(manifest,('train','val','confirmation'))==summary['feature_sha256']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'checkpoint_sha256':checkpoint_hashes,
        'diagnostics':diagnostics,'direct_feature_checks':feature_checks,
        'verification':{'all140_metadata_sets_match_between_arms':True,'all80_teacher_query_labels_unchanged':True,
                        'all40_confirmation_queries_reproduced':True,'all12_logit_packages_recomputed_bitwise':True,
                        'all_raw_calibrated_video_and_subset_metrics_reproduced':True,'sampling_budget_checkpoint_and_calibration_valid':True,
                        'new_confirmation_excludes420_and_selection_uses_validation':True,'all_inputs_and_feature_hashes_unchanged':True},
        'environment':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':torch.cuda.get_device_name(0)}})
    print('DONE independent resolution audit',flush=True)


if __name__=='__main__':main()
