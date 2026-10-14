"""Native RGB audit and disclosed correction to the preregistered bootstrap seed."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from semantic.dino_native_rgb_control import ROOT,ARMS,prior,freeze,read,checkpoint,feature_path,feature_hashes
from semantic.verify_dino_resolution import independent_features
from semantic.ytvis import load_dataset,sample_queries
from semantic.model import SemanticHead
from semantic.background_calibration import evaluate,predictions,pick_offset
from semantic.fresh_video_confirmation import bootstrap_video_differences
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics


def main():
    summary_path=ROOT/'summary.json';summary=json.loads(summary_path.read_text());selection=json.loads((ROOT/'frozen_selection.json').read_text())
    ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    manifest,protocol=freeze(ds);assert summary['manifest']==manifest and summary['protocol']==protocol
    assert summary['frozen_selection_sha256']==file_hash(ROOT/'frozen_selection.json')
    reserve={c['video_id']for c in manifest['final_reserve']};assert len(reserve)==40
    assert not reserve&set(manifest['excluded_video_ids'])and not reserve&{c['video_id']for c in manifest['confirmation']}
    for v in reserve:assert not any(feature_path(a,'confirmation',v).exists()for a in ARMS)
    assert feature_hashes(manifest,('train','val','confirmation'))==summary['feature_sha256']
    for split in ('train','val','confirmation'):
        old=read('old_sam',split,manifest);native=read('native_sam',split,manifest)
        for a,b in zip(old,native):
            for k in ('labels','xy','t_tgt','t_cam','video_id'):assert np.array_equal(a[k],b[k])
        if split=='train':
            for family,clips in [('old',old),('native',native)]:
                gt=read(family+'_gt',split,manifest)
                for a,b in zip(gt,clips):
                    mask=b['xy'][:,2]==0
                    for k in ('decoded','labels','xy','t_tgt','t_cam'):assert np.array_equal(a[k],b[k][mask])
                del gt
        if split=='confirmation':
            lookup={int(c['video_id']):c for c in old}
            for clip in manifest[split]:
                maps=np.stack([ds.gt_label_map(clip,t)for t in range(8)]);q,y,xy=sample_queries(maps,per_frame=256,seed=42+clip['video_id']);c=lookup[clip['video_id']]
                assert np.array_equal(y.numpy(),c['labels'])and np.array_equal(xy.numpy(),c['xy'])
                for k in ('t_tgt','t_cam'):assert np.array_equal(q[k][0].numpy(),c[k])
        del old,native
    hashes={}
    for arm in ARMS:
        validation=[]
        for seed in range(3):
            path=checkpoint(arm,seed);sha=file_hash(path);hashes[f'{arm}_{seed}']=sha;assert sha==selection['checkpoint_sha256'][f'{arm}_{seed}']
            report=json.loads(path.with_suffix('.json').read_text());state=torch.load(path,map_location='cpu',weights_only=True)
            if arm.startswith('native_'):
                assert report['protocol']==dict(protocol,arm=arm,feature_sha256=feature_hashes(manifest,('train','val')))
                best=max(report['history'],key=lambda x:x['val_macro_f1']);assert report['best_step']==best['step']and report['best_val']==best['val_macro_f1']
                baseline=json.loads(checkpoint('old_'+arm.split('_')[1],seed).with_suffix('.json').read_text());assert report['sampled_class_counts']==baseline['sampled_class_counts']
            model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval();model.load_state_dict(state['model'])
            for split in ('val','confirmation'):
                clips=read(arm,split,manifest)
                with torch.inference_mode():logits=np.concatenate([model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in clips])
                package={'logits':logits,'labels':np.concatenate([c['labels']for c in clips]),'times':np.concatenate([c['xy'][:,2]for c in clips]),'video_ids':np.concatenate([np.full(len(c['labels']),int(c['video_id']),dtype=np.int32)for c in clips])}
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
                del clips
            if arm.startswith('native_'):assert report['best_val']==evaluate(validation[-1],0)['macro_f1_present']
            del model
        if arm.startswith('native_'):
            offset,curve=pick_offset(validation,protocol['calibration_grid']);assert offset==selection['offsets'][arm]and curve==selection['validation_curves'][arm]
        else:assert selection['offsets'][arm]==protocol['fixed_offsets'][arm]
        print('verified native RGB',arm,flush=True)
    selected=max(ARMS,key=lambda a:np.mean([r['calibrated_validation']['macro_f1_present']for r in summary['results'][a]['runs']]))
    assert selected==summary['validation_selected_arm']==selection['validation_selected_arm']
    # Preserve the original seed20 helper analysis; seed22 was fixed in the protocol.
    original_path=ROOT/'summary_initial_bootstrap_seed20.json'
    if not original_path.exists():atomic_json(original_path,summary)
    original=json.loads(original_path.read_text());original_intervals={}
    for name,c in summary['contrasts'].items():
        values=list(c['per_video_differences'].values());expected=bootstrap_video_differences(values,seed=20261020)
        assert original['contrasts'][name]['video_bootstrap']==expected
        original_intervals[name]=expected;c['video_bootstrap']=bootstrap_video_differences(values,seed=20261022)
    summary['analysis_amendment']={'reason':'aggregation reused round20 seed, corrected to preregistered seed20261022 without model/parameter changes',
        'initial_summary_sha256':file_hash(original_path),'initial_bootstrap_seed':20261020,'preregistered_bootstrap_seed':20261022,'initial_video_bootstrap':original_intervals}
    model=prior.resolution.load_dino();checks=[]
    for split in ('train','val','confirmation'):
        clip=manifest[split][0];frames=[]
        for path in clip['frames']:
            with Image.open(path)as im:frames.append(np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1))
        # Match the declared CPU float32 scaling and contiguous NCHW layout.
        video=(torch.from_numpy(np.stack(frames)).contiguous()/255).cuda()
        with np.load(feature_path('native_sam',split,clip['video_id']))as p:xy=p['xy'];cached=p['decoded']
        computed=independent_features(model,video,xy,448)
        if split=='train':
            mask=xy[:,2]==0;computed[mask]=independent_features(model,video[:1],xy[mask],448)
        assert np.array_equal(computed,cached);checks.append({'split':split,'video_id':clip['video_id'],'originalJPEG448_features_bitwise':True})
        del video
    del model
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    atomic_json(summary_path,summary)
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(summary_path),'checkpoint_sha256':hashes,'direct_feature_checks':checks,
        'verification':{'all24_output_packages_recomputed_bitwise':True,'all_queries_labels_GT_prefixes_and_sampling_match':True,
                        'all_metrics_checkpoint_calibration_and_validation_choice_valid':True,'preregistered_bootstrap_seed22_recomputed_initial_seed20_preserved':True,
                        'all_old_inputs_and_features_unchanged':True,'final40_reserve_not_evaluated':True}})
    print('DONE native RGB independent audit, bootstrap correction disclosed',flush=True)


if __name__=='__main__':main()
