"""Replay all frozen final predictions and check reservation and no reselection."""
from __future__ import annotations
import json
import numpy as np
import torch
from semantic.dino_final_confirmation import ROOT,ARMS,prior,freeze,read,checkpoint,feature_path,comparison
from semantic.semantic_feature_control import aggregate_hash
from semantic.verify_dino_resolution import independent_features
from semantic.ytvis import load_dataset,load_video,sample_queries
from semantic.model import SemanticHead
from semantic.background_calibration import evaluate,predictions
from semantic.run import atomic_json,classification_metrics
from semantic.oracle_probe import file_hash
from pathlib import Path


def main():
    summary=json.loads((ROOT/'summary.json').read_text());manifest,protocol=freeze()
    assert summary['manifest']==manifest and summary['protocol']==protocol and summary['validation_selected_arm_unchanged']=='native_gt'
    reserved=json.loads((prior.ROOT/'manifest.json').read_text())['final_reserve'];assert reserved==manifest['confirmation']
    ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    old=read('old_sam',manifest);native=read('native_sam',manifest)
    for a,b in zip(old,native):
        for k in ('labels','xy','t_tgt','t_cam','video_id'):assert np.array_equal(a[k],b[k])
    lookup={int(c['video_id']):c for c in old}
    for clip in manifest['confirmation']:
        maps=np.stack([ds.gt_label_map(clip,t)for t in range(8)]);q,y,xy=sample_queries(maps,per_frame=256,seed=42+clip['video_id']);cached=lookup[clip['video_id']]
        assert np.array_equal(cached['labels'],y.numpy())and np.array_equal(cached['xy'],xy.numpy())
        for k in ('t_tgt','t_cam'):assert np.array_equal(cached[k],q[k][0].numpy())
    del old,native
    hashes={}
    for arm in ARMS:
        for seed in range(3):
            path=checkpoint(arm,seed);sha=file_hash(path);hashes[f'{arm}_{seed}']=sha;assert sha==protocol['checkpoint_sha256'][f'{arm}_{seed}']
            model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
            clips=read(arm,manifest)
            with torch.inference_mode():logits=np.concatenate([model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in clips])
            package={'logits':logits,'labels':np.concatenate([c['labels']for c in clips]),'times':np.concatenate([c['xy'][:,2]for c in clips]),
                'video_ids':np.concatenate([np.full(len(c['labels']),int(c['video_id']),dtype=np.int32)for c in clips])}
            with np.load(ROOT/arm/'logits'/f'dinov2_{seed}_confirmation.npz')as p:
                assert np.array_equal(logits,p['logits'])and str(p['checkpoint_sha256'])==sha
                for k in ('labels','times','video_ids'):assert np.array_equal(package[k],p[k])
            saved=summary['results'][arm]['runs'][seed];offset=protocol['offsets'][arm]
            assert evaluate(package,0)==saved['raw_confirmation']and evaluate(package,offset)==saved['calibrated_confirmation']
            pred=predictions(logits,offset)
            for c in manifest['confirmation']:
                mask=package['video_ids']==c['video_id'];assert classification_metrics(pred[mask],package['labels'][mask],26)==saved['per_video'][str(c['video_id'])]
            del model,clips
        print('verified final',arm,flush=True)
    terms={'selected_vs_previous':[('native_gt',1),('old_sam',-1)],'RGB_GT':[('native_gt',1),('old_gt',-1)],'RGB_SAM':[('native_sam',1),('old_sam',-1)],
           'SAM_native':[('native_sam',1),('native_gt',-1)],'SAM_old':[('old_sam',1),('old_gt',-1)],'interaction':[('native_sam',1),('native_gt',-1),('old_sam',-1),('old_gt',1)]}
    for n,t in terms.items():assert comparison(t,summary['results'],manifest)==summary['contrasts'][n]
    feature_hashes={f:aggregate_hash([feature_path(f+'_sam',c['video_id'])for c in manifest['confirmation']])for f in ('old','native')};assert feature_hashes==summary['feature_sha256']
    model=prior.prior.resolution.load_dino();clip=manifest['confirmation'][0];checks=[]
    for f in ('old','native'):
        video=(load_video(clip)if f=='old'else prior.load_native_video(clip)).cuda()
        with np.load(feature_path(f+'_sam',clip['video_id']))as p:xy=p['xy'];cached=p['decoded']
        assert np.array_equal(independent_features(model,video,xy,448),cached);checks.append({'family':f,'video_id':clip['video_id'],'all_features_recomputed_bitwise':True})
        del video
    del model
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'checkpoint_sha256':hashes,'direct_feature_checks':checks,
        'verification':{'reserved40_exact_and_disjoint_from_previous540':True,'all40_queries_shared_and_reproduced':True,
                        'all12_predictions_and_metrics_recomputed_bitwise':True,'all_six_contrasts_and_seed24_bootstrap_recomputed':True,
                        'all_inputs_features_and_model_selection_unchanged':True,'no_training_calibration_or_reselection':True}})
    print('DONE frozen final confirmation audit',flush=True)


if __name__=='__main__':main()
