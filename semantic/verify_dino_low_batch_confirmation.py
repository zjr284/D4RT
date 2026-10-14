"""Independent last27 GT/feature/logit/paired statistic verification."""
import json
from pathlib import Path
import numpy as np
import torch
from semantic.dino_low_batch_confirmation import ROOT,ARMS,CANDIDATES,SEED,checkpoint,feature_path,freeze,rgb
from semantic.semantic_feature_control import read_features
from semantic.verify_dino_resolution import independent_features
from semantic.ytvis import load_dataset,sample_queries
from semantic.model import SemanticHead
from semantic.background_calibration import evaluate,predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics


def check_contrast(c,results,m):
    a,b=c['left'],c['right'];d=[results[a]['runs'][i]['calibrated_confirmation']['macro_f1_present']-results[b]['runs'][i]['calibrated_confirmation']['macro_f1_present']for i in range(3)]
    assert d==c['per_seed_macro_difference']and float(np.mean(d))==c['mean_macro_difference']
    v={str(t['video_id']):float(np.mean([results[a]['runs'][i]['per_video'][str(t['video_id'])]['macro_f1_present']-results[b]['runs'][i]['per_video'][str(t['video_id'])]['macro_f1_present']for i in range(3)]))for t in m['confirmation']}
    assert v==c['per_video_differences'];values=np.array(list(v.values()));rng=np.random.default_rng(SEED)
    means=values[rng.integers(0,27,(2000,27))].mean(1);bs=c['video_bootstrap']
    assert float(values.mean())==bs['mean_per_video_difference']and np.quantile(means,[.025,.975]).tolist()==bs['percentile_95_interval']
    assert int((values>0).sum())==bs['positive_videos']and int((values<0).sum())==bs['negative_videos']


def main():
    s=json.loads((ROOT/'summary.json').read_text());ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    m,p=freeze(ds);assert m==s['manifest']and p==s['protocol']
    ids={c['video_id']for c in m['confirmation']};assert len(ids)==27 and not ids.intersection(m['excluded_video_ids'])
    assert len(set(m['excluded_video_ids'])|ids)==len(ds.videos)==607
    clips=read_features('dinov2','confirmation',m,path_resolver=lambda _,split,vid:feature_path(vid));lookup={int(c['video_id']):c for c in clips}
    for clip in m['confirmation']:
        maps=np.stack([ds.gt_label_map(clip,t)for t in range(8)]);q,y,xy=sample_queries(maps,per_frame=256,seed=42+clip['video_id']);saved=lookup[clip['video_id']]
        assert np.array_equal(saved['xy'],xy.numpy())and np.array_equal(saved['labels'],y.numpy())
        for k in ('t_tgt','t_cam'):assert np.array_equal(saved[k],q[k][0].numpy())
    outputs={};count=0
    for arm in ARMS:
        for seed in range(3):
            sha=file_hash(checkpoint(arm,seed));assert sha==p['selection'][arm]['checkpoint_sha256'][str(seed)]
            model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval();model.load_state_dict(torch.load(checkpoint(arm,seed),map_location='cpu',weights_only=True)['model'])
            with torch.inference_mode():logits=np.concatenate([model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in clips])
            path=ROOT/arm/'logits'/f'dinov2_{seed}_confirmation.npz'
            with np.load(path)as z:pack={k:z[k].copy()for k in z.files}
            assert np.array_equal(logits,pack['logits'])and str(pack['checkpoint_sha256'])==sha
            assert np.array_equal(pack['labels'],np.concatenate([c['labels']for c in clips]))and np.array_equal(pack['times'],np.concatenate([c['xy'][:,2]for c in clips]))
            assert np.array_equal(pack['video_ids'],np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int32)for c in clips]))
            run=s['results'][arm]['runs'][seed];offset=p['selection'][arm]['offset'];assert offset==s['results'][arm]['offset']
            assert evaluate(pack,0)==run['raw_confirmation']and evaluate(pack,offset)==run['calibrated_confirmation']
            pred=predictions(logits,offset);cm=np.zeros((26,26),np.int64);np.add.at(cm,(pack['labels'],pred),1)
            support=cm.sum(1);f1=2*np.diag(cm)/np.maximum(cm.sum(0)+support,1)
            assert np.isclose(f1[support>0].mean(),run['calibrated_confirmation']['macro_f1_present'],rtol=0,atol=1e-14)
            for clip in m['confirmation']:
                mask=pack['video_ids']==clip['video_id'];assert classification_metrics(pred[mask],pack['labels'][mask],26)==run['per_video'][str(clip['video_id'])]
            outputs[str(path)]=file_hash(path);count+=1;del model
        print('verified last27',arm,flush=True)
    check_contrast(s['primary_contrast'],s['results'],m)
    for c in s['secondary_descriptive_contrasts'].values():check_contrast(c,s['results'],m)
    selected=max(ARMS,key=lambda a:p['selection'][a]['mean_validation_macro'])
    assert selected==p['selected_before_confirmation']==s['primary_contrast']['left']
    assert s['primary_contrast']['right']=='source_gt'and s['deployment_unchanged']=='native_gt'
    assert s['model_evaluated_videos_after_round']==607 and s['model_unevaluated_videos_remaining']==0
    for path,h in s['feature_sha256'].items():assert file_hash(path)==h
    model=rgb.prior.resolution.load_dino();clip=m['confirmation'][0]
    assert np.array_equal(independent_features(model,rgb.load_native_video(clip).cuda(),lookup[clip['video_id']]['xy'],448),lookup[clip['video_id']]['decoded'])
    del model
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'output_sha256':outputs,
        'counts':{'new_videos':27,'frozen_models':30,'outputs_recomputed':count,'GT_queries_recomputed':len(clips),'paired_contrasts_recomputed':10},
        'direct_feature_check':{'video_id':clip['video_id'],'all_features_recomputed_bitwise':True},
        'verification':{'excluded580_and_all_remaining27_fixed':True,'validation_selection_and_offsets_unchanged':True,
        'all_GTs_queries_and_metadata_reproduced':True,'all_logits_recomputed_bitwise':True,'all_metrics_and_per_video_outputs_recomputed':True,
        'all_seed33_bootstrap_and_contrasts_recomputed':True,'all_inputs_weights_and_features_unchanged':True,'no_deployment_change':True}})
    print('DONE last27 independent confirmation audit',count)


if __name__=='__main__':main()
