"""Recompute both temporal query sets, all logits and paired statistics."""
import json
from pathlib import Path
import numpy as np
import torch
from semantic.dino_temporal_stride_probe import ROOT,ARMS,SEED,previous,feature_path,freeze
from semantic.semantic_feature_control import read_features
from semantic.verify_dino_resolution import independent_features
from semantic.ytvis import load_dataset,sample_queries
from semantic.model import SemanticHead
from semantic.background_calibration import evaluate,predictions
from semantic.fresh_video_confirmation import bootstrap_video_differences
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics


def main():
    s=json.loads((ROOT/'summary.json').read_text());ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    p=freeze(ds);assert p==s['protocol'];m={'val':p['clips']}
    data={n:read_features('dinov2','val',m,path_resolver=lambda _,split,vid,n=n:feature_path(n,vid))for n in [1,4]}
    lookup={n:{int(c['video_id']):c for c in data[n]}for n in [1,4]}
    for c in p['clips']:
        for stride in [1,4]:
            indices=[c['start']+stride*t for t in range(8)];maps=np.stack([ds.gt_label_map({**c,'start':t},0)for t in indices])
            q,y,xy=sample_queries(maps,per_frame=256,seed=42+c['video_id']);saved=lookup[stride][c['video_id']]
            assert np.array_equal(saved['labels'],y.numpy())and np.array_equal(saved['xy'],xy.numpy())
            for k in ['t_tgt','t_cam']:assert np.array_equal(saved[k],q[k][0].numpy())
            if stride==4:assert saved['global_frame_indices'].tolist()==indices
    for check in s['source0_feature_checks']:
        a,b=lookup[1][check['video_id']],lookup[4][check['video_id']];aa=a['decoded'][a['xy'][:,2]==0];bb=b['decoded'][b['xy'][:,2]==0]
        assert bool(np.array_equal(aa,bb))==check['source0_features_equal']
        assert float(np.abs(aa.astype(np.float32)-bb.astype(np.float32)).max())==check['max_source0_feature_difference']
    outputs={}
    for arm in ARMS:
        packages={}
        for n in [1,4]:
            packages[n]=[]
            for seed in range(3):
                model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval();ck=previous.checkpoint(arm,seed);model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model'])
                with torch.inference_mode():logits=np.concatenate([model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in data[n]])
                path=ROOT/arm/str(n)/'logits'/f'dinov2_{seed}_val.npz'
                with np.load(path)as z:pack={k:z[k].copy()for k in z.files}
                assert np.array_equal(logits,pack['logits'])and str(pack['checkpoint_sha256'])==file_hash(ck)
                assert np.array_equal(pack['labels'],np.concatenate([c['labels']for c in data[n]]))and np.array_equal(pack['times'],np.concatenate([c['xy'][:,2]for c in data[n]]))
                assert np.array_equal(pack['video_ids'],np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int32)for c in data[n]]))
                run=s['results'][arm][str(n)]['runs'][seed];offset=p['fixed_offsets'][arm]
                assert evaluate(pack,0)==run['raw']and evaluate(pack,offset)==run['calibrated']
                pred=predictions(logits,offset)
                for c in p['clips']:
                    mask=pack['video_ids']==c['video_id'];assert classification_metrics(pred[mask],pack['labels'][mask],26)==run['per_video'][str(c['video_id'])]
                outputs[str(path)]=file_hash(path);packages[n].append(pack);del model
        for seed in range(3):
            a,b=packages[1][seed],packages[4][seed];aa=a['logits'][a['times']==0];bb=b['logits'][b['times']==0];r=s['source0_output_checks'][arm][seed]
            assert bool(np.array_equal(aa,bb))==r['source0_logits_equal']and float(np.abs(aa-bb).max())==r['max_source0_logits_difference']
            assert int((predictions(aa,p['fixed_offsets'][arm])!=predictions(bb,p['fixed_offsets'][arm])).sum())==r['source0_label_changes']
        out=s['stride4_minus_stride1'][arm];d=[];per_video={};common={}
        for seed in range(3):d.append(s['results'][arm]['4']['runs'][seed]['calibrated']['macro_f1_present']-s['results'][arm]['1']['runs'][seed]['calibrated']['macro_f1_present'])
        assert d==out['per_seed_macro_difference']and float(np.mean(d))==out['mean_macro_difference']
        for c in p['clips']:
            vid=str(c['video_id']);r1=[s['results'][arm]['1']['runs'][i]['per_video'][vid]for i in range(3)];r4=[s['results'][arm]['4']['runs'][i]['per_video'][vid]for i in range(3)]
            per_video[vid]=float(np.mean([b['macro_f1_present']-a['macro_f1_present']for a,b in zip(r1,r4)]))
            cls=sorted(set(r1[0]['support'])&set(r4[0]['support']),key=int)
            common[vid]={'classes':cls,'difference':float(np.mean([np.mean([b['per_class_f1'][k]-a['per_class_f1'][k]for k in cls])for a,b in zip(r1,r4)]))}
        assert per_video==out['per_video_differences']and common==out['common_class_per_video']
        assert bootstrap_video_differences(list(per_video.values()),seed=SEED)==out['video_bootstrap']
        assert bootstrap_video_differences([common[str(c['video_id'])]['difference']for c in p['clips']],seed=SEED)==out['common_class_bootstrap']
    model=previous.rgb.prior.resolution.load_dino();clip=p['clips'][0]
    assert np.array_equal(independent_features(model,previous.rgb.load_native_video({**clip,'frames':clip['sparse_frames']}).cuda(),lookup[4][clip['video_id']]['xy'],448),lookup[4][clip['video_id']]['decoded']);del model
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    for path,h in s['feature_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'output_sha256':outputs,
        'counts':{'known_videos':18,'query_sets_recomputed':36,'frozen_outputs_recomputed':24,'new_independent_videos':0},
        'direct_feature_check':{'video_id':clip['video_id'],'sparse_features_recomputed_bitwise':True},
        'verification':{'all_dense_sparse_queries_and_global_indices_reproduced':True,'all_logits_and_metrics_recomputed':True,
        'all_source0_checks_and_paired_common_class_statistics_recomputed':True,'all_registered_inputs_unchanged':True,'no_training_calibration_or_deployment_change':True}})
    print('DONE temporal stride independent audit',len(outputs))


if __name__=='__main__':main()
