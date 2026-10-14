"""Frozen descriptive temporal-stride diagnostic on18 prior validation clips."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from semantic import dino_low_batch_confirmation as previous
from semantic.semantic_feature_control import infer,read_features,save_npz,patch_at_queries
from semantic.ytvis import load_dataset,sample_queries
from semantic.fresh_video_confirmation import bootstrap_video_differences
from semantic.background_calibration import evaluate,predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics

ROOT=Path('semantic_artifacts/dino_temporal_stride_probe')
ARMS=('source_gt','source_gt32','lambda0','native_sam')
SEED=20261034


def feature_path(stride,vid):
    if stride==1:return previous.rgb.feature_path('native_sam','val',vid)
    return ROOT/'features'/f'{vid}.npz'


def freeze(ds):
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((previous.ROOT/'summary.json').read_text());v=json.loads((previous.ROOT/'verification.json').read_text())
    assert file_hash(previous.ROOT/'summary.json')==v['summary_sha256']
    source=json.loads((previous.rgb.ROOT/'summary.json').read_text())['manifest']['val'];clips=[];excluded=[]
    for c in source:
        names=ds.videos[c['video_id']]['file_names'];indices=[c['start']+4*t for t in range(8)]
        if indices[-1]>=len(names):excluded.append({'video_id':c['video_id'],'reason':'too_short','length':len(names)});continue
        frames=[str(ds.frame_path(names[t]))for t in indices]
        clips.append({**c,'sparse_frames':frames,'global_frame_indices':indices})
    assert [c['video_id']for c in clips]==[258,45,130,329,28,265,372,286,109,527,38,480,294,385,86,299,167,518]
    assert [c['video_id']for c in excluded]==[414,493]
    paths=[previous.ROOT/'summary.json',previous.ROOT/'verification.json',previous.rgb.ROOT/'summary.json',
           Path('semantic/dino_temporal_stride_probe.py'),Path('semantic/OVIS_DINO_TEMPORAL_STRIDE_PROTOCOL.md'),
           Path('semantic_artifacts/ovis/annotations_train.json'),previous.rgb.prior.resolution.WEIGHTS]
    paths+=[previous.checkpoint(a,s)for a in ARMS for s in range(3)]
    paths+=[feature_path(1,c['video_id'])for c in clips]+[Path(f)for c in clips for k in ('frames','sparse_frames')for f in c[k]]
    for path,h in old['protocol']['input_sha256'].items():
        if path.startswith('semantic_artifacts/vendor/dinov2/')or path==str(previous.rgb.prior.resolution.WEIGHTS):assert file_hash(path)==h;paths.append(Path(path))
    p={'round':34,'analysis':'descriptive known validation temporal stress, no new video confirmation','strides':[1,4],
        'clips':clips,'excluded_by_length':excluded,'seeds':[0,1,2],'arms':list(ARMS),'bootstrap_seed':SEED,'bootstrap_repetitions':2000,
        'fixed_offsets':{a:old['protocol']['selection'][a]['offset']for a in ARMS},'deployment_change':False,'training_updates':0,
        'input_sha256':{str(p):file_hash(p)for p in paths}}
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=p:raise ValueError('Temporal stress protocol changed')
    atomic_json(path,p);return p


def cache(ds,p):
    tasks=[c for c in p['clips']if not feature_path(4,c['video_id']).exists()]
    if not tasks:return
    model=previous.rgb.prior.resolution.load_dino()
    for c in tasks:
        maps=np.stack([ds.gt_label_map({**c,'start':t},0)for t in c['global_frame_indices']])
        q,y,xy=sample_queries(maps,per_frame=256,seed=42+c['video_id'])
        video=previous.rgb.load_native_video({**c,'frames':c['sparse_frames']}).cuda()
        tokens=previous.rgb.prior.resolution.extract_tokens(model,video,448)
        with torch.inference_mode():features=patch_at_queries(tokens,xy.numpy()).half().cpu().numpy()
        assert np.isfinite(features).all()
        save_npz(feature_path(4,c['video_id']),decoded=features,labels=y.numpy().astype(np.int16),xy=xy.numpy().astype(np.int16),
            t_tgt=q['t_tgt'][0].numpy(),t_cam=q['t_cam'][0].numpy(),video_id=np.int64(c['video_id']),global_frame_indices=np.array(c['global_frame_indices']))
        print('cached stride4',c['video_id'],flush=True);del video,tokens
    del model;gc.collect();torch.cuda.empty_cache()


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed temporal stress exists')
    ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'));p=freeze(ds);cache(ds,p)
    m={'val':p['clips']};features={str(feature_path(n,c['video_id'])):file_hash(feature_path(n,c['video_id']))for n in [1,4]for c in p['clips']}
    data={n:read_features('dinov2','val',m,path_resolver=lambda _,split,vid,n=n:feature_path(n,vid))for n in [1,4]}
    source=[]
    for dense,sparse in zip(data[1],data[4]):
        mask1=dense['xy'][:,2]==0;mask4=sparse['xy'][:,2]==0
        assert np.array_equal(dense['xy'][mask1],sparse['xy'][mask4])and np.array_equal(dense['labels'][mask1],sparse['labels'][mask4])
        diff=np.abs(dense['decoded'][mask1].astype(np.float32)-sparse['decoded'][mask4].astype(np.float32))
        source.append({'video_id':int(dense['video_id']),'source0_features_equal':bool(np.array_equal(dense['decoded'][mask1],sparse['decoded'][mask4])),
            'max_source0_feature_difference':float(diff.max())})
    results={};contrasts={};consistency={}
    for arm in ARMS:
        offsets=p['fixed_offsets'][arm];results[arm]={};packages={}
        for stride in [1,4]:
            runs=[];packages[stride]=[]
            for seed in range(3):
                pack=infer('dinov2',seed,'val',data[stride],checkpoint_path=previous.checkpoint(arm,seed),output_root=ROOT/arm/str(stride));packages[stride].append(pack)
                pred=predictions(pack['logits'],offsets)
                per_video={str(c['video_id']):classification_metrics(pred[pack['video_ids']==c['video_id']],pack['labels'][pack['video_ids']==c['video_id']],26)for c in p['clips']}
                runs.append({'seed':seed,'raw':evaluate(pack,0),'calibrated':evaluate(pack,offsets),'per_video':per_video})
            results[arm][str(stride)]={'offset':offsets,'runs':runs}
        consistency[arm]=[]
        for seed in range(3):
            a,b=packages[1][seed],packages[4][seed];m1=a['times']==0;m4=b['times']==0
            consistency[arm].append({'seed':seed,'source0_logits_equal':bool(np.array_equal(a['logits'][m1],b['logits'][m4])),
                'max_source0_logits_difference':float(np.abs(a['logits'][m1]-b['logits'][m4]).max()),
                'source0_label_changes':int((predictions(a['logits'][m1],offsets)!=predictions(b['logits'][m4],offsets)).sum())})
        differences={};common={}
        for c in p['clips']:
            vid=str(c['video_id']);differences[vid]=float(np.mean([results[arm]['4']['runs'][s]['per_video'][vid]['macro_f1_present']-results[arm]['1']['runs'][s]['per_video'][vid]['macro_f1_present']for s in range(3)]))
            cls=sorted(set(results[arm]['1']['runs'][0]['per_video'][vid]['support'])&set(results[arm]['4']['runs'][0]['per_video'][vid]['support']),key=int)
            common[vid]={'classes':sorted(cls,key=int),'difference':float(np.mean([np.mean([results[arm]['4']['runs'][s]['per_video'][vid]['per_class_f1'][k]-results[arm]['1']['runs'][s]['per_video'][vid]['per_class_f1'][k]for k in cls])for s in range(3)]))}
        d=[results[arm]['4']['runs'][s]['calibrated']['macro_f1_present']-results[arm]['1']['runs'][s]['calibrated']['macro_f1_present']for s in range(3)]
        contrasts[arm]={'per_seed_macro_difference':d,'mean_macro_difference':float(np.mean(d)),'per_video_differences':differences,
            'video_bootstrap':bootstrap_video_differences(list(differences.values()),seed=SEED),'common_class_per_video':common,
            'common_class_bootstrap':bootstrap_video_differences([common[str(c['video_id'])]['difference']for c in p['clips']],seed=SEED)}
        print('temporal diagnostic',arm,[float(np.mean([r['calibrated']['macro_f1_present']for r in results[arm][str(n)]['runs']]))for n in [1,4]],flush=True)
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    for path,h in features.items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'feature_sha256':features,'source0_feature_checks':source,'source0_output_checks':consistency,
        'results':results,'stride4_minus_stride1':contrasts,'deployment_unchanged':'native_gt',
        'verification':{'all_frozen_inputs_and_weights_unchanged':True,'no_training_recalibration_or_new_independent_videos':True},
        'limits':['known validation videos, descriptive only','queries and class supports differ with source frames','no object tracking alignment or true frame seconds']})
    print('DONE descriptive temporal stress',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
