"""Descriptive GT-instance association of existing frozen semantic outputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from semantic import dino_low_batch_confirmation as previous
from semantic.ytvis import load_dataset
from semantic.background_calibration import predictions
from semantic.fresh_video_confirmation import bootstrap_video_differences
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from semantic.semantic_feature_control import save_npz

ROOT=Path('semantic_artifacts/dino_instance_audit')
SEED=20261035


def associate(ds,clip,xy,labels):
    instances=np.full(len(labels),-1,np.int64)
    for t in range(8):
        ids=np.full((128,128),-1,np.int64);counts=np.zeros((128,128),np.int32)
        for ann in ds.annotations.get(clip['video_id'],[]):
            global_t=clip['start']+t
            if global_t>=len(ann['segmentations'])or ann['segmentations'][global_t]is None:continue
            decoded=coco_mask.decode(ann['segmentations'][global_t])
            if decoded.ndim==3:decoded=decoded.any(axis=2)
            mask=np.asarray(Image.fromarray(decoded.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST),dtype=bool)
            ids[mask&(counts==0)]=int(ann['id']);ids[mask&(counts>0)]=-2;counts[mask]+=1
        rows=np.flatnonzero(xy[:,2]==t);values=ids[xy[rows,1],xy[rows,0]]
        assert np.all(values[labels[rows]>0]>=0)and np.all(values[labels[rows]==0]==-1)
        instances[rows]=values
    categories={int(a['id']):int(a['category_id'])for a in ds.annotations.get(clip['video_id'],[])}
    for ann_id in np.unique(instances[instances>=0]):assert np.all(labels[instances==ann_id]==categories[int(ann_id)])
    return instances


def statistics(pred,labels,times,vids,inst):
    per_instance={};per_video={};by_class={};multi=[]
    for vid in np.unique(vids):
        objects=[]
        for aid in np.unique(inst[(vids==vid)&(inst>=0)]):
            ix=(vids==vid)&(inst==aid);category=int(np.unique(labels[ix]).item());observed=np.unique(times[ix]);modes=[]
            for t in observed:
                counts=np.bincount(pred[ix&(times==t)],minlength=26);modes.append(int(counts.argmax()))
            majority=int(np.bincount(modes,minlength=26).argmax());consistency=float(np.mean(np.array(modes)==majority))
            r={'video_id':int(vid),'annotation_id':int(aid),'category':category,'queries':int(ix.sum()),'source_frames':observed.tolist(),
               'query_accuracy':float((pred[ix]==labels[ix]).mean()),'prediction_counts':np.bincount(pred[ix],minlength=26).tolist(),
               'per_frame_majority_classes':modes,'frame_majority_accuracy':float(np.mean(np.array(modes)==category)),
               'majority_consistency':consistency,'majority_class':majority}
            per_instance[f'{vid}:{aid}']=r;objects.append(r);by_class.setdefault(str(category),[]).append(r['query_accuracy'])
            if len(observed)>=2:multi.append(r)
        assert objects
        per_video[str(vid)]={'sampled_instances':len(objects),'instance_mean_query_accuracy':float(np.mean([r['query_accuracy']for r in objects]))}
    values=list(per_instance.values());fg=labels>0
    return {'foreground_query_accuracy':float((pred[fg]==labels[fg]).mean()),'sampled_instances':len(values),
       'instance_mean_query_accuracy':float(np.mean([r['query_accuracy']for r in values])),
       'class_mean_instance_query_accuracy':float(np.mean([np.mean(v)for v in by_class.values()])),
       'per_class_instance_accuracy':{k:float(np.mean(v))for k,v in by_class.items()},'per_video':per_video,'per_instance':per_instance,
       'multi_frame_sampled_instances':len(multi),'multi_frame_mean_majority_consistency':float(np.mean([r['majority_consistency']for r in multi]))if multi else None,
       'multi_frame_mean_majority_accuracy':float(np.mean([r['frame_majority_accuracy']for r in multi]))if multi else None,
       'multi_frame_always_consistent_but_wrong':sum(r['majority_consistency']==1 and r['majority_class']!=r['category']for r in multi)}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed instance audit exists')
    old=json.loads((previous.ROOT/'summary.json').read_text());v=json.loads((previous.ROOT/'verification.json').read_text())
    assert file_hash(previous.ROOT/'summary.json')==v['summary_sha256'];m=old['manifest'];arms=previous.ARMS
    paths=[previous.ROOT/'summary.json',previous.ROOT/'verification.json',Path('semantic_artifacts/ovis/annotations_train.json'),
       Path('semantic/dino_instance_audit.py'),Path('semantic/OVIS_DINO_INSTANCE_AUDIT_PROTOCOL.md')]
    paths+=[previous.feature_path(c['video_id'])for c in m['confirmation']]
    paths+=[previous.ROOT/a/'logits'/f'dinov2_{s}_confirmation.npz'for a in arms for s in range(3)]
    p={'round':35,'analysis':'descriptive frozen query GT-instance association, not new generalization','arms':list(arms),'seeds':[0,1,2],
       'bootstrap_seed':SEED,'bootstrap_repetitions':2000,'input_sha256':{str(path):file_hash(path)for path in paths}}
    pp=ROOT/'protocol.json'
    if pp.exists()and json.loads(pp.read_text())!=p:raise ValueError('Instance audit protocol changed')
    atomic_json(pp,p);ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    all_ids=[];all_labels=[];all_times=[];all_vids=[]
    for c in sorted(m['confirmation'],key=lambda c:str(c['video_id'])):
        with np.load(previous.feature_path(c['video_id']))as z:xy=z['xy'];labels=z['labels']
        ids=associate(ds,c,xy,labels);path=ROOT/'associations'/f"{c['video_id']}.npz"
        if path.exists():
            with np.load(path)as z:assert np.array_equal(z['annotation_ids'],ids)and np.array_equal(z['xy'],xy)
        else:save_npz(path,annotation_ids=ids,xy=xy,labels=labels,video_id=np.int64(c['video_id']))
        all_ids.append(ids);all_labels.append(labels);all_times.append(xy[:,2]);all_vids.append(np.full(len(labels),c['video_id'],np.int32))
    inst=np.concatenate(all_ids);labels=np.concatenate(all_labels);times=np.concatenate(all_times);vids=np.concatenate(all_vids)
    results={}
    for arm in arms:
        runs=[]
        for seed in range(3):
            with np.load(previous.ROOT/arm/'logits'/f'dinov2_{seed}_confirmation.npz')as z:
                assert np.array_equal(z['labels'],labels)and np.array_equal(z['times'],times)and np.array_equal(z['video_ids'],vids)
                pred=predictions(z['logits'],old['results'][arm]['offset'])
            r=statistics(pred,labels,times,vids,inst)
            assert r['foreground_query_accuracy']==old['results'][arm]['runs'][seed]['calibrated_confirmation']['foreground_accuracy']
            runs.append({'seed':seed,**r})
        results[arm]={'runs':runs};print('instance audit',arm,runs[0]['sampled_instances'],float(np.mean([r['instance_mean_query_accuracy']for r in runs])),flush=True)
    contrasts={}
    for left in ('source_gt32','native_sam','lambda0'):
        delta={str(c['video_id']):float(np.mean([results[left]['runs'][s]['per_video'][str(c['video_id'])]['instance_mean_query_accuracy']-
            results['source_gt']['runs'][s]['per_video'][str(c['video_id'])]['instance_mean_query_accuracy']for s in range(3)]))for c in m['confirmation']}
        contrasts[left+'_minus_source_gt']={'per_video_instance_accuracy_differences':delta,'video_bootstrap':bootstrap_video_differences(list(delta.values()),seed=SEED),
            'per_seed_instance_accuracy_differences':[results[left]['runs'][s]['instance_mean_query_accuracy']-results['source_gt']['runs'][s]['instance_mean_query_accuracy']for s in range(3)]}
    classes=old['results']['source_gt']['runs'][0]['calibrated_confirmation']['support'];contributions={}
    for k in classes:
        d=[old['results']['source_gt32']['runs'][s]['calibrated_confirmation']['per_class_f1'][k]-old['results']['source_gt']['runs'][s]['calibrated_confirmation']['per_class_f1'][k]for s in range(3)]
        contributions[k]={'mean_class_F1_difference':float(np.mean(d)),'contribution_to_macro_difference':float(np.mean(d))/len(classes),'support':classes[k]}
    assert np.isclose(sum(r['contribution_to_macro_difference']for r in contributions.values()),old['primary_contrast']['mean_macro_difference'],rtol=0,atol=1e-14)
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    association_hashes={str(path):file_hash(path)for path in sorted((ROOT/'associations').glob('*.npz'))}
    atomic_json(ROOT/'summary.json',{'protocol':p,'manifest':m,'association_sha256':association_hashes,'results':results,'contrasts':contrasts,
       'primary_class_F1_contributions':contributions,'deployment_unchanged':'native_gt',
       'verification':{'GT_association_only_in_evaluation':True,'foreground_accuracy_reproduced':True,'macro_contribution_sum_reproduced':True,'all_frozen_inputs_unchanged':True},
       'limits':['sampled GT instances only, not all visible instances','accuracy/consistency not instance AP or same-point tracking',
       'known27 set descriptive diagnosis, no new model or video confirmation']})
    print('DONE frozen instance diagnostic',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
