"""Point-membership GT replay and independent instance/count statistics."""
import collections
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from semantic.dino_instance_audit import ROOT,SEED,previous
from semantic.ytvis import load_dataset
from semantic.background_calibration import predictions
from semantic.fresh_video_confirmation import bootstrap_video_differences
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

METRIC_NOTE='per-video mean sampled-instance query accuracy difference; not pooled macro F1 or all-instance mean confidence interval'


def point_association(ds,clip,xy,labels):
    out=np.full(len(labels),-1,np.int64)
    for t in range(8):
        rows=np.flatnonzero(xy[:,2]==t);annotations=[];hits=[]
        for a in ds.annotations.get(clip['video_id'],[]):
            global_t=clip['start']+t
            if global_t>=len(a['segmentations'])or a['segmentations'][global_t]is None:continue
            mask=coco_mask.decode(a['segmentations'][global_t])
            if mask.ndim==3:mask=np.any(mask,axis=2)
            raster=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST),dtype=bool)
            annotations.append(a);hits.append(raster[xy[rows,1],xy[rows,0]])
        matrix=np.stack(hits,axis=1)if hits else np.zeros((len(rows),0),bool)
        for j,row in enumerate(rows):
            matches=np.flatnonzero(matrix[j])
            if labels[row]==0:assert len(matches)==0
            else:
                assert labels[row]>0 and len(matches)==1
                a=annotations[int(matches[0])];assert a['category_id']==int(labels[row]);out[row]=a['id']
    return out


def majority(values):
    counts=collections.Counter(map(int,values));return min(counts,key=lambda k:(-counts[k],k))


def close(left,right):assert np.isclose(left,right,rtol=0,atol=1e-14)


def amend_metric_note(summary):
    if 'analysis_amendment'in summary:
        original=ROOT/'summary_initial_metric_note.json'
        assert file_hash(original)==summary['analysis_amendment']['initial_summary_sha256']
        return summary
    original=ROOT/'summary_initial_metric_note.json'
    if original.exists():assert file_hash(original)==file_hash(ROOT/'summary.json')
    else:shutil.copyfile(ROOT/'summary.json',original)
    old_notes={name:c['video_bootstrap']['note']for name,c in summary['contrasts'].items()}
    for c in summary['contrasts'].values():c['video_bootstrap']['note']=METRIC_NOTE
    summary['analysis_amendment']={'initial_summary_sha256':file_hash(original),'initial_summary_path':str(original),
        'reason':'Generic bootstrap helper described macro F1; these contrasts estimate video mean instance accuracy.',
        'original_notes':old_notes,'corrected_note':METRIC_NOTE,'all_numeric_results_inputs_and_model_choices_unchanged':True}
    atomic_json(ROOT/'summary.json',summary);return summary


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];m=s['manifest'];assert p==json.loads((ROOT/'protocol.json').read_text())
    ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    parts=[];ys=[];ts=[];vs=[]
    for c in sorted(m['confirmation'],key=lambda c:str(c['video_id'])):
        with np.load(previous.feature_path(c['video_id']))as z:xy=z['xy'];y=z['labels']
        ids=point_association(ds,c,xy,y);path=ROOT/'associations'/f"{c['video_id']}.npz"
        assert file_hash(path)==s['association_sha256'][str(path)]
        with np.load(path)as z:assert np.array_equal(ids,z['annotation_ids'])and np.array_equal(xy,z['xy'])and np.array_equal(y,z['labels'])
        parts.append(ids);ys.append(y);ts.append(xy[:,2]);vs.append(np.full(len(y),c['video_id'],np.int32))
    inst=np.concatenate(parts);labels=np.concatenate(ys);times=np.concatenate(ts);vids=np.concatenate(vs);fg=labels>0
    old=json.loads((previous.ROOT/'summary.json').read_text());count=0
    for arm in previous.ARMS:
        for seed in range(3):
            with np.load(previous.ROOT/arm/'logits'/f'dinov2_{seed}_confirmation.npz')as z:
                assert np.array_equal(z['labels'],labels)and np.array_equal(z['times'],times)and np.array_equal(z['video_ids'],vids)
                assert str(z['checkpoint_sha256'])==old['protocol']['selection'][arm]['checkpoint_sha256'][str(seed)]
                pred=predictions(z['logits'],old['results'][arm]['offset'])
            r=s['results'][arm]['runs'][seed];objects={};by_video={};by_class={};multi=[]
            for vid in np.unique(vids):
                for aid in np.unique(inst[(vids==vid)&fg]):
                    rows=np.flatnonzero((vids==vid)&(inst==aid));category=int(labels[rows[0]]);assert np.all(labels[rows]==category)
                    obj=r['per_instance'][f'{vid}:{aid}'];accuracy=int((pred[rows]==category).sum())/len(rows)
                    assert obj['queries']==len(rows)and obj['category']==category and obj['query_accuracy']==accuracy
                    assert obj['prediction_counts']==np.bincount(pred[rows],minlength=26).tolist()
                    frames=sorted(set(map(int,times[rows])));modes=[majority(pred[rows[times[rows]==t]])for t in frames]
                    modal=majority(modes);consistency=sum(k==modal for k in modes)/len(modes);correct=sum(k==category for k in modes)/len(modes)
                    assert obj['source_frames']==frames and obj['per_frame_majority_classes']==modes and obj['majority_class']==modal
                    close(obj['majority_consistency'],consistency);close(obj['frame_majority_accuracy'],correct)
                    objects[f'{vid}:{aid}']=accuracy;by_video.setdefault(str(vid),[]).append(accuracy);by_class.setdefault(str(category),[]).append(accuracy)
                    if len(frames)>=2:multi.append((consistency,correct,modal!=category))
            assert set(objects)==set(r['per_instance'])and len(objects)==r['sampled_instances']==141
            close(r['foreground_query_accuracy'],int((pred[fg]==labels[fg]).sum())/int(fg.sum()))
            close(r['instance_mean_query_accuracy'],np.mean(list(objects.values())))
            close(r['class_mean_instance_query_accuracy'],np.mean([np.mean(v)for v in by_class.values()]))
            for vid,v in by_video.items():assert r['per_video'][vid]['sampled_instances']==len(v);close(r['per_video'][vid]['instance_mean_query_accuracy'],np.mean(v))
            for category,v in by_class.items():close(r['per_class_instance_accuracy'][category],np.mean(v))
            assert len(multi)==r['multi_frame_sampled_instances']==137
            close(r['multi_frame_mean_majority_consistency'],np.mean([a for a,b,c in multi]));close(r['multi_frame_mean_majority_accuracy'],np.mean([b for a,b,c in multi]))
            assert r['multi_frame_always_consistent_but_wrong']==sum(a==1 and c for a,b,c in multi);count+=1
    for name,c in s['contrasts'].items():
        left=name.split('_minus_')[0]
        expected={str(v['video_id']):float(np.mean([s['results'][left]['runs'][seed]['per_video'][str(v['video_id'])]['instance_mean_query_accuracy']-
            s['results']['source_gt']['runs'][seed]['per_video'][str(v['video_id'])]['instance_mean_query_accuracy']for seed in range(3)]))for v in m['confirmation']}
        assert expected==c['per_video_instance_accuracy_differences']
        bs=bootstrap_video_differences(list(expected.values()),seed=SEED)
        assert {k:v for k,v in bs.items()if k!='note'}=={k:v for k,v in c['video_bootstrap'].items()if k!='note'}
        assert [s['results'][left]['runs'][i]['instance_mean_query_accuracy']-s['results']['source_gt']['runs'][i]['instance_mean_query_accuracy']for i in range(3)]==c['per_seed_instance_accuracy_differences']
    for k,c in s['primary_class_F1_contributions'].items():
        values=[old['results']['source_gt32']['runs'][i]['calibrated_confirmation']['per_class_f1'][k]-old['results']['source_gt']['runs'][i]['calibrated_confirmation']['per_class_f1'][k]for i in range(3)]
        assert c['mean_class_F1_difference']==float(np.mean(values))and c['contribution_to_macro_difference']==float(np.mean(values))/16
    close(sum(c['contribution_to_macro_difference']for c in s['primary_class_F1_contributions'].values()),old['primary_contrast']['mean_macro_difference'])
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    s=amend_metric_note(s)
    for c in s['contrasts'].values():assert c['video_bootstrap']['note']==METRIC_NOTE
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'counts':{'GT_point_associations_recomputed':27,
        'frozen_output_statistics_recomputed':count,'sampled_instances':141,'multi_frame_sampled_instances':137,'class_F1_contributions':16},
        'verification':{'GT_instances_independently_reassociated_by_point_membership':True,'all_instance_and_weighted_statistics_recomputed':True,
        'all_majority_consistency_correctness_and_wrong_stable_counts_recomputed':True,'all_contrasts_and_seed35_bootstrap_recomputed':True,
        'macro_contribution_sum_recomputed':True,'all_registered_inputs_unchanged':True,'metric_note_amendment_preserves_all_numeric_results':True}})
    print('DONE independent frozen GT-instance audit',count)


if __name__=='__main__':main()
