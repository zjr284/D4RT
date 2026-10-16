"""Replay every metadata group and independent coverage-count GT construction."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from semantic.external_coco_adapter import ROOT,ANN,OVIS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    payload=json.loads(ANN.read_bytes());ovis=json.loads(OVIS.read_bytes())
    normalize=lambda x:' '.join(x.lower().replace('_',' ').split())
    mapping={c['id']:next(v['id']for v in ovis['categories']if normalize(c['name'])==normalize(v['name']))for c in payload['categories']if any(normalize(c['name'])==normalize(v['name'])for v in ovis['categories'])}
    assert {str(k):v for k,v in mapping.items()}==s['shared_category_mapping']
    raw=defaultdict(list)
    for image in payload['images']:raw[str(PurePosixPath(image['file_name']).parent)].append(image)
    saved=json.loads((ROOT/'groups.json').read_text());eligible=saved['groups'];excluded=saved['excluded']
    assert file_hash(ROOT/'groups.json')==s['groups_sha256']
    assert len(raw)==s['directory_group_count']and len(eligible)==s['eligible_group_count']and excluded==s['exclusions']
    expected_exclusions=[]
    for key,frames in sorted(raw.items()):
        reason='fixed_pilot'if key=='00000'else'too_short'if len(frames)<8 else'variable_dimensions'if len({(im['height'],im['width'])for im in frames})!=1 else None
        if reason:expected_exclusions.append({'group':key,'reason':reason,'frame_count':len(frames)})
    assert expected_exclusions==excluded
    assert set(raw)=={c['group']for c in eligible}|{e['group']for e in excluded}
    for clip in eligible:
        frames=sorted(raw[clip['group']],key=lambda im:int(PurePosixPath(im['file_name']).stem))
        indices=[int(round(i*(len(frames)-1)/7))for i in range(8)]
        assert indices==clip['selected_indices']and [frames[i]for i in indices]==clip['frames']
        assert clip['frame_count']==len(frames)
    selected=[eligible[int(round(i*(len(eligible)-1)/23))]for i in range(24)]
    assert [c['group']for c in selected]==s['audit_groups']
    by_image=defaultdict(list);image_index={im['id']:im for im in payload['images']}
    for ann in payload['annotations']:by_image[ann['image_id']].append(ann)
    checks=json.loads((ROOT/'raster_checks.json').read_text());assert file_hash(ROOT/'raster_checks.json')==s['raster_checks_sha256']
    assert [(c['group'],c['image_id'],c['time'])for c in checks]==[(clip['group'],clip['frames'][time]['id'],time)for clip in selected for time in (0,7)]
    hist=Counter();empty=0;total=0
    for check in checks:
        im=image_index[check['image_id']];coverage=np.zeros((128,128),np.int32);single_value=np.zeros((128,128),np.int16)
        for ann in by_image[im['id']]:
            mask=coco_mask.decode(ann['segmentation']);assert mask.shape==(im['height'],im['width'])
            raster=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST),dtype=bool)
            coverage+=raster.astype(np.int32)
            single_value[raster]=-1 if ann.get('iscrowd',0)else mapping.get(ann['category_id'],-1)
        labels=np.where(coverage==0,0,np.where(coverage==1,single_value,-1)).astype(np.int16)
        assert digest(labels)==check['mask_sha256']
        assert {str(k):v for k,v in sorted(Counter(map(int,labels.ravel())).items())}==check['pixel_counts']
        pool=np.flatnonzero(labels.ravel()>0)
        if len(pool):
            indices=np.random.default_rng(p['query_seed']+im['id']).choice(pool,256,replace=len(pool)<256)
            xy=np.stack([indices%128,indices//128,np.full(256,check['time'],dtype=np.int64)],axis=1);target=labels.ravel()[indices].astype(np.int64)
            assert np.all(target>0)and np.all(xy[:,:2]>=0)and np.all(xy[:,:2]<=127)
        else:xy=np.empty((0,3),np.int64);target=np.empty(0,np.int64);empty+=1
        assert digest(xy)==check['xy_sha256']and digest(target)==check['labels_sha256']and len(target)==check['query_count']
        assert len(np.unique(xy[:,:2],axis=0))==check['distinct_query_pixels'];hist.update(map(int,target));total+=len(target)
    assert empty==s['empty_known_foreground_frames']and total==s['query_count']and len(checks)==s['audited_frames']
    assert {str(k):v for k,v in sorted(hist.items())}==s['query_label_histogram']
    assert len({im['file_name']for im in payload['images']}&{f for video in ovis['videos']for f in video['file_names']})==s['OVIS_image_path_intersection']
    assert not s['duplicate_screening_complete']and not s['image_acquisition_ready']and not s['formal_confirmation_frozen']and s['no_model_evaluation']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),
        'replayed_groups':len(eligible),'replayed_rasters_and_queries':len(checks),'empty_frames':empty,'query_count':total,
        'verification':{'all_input_fingerprints_unchanged':True,'all_groups_and_exclusions_recomputed':True,'strict_class_mapping_recomputed':True,
        'independent_coverage_count_masks_identical':True,'queries_and_empty_frames_identical':True,'no_duplicate_clearance_or_model_score_claimed':True}})
    print('DONE independent COCO adapter audit',len(eligible),len(checks),flush=True)


if __name__=='__main__':main()
