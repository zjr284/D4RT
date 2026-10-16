"""Metadata-only COCO directory grouping and conditional foreground queries."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from semantic.external_holdout_readiness import category_mapping
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_coco_adapter')
ANN=Path('semantic_artifacts/external_annotation_acquisition/downloads/lvvis_validation_oracle.data')
OVIS=Path('semantic_artifacts/ovis/annotations_train.json')
SEED=20261041


def groups(images):
    by_parent=defaultdict(list);seen_ids=set();seen_paths=set()
    for image in images:
        path=PurePosixPath(image['file_name'])
        if path.is_absolute()or'..'in path.parts or'\\'in image['file_name']or len(path.parts)!=2 or path.suffix.lower()!='.jpg':raise ValueError('unsafe/non-frame image path')
        if not path.stem.isdigit()or not path.parent.name.isdigit():raise ValueError('nonnumeric directory/frame')
        if image['id']in seen_ids or str(path)in seen_paths:raise ValueError('duplicate image id/path')
        if not isinstance(image['height'],int)or not isinstance(image['width'],int)or min(image['height'],image['width'])<=0:raise ValueError('invalid image dimensions')
        seen_ids.add(image['id']);seen_paths.add(str(path));by_parent[str(path.parent)].append(image)
    records=[];excluded=[]
    for parent,frames in sorted(by_parent.items()):
        frames=sorted(frames,key=lambda im:int(PurePosixPath(im['file_name']).stem))
        numbers=[int(PurePosixPath(im['file_name']).stem)for im in frames]
        if len(set(numbers))!=len(numbers):raise ValueError('duplicate numeric frame')
        reason='fixed_pilot'if parent=='00000'else'too_short'if len(frames)<8 else'variable_dimensions'if len({(im['height'],im['width'])for im in frames})!=1 else None
        if reason:excluded.append({'group':parent,'reason':reason,'frame_count':len(frames)});continue
        indexes=np.rint(np.linspace(0,len(frames)-1,8)).astype(int).tolist()
        assert len(set(indexes))==8
        records.append({'group':parent,'frame_count':len(frames),'selected_indices':indexes,'frames':[frames[i]for i in indexes]})
    return records,excluded


def label_map(annotations,mapping,height,width,size=128):
    out=np.zeros((size,size),np.int16);occupied=np.zeros_like(out,dtype=bool)
    for ann in annotations:
        mask=coco_mask.decode(ann['segmentation'])
        if mask.ndim==3:mask=np.any(mask,axis=2)
        if mask.shape!=(height,width):raise ValueError('RLE dimensions mismatch')
        raster=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST),dtype=bool)
        overlap=raster&occupied
        out[raster&~occupied]=-1 if ann.get('iscrowd',0)else mapping.get(int(ann['category_id']),-1)
        out[overlap]=-1;occupied|=raster
    return out


def queries(labels,image_id,time):
    pool=np.flatnonzero(labels.ravel()>0)
    if not len(pool):return np.empty((0,3),np.int64),np.empty(0,np.int64)
    indices=np.random.default_rng(SEED+int(image_id)).choice(pool,size=256,replace=len(pool)<256)
    y,x=np.divmod(indices,labels.shape[1]);xy=np.column_stack((x,y,np.full(256,time,dtype=np.int64)))
    return xy,labels.ravel()[indices].astype(np.int64)


def array_hash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed adapter exists')
    prior=Path('semantic_artifacts/external_archive_bounded');old=json.loads((prior/'summary.json').read_text());v=json.loads((prior/'verification.json').read_text())
    assert file_hash(prior/'summary.json')==v['summary_sha256']
    assert file_hash(ANN)=='4ae0f79cd7cde4d0f8aeb7f5c445175e2980691555e6fece36a21c1a77c65203'
    paths=[ANN,OVIS,prior/'summary.json',prior/'verification.json',Path('semantic/external_coco_adapter.py'),Path('semantic/OVIS_EXTERNAL_COCO_ADAPTER_PROTOCOL.md')]
    for arm,record in old['protocol']['frozen_selection'].items():
        for seed in range(3):
            path=checkpoint(arm,seed);assert file_hash(path)==record['checkpoint_sha256'][str(seed)];paths.append(path)
    protocol={'round':41,'query_seed':SEED,'frozen_selection':old['protocol']['frozen_selection'],
        'model_evaluation_updates':0,'input_sha256':{str(p):file_hash(p)for p in paths}}
    atomic_json(ROOT/'protocol.json',protocol)
    payload=json.loads(ANN.read_bytes());ovis=json.loads(OVIS.read_bytes());mapping=category_mapping(payload['categories'],ovis['categories'])
    clips,excluded=groups(payload['images']);atomic_json(ROOT/'groups.json',{'groups':clips,'excluded':excluded})
    audit_indices=np.rint(np.linspace(0,len(clips)-1,24)).astype(int).tolist();audit_clips=[clips[i]for i in audit_indices]
    by_image=defaultdict(list)
    for ann in payload['annotations']:by_image[ann['image_id']].append(ann)
    checks=[];all_hist=Counter();empty=0
    for clip in audit_clips:
        for time in (0,7):
            image=clip['frames'][time];labels=label_map(by_image[image['id']],mapping,image['height'],image['width'])
            assert np.array_equal(labels,label_map(list(reversed(by_image[image['id']])),mapping,image['height'],image['width']))
            xy,target=queries(labels,image['id'],time);counts=dict(sorted(Counter(map(int,labels.ravel())).items()))
            all_hist.update(map(int,target));empty+=int(not len(target))
            checks.append({'group':clip['group'],'image_id':image['id'],'time':time,'file_name':image['file_name'],
                'mask_sha256':array_hash(labels),'xy_sha256':array_hash(xy),'labels_sha256':array_hash(target),
                'pixel_counts':counts,'query_count':len(target),'distinct_query_pixels':len(np.unique(xy[:,:2],axis=0))})
    atomic_json(ROOT/'raster_checks.json',checks)
    external_paths={im['file_name']for im in payload['images']};ovis_paths={f for video in ovis['videos']for f in video['file_names']}
    for path,h in protocol['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'image_count':len(payload['images']),'annotation_count':len(payload['annotations']),
        'directory_group_count':len(clips)+len(excluded),'eligible_group_count':len(clips),'exclusions':excluded,
        'group_frame_count_histogram':dict(sorted(Counter(c['frame_count']for c in clips).items())),
        'shared_category_mapping':{str(k):v for k,v in mapping.items()},'groups_sha256':file_hash(ROOT/'groups.json'),
        'audit_groups':[c['group']for c in audit_clips],'raster_checks_sha256':file_hash(ROOT/'raster_checks.json'),
        'audited_frames':len(checks),'empty_known_foreground_frames':empty,'query_count':sum(c['query_count']for c in checks),
        'query_label_histogram':dict(sorted(all_hist.items())),'OVIS_image_path_intersection':len(external_paths&ovis_paths),
        'duplicate_screening_complete':False,'image_acquisition_ready':False,'formal_confirmation_frozen':False,
        'no_model_evaluation':True,'deployment_unchanged':'native_gt'})
    print('DONE COCO adapter',len(clips),'eligible',len(checks),'frames',empty,'empty',flush=True)


if __name__=='__main__':main()
