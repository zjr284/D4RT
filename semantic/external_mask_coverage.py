"""No-model descriptive audit of already scored external annotation coverage."""
from collections import defaultdict
import fcntl,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from semantic.external_coco_adapter import ANN,label_map
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/external_mask_coverage')
PRIOR=Path('semantic_artifacts/external_nonperson_confirmation')
MAPPING=Path('semantic_artifacts/external_coco_adapter/summary.json')
FIELDS=('native_union_pixels','candidate_128_pixels','kept_unique_pixels','unknown_or_crowd_overlap_pixels','different_known_overlap_pixels','same_class_only_overlap_pixels','instances','native_nonempty_instances','native_nonempty_raster_empty_instances')
def digest(mask):return hashlib.sha256(np.asarray(mask.shape,dtype='<i8').tobytes()+np.ascontiguousarray(mask,dtype=np.bool_).tobytes()).hexdigest()
def audit_frame(annotations,mapping,categories,height,width,size=128):
    rows=[];rasters=[];native_by_class={};dups=defaultdict(list)
    cover=np.zeros((size,size),np.int32);bad=np.zeros((size,size),bool);known=defaultdict(lambda:np.zeros((size,size),bool))
    for ann in annotations:
        native=coco_mask.decode(ann['segmentation']).astype(bool);assert native.shape==(height,width)
        raster=np.asarray(Image.fromarray(native.astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST),dtype=bool)
        cid=mapping.get(int(ann['category_id']));crowd=bool(ann.get('iscrowd',0));h=digest(native)
        row={'annotation_id':ann['id'],'category_id':ann['category_id'],'category_name':categories[ann['category_id']],'mapped_class':cid,'crowd':crowd,'metadata_area':ann.get('area'),'native_pixels':int(native.sum()),'raster_pixels':int(raster.sum()),'native_geometry_sha256':h,'raster_sha256':digest(raster)}
        rows.append(row);rasters.append(raster);dups[h].append(row);cover+=raster
        if cid is None or crowd:bad|=raster
        else:
            known[cid]|=raster
            if cid>1:
                if cid not in native_by_class:native_by_class[cid]=native.copy()
                else:native_by_class[cid]|=native
    classes=[]
    for cid in sorted(c for c in known if c>1):
        candidate=known[cid];other=np.zeros_like(candidate)
        for c,m in known.items():
            if c!=cid:other|=m
        kept=candidate&(cover==1);unknown=candidate&bad;different=candidate&~bad&other;same=candidate&~bad&~other&(cover>1)
        assert np.array_equal(candidate,kept|unknown|different|same)
        inst=[r for r in rows if r['mapped_class']==cid and not r['crowd']]
        classes.append(dict(class_id=cid,native_union_pixels=int(native_by_class[cid].sum()),candidate_128_pixels=int(candidate.sum()),kept_unique_pixels=int(kept.sum()),unknown_or_crowd_overlap_pixels=int(unknown.sum()),different_known_overlap_pixels=int(different.sum()),same_class_only_overlap_pixels=int(same.sum()),instances=len(inst),native_nonempty_instances=sum(r['native_pixels']>0 for r in inst),native_nonempty_raster_empty_instances=sum(r['native_pixels']>0 and not r['raster_pixels'] for r in inst)))
    out=np.zeros((size,size),np.int16)
    for r,m in zip(rows,rasters):out[m]=-1 if r['crowd']or r['mapped_class']is None else r['mapped_class']
    out[cover>1]=-1;assert np.array_equal(out,label_map(annotations,mapping,height,width,size))
    return {'annotations':rows,'classes':classes,'exact_native_geometry_duplicates':[{'native_geometry_sha256':h,'annotations':r}for h,r in sorted(dups.items())if len(r)>1],'label_sha256':digest(out),'nonperson_pixels':int((out>1).sum()),'ignored_pixels':int((out<0).sum())}
def aggregate(frames):
    classes={};groups={};repeated=[]
    for f in frames:
        group=f['group'];g=groups.setdefault(group,{'frames':0,'frames_with_nonperson':0,'nonperson_pixels':0,'class_ids':set()})
        g['frames']+=1;g['frames_with_nonperson']+=int(f['nonperson_pixels']>0);g['nonperson_pixels']+=f['nonperson_pixels']
        for r in f['classes']:
            cid=r['class_id'];d=classes.setdefault(str(cid),dict.fromkeys(FIELDS,0)|dict(frames_with_metadata_instance=0,frames_native_nonempty=0,frames_candidate_nonempty=0,frames_kept_nonempty=0))
            for key in FIELDS:d[key]+=r[key]
            d['frames_with_metadata_instance']+=1;d['frames_native_nonempty']+=int(r['native_union_pixels']>0);d['frames_candidate_nonempty']+=int(r['candidate_128_pixels']>0);d['frames_kept_nonempty']+=int(r['kept_unique_pixels']>0)
            if r['kept_unique_pixels']:g['class_ids'].add(cid)
        for dup in f['exact_native_geometry_duplicates']:
            repeated.append({'group':group,'image_id':f['image_id'],'geometry_sha256':dup['native_geometry_sha256'],'annotation_count':len(dup['annotations']),'category_ids':sorted({a['category_id']for a in dup['annotations']}),'mapped_classes':sorted({a['mapped_class']for a in dup['annotations']if a['mapped_class']is not None}),'has_unknown_or_crowd':any(a['mapped_class']is None or a['crowd']for a in dup['annotations'])})
    for g in groups.values():g['class_ids']=sorted(g['class_ids'])
    return {'by_class':dict(sorted(classes.items(),key=lambda kv:int(kv[0]))),'by_group':groups,'native_geometry_duplicate_groups':repeated,'duplicate_geometry_group_count':len(repeated),'cross_category_duplicate_group_count':sum(len(r['category_ids'])>1 for r in repeated),'same_category_duplicate_group_count':sum(len(r['category_ids'])==1 for r in repeated),'annotated_instances':sum(len(f['annotations'])for f in frames),'frames':len(frames)}
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'summary.json').exists():raise RuntimeError('Completed coverage audit exists')
        s=json.loads((PRIOR/'summary.json').read_text());v=json.loads((PRIOR/'verification.json').read_text());assert file_hash(PRIOR/'summary.json')==v['summary_sha256']
        inputs=dict(s['protocol']['input_sha256']);inputs.update(s['feature_hashes'])
        paths=[PRIOR/'summary.json',PRIOR/'verification.json',PRIOR/'manifest.json',PRIOR/'query_coverage.json',MAPPING,Path('semantic/external_mask_coverage.py'),Path('semantic/OVIS_EXTERNAL_MASK_COVERAGE_PROTOCOL.md')]+sorted(PRIOR.glob('*/logits/*.npz'))
        inputs.update({str(p):file_hash(p)for p in paths})
        for path,h in inputs.items():assert file_hash(path)==h
        p={'round':54,'scope':'post-score descriptive GT coverage audit of existing ten groups','training_updates':0,'new_features_or_predictions':0,'label_or_score_changes':0,'loss_partition_priority':['unique_instance','unknown_or_crowd','different_known','same_class_only'],'input_sha256':inputs};atomic_json(ROOT/'protocol.json',p)
        payload=json.loads(ANN.read_bytes());cats={c['id']:c['name']for c in payload['categories']};mapping={int(k):int(v)for k,v in json.loads(MAPPING.read_text())['shared_category_mapping'].items()}
        by_image=defaultdict(list)
        for a in payload['annotations']:by_image[a['image_id']].append(a)
        clips=json.loads((PRIOR/'manifest.json').read_text())['confirmation'];coverage={(r['group'],r['time']):r for r in json.loads((PRIOR/'query_coverage.json').read_text())};frames=[]
        for clip in clips:
            for slot,im in enumerate(clip['frames']):
                row=audit_frame(by_image[im['id']],mapping,cats,im['height'],im['width']);row.update(group=clip['group'],time=slot,image_id=im['id'],height=im['height'],width=im['width'])
                old=coverage[(clip['group'],slot)];assert row['nonperson_pixels']==old['nonperson_pixels']and row['ignored_pixels']==old['ignored_pixels']
                assert {str(r['class_id'])for r in row['classes']if r['kept_unique_pixels']}==set(old['query_class_counts']);frames.append(row)
            print('coverage audit',clip['group'],flush=True)
        atomic_json(ROOT/'frame_ledger.json',frames);total=aggregate(frames)
        assert {int(c)for c,r in total['by_class'].items()if r['frames_kept_nonempty']}==set(map(int,s['actual_support']))
        for path,h in inputs.items():assert file_hash(path)==h
        atomic_json(ROOT/'summary.json',{'protocol':p,**total,'frame_ledger_sha256':file_hash(ROOT/'frame_ledger.json'),'prior_scores_unchanged':True,'unused_reserve_groups_processed':0,'deployment_unchanged':'native_gt'})
        print('DONE coverage audit',json.dumps(total['by_class']),flush=True)
if __name__=='__main__':main()
