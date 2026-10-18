"""Independent mask-stack and direct-equality replay of coverage audit."""
from collections import defaultdict,Counter
import hashlib,json
import numpy as np
from PIL import Image
from pycocotools import mask as coco
from semantic.external_mask_coverage import ROOT,PRIOR,MAPPING,ANN,FIELDS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

def sha(a):return hashlib.sha256(np.array(a.shape,dtype='<i8').tobytes()+a.astype(np.bool_).tobytes(order='C')).hexdigest()
def main():
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 frames=json.loads((ROOT/'frame_ledger.json').read_text());assert file_hash(ROOT/'frame_ledger.json')==s['frame_ledger_sha256']
 clips=json.loads((PRIOR/'manifest.json').read_text())['confirmation'];assert [(f['group'],f['image_id'])for f in frames]==[(c['group'],im['id'])for c in clips for im in c['frames']]
 payload=json.loads(ANN.read_bytes());cats={c['id']:c['name']for c in payload['categories']};mapping={int(k):v for k,v in json.loads(MAPPING.read_text())['shared_category_mapping'].items()}
 by=defaultdict(list)
 for a in payload['annotations']:by[a['image_id']].append(a)
 agg={};group_stats={};repeated=[];allann=0
 coverage={(r['group'],r['time']):r for r in json.loads((PRIOR/'query_coverage.json').read_text())}
 for f in frames:
  anns=by[f['image_id']];native=[coco.decode(a['segmentation']).astype(bool)for a in anns];allann+=len(anns)
  for m in native:assert m.shape==(f['height'],f['width'])
  raster=np.stack([np.array(Image.fromarray(m.astype('uint8')).resize((128,128),Image.Resampling.NEAREST),dtype=bool)for m in native])if anns else np.zeros((0,128,128),bool)
  cover=raster.sum(0);codes=[mapping.get(a['category_id'])if not a.get('iscrowd',0)else None for a in anns]
  owner=np.zeros((128,128),np.int16)
  expected_rows=[]
  for i,(a,m)in enumerate(zip(anns,native)):
   owner[raster[i]]=-1 if codes[i]is None else codes[i]
   expected_rows.append({'annotation_id':a['id'],'category_id':a['category_id'],'category_name':cats[a['category_id']],'mapped_class':mapping.get(a['category_id']),'crowd':bool(a.get('iscrowd',0)),'metadata_area':a.get('area'),'native_pixels':int(np.count_nonzero(m)),'raster_pixels':int(np.count_nonzero(raster[i])),'native_geometry_sha256':sha(m),'raster_sha256':sha(raster[i])})
  assert expected_rows==f['annotations'];owner[cover>1]=-1
  assert sha(owner)==f['label_sha256']and int((owner>1).sum())==f['nonperson_pixels']and int((owner<0).sum())==f['ignored_pixels']
  old=coverage[(f['group'],f['time'])];assert old['nonperson_pixels']==f['nonperson_pixels']and old['ignored_pixels']==f['ignored_pixels']
  classrows=[]
  for cid in sorted(set(c for c in codes if c is not None and c>1)):
   indexes=[i for i,c in enumerate(codes)if c==cid];pool=raster[indexes].any(0);native_union=np.logical_or.reduce([native[i]for i in indexes]);badix=[i for i,c in enumerate(codes)if c is None];otherix=[i for i,c in enumerate(codes)if c is not None and c!=cid]
   bad=raster[badix].any(0)if badix else np.zeros_like(pool);other=raster[otherix].any(0)if otherix else np.zeros_like(pool)
   n=int(pool.sum());kept=int((owner==cid).sum());unknown=int((pool&bad).sum());different=int((pool&~bad&other).sum());same=n-kept-unknown-different;assert same>=0
   r=dict(class_id=cid,native_union_pixels=int(native_union.sum()),candidate_128_pixels=n,kept_unique_pixels=kept,unknown_or_crowd_overlap_pixels=unknown,different_known_overlap_pixels=different,same_class_only_overlap_pixels=same,instances=len(indexes),native_nonempty_instances=sum(bool(native[i].any())for i in indexes),native_nonempty_raster_empty_instances=sum(bool(native[i].any())and not raster[i].any()for i in indexes))
   classrows.append(r);d=agg.setdefault(str(cid),Counter())
   for k in FIELDS:d[k]+=int(r[k])
   for key,value in [('frames_with_metadata_instance',True),('frames_native_nonempty',r['native_union_pixels']),('frames_candidate_nonempty',n),('frames_kept_nonempty',kept)]:d[key]+=int(bool(value))
  assert classrows==f['classes']
  assert {str(r['class_id'])for r in classrows if r['kept_unique_pixels']}==set(old['query_class_counts'])
  # Direct equality rather than production's hash grouping identifies duplicate sets.
  seen=set();dups=[]
  for i in range(len(anns)):
   if i in seen:continue
   indexes=[j for j in range(i,len(anns))if np.array_equal(native[i],native[j])];seen.update(indexes)
   if len(indexes)>1:dups.append({'native_geometry_sha256':sha(native[i]),'annotations':[expected_rows[j]for j in indexes]})
  dups.sort(key=lambda r:r['native_geometry_sha256']);assert dups==f['exact_native_geometry_duplicates']
  for d in dups:
   repeated.append({'group':f['group'],'image_id':f['image_id'],'geometry_sha256':d['native_geometry_sha256'],'annotation_count':len(d['annotations']),'category_ids':sorted({a['category_id']for a in d['annotations']}),'mapped_classes':sorted({a['mapped_class']for a in d['annotations']if a['mapped_class']is not None}),'has_unknown_or_crowd':any(a['mapped_class']is None or a['crowd']for a in d['annotations'])})
  g=group_stats.setdefault(f['group'],dict(frames=0,frames_with_nonperson=0,nonperson_pixels=0,class_ids=set()));g['frames']+=1;g['frames_with_nonperson']+=int(f['nonperson_pixels']>0);g['nonperson_pixels']+=f['nonperson_pixels'];g['class_ids'].update(r['class_id']for r in classrows if r['kept_unique_pixels'])
  print('verified coverage frame',f['group'],f['time'],flush=True)
 for g in group_stats.values():g['class_ids']=sorted(g['class_ids'])
 assert {k:dict(v)for k,v in agg.items()}==s['by_class']and group_stats==s['by_group']and repeated==s['native_geometry_duplicate_groups']
 assert len(repeated)==s['duplicate_geometry_group_count']and sum(len(r['category_ids'])>1 for r in repeated)==s['cross_category_duplicate_group_count']and sum(len(r['category_ids'])==1 for r in repeated)==s['same_category_duplicate_group_count']
 assert allann==s['annotated_instances']and len(frames)==s['frames']==80
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'frames_independently_replayed':80,'annotation_instances_redecoded':allann,'native_duplicate_groups_directly_compared':len(repeated),'old_inputs_unchanged':True,'per_class_pixel_partition_and_aggregates_recomputed':True,'no_new_predictions_or_label_changes':True})
 print('DONE independent coverage verification',flush=True)
if __name__=='__main__':main()
