"""Independent Boolean TP/FP/FN, distinct-point sets and bitset pixel coverage."""
from collections import defaultdict
import csv,json
import numpy as np
from PIL import Image
from pycocotools import mask as masks
from semantic import external_reserve_class_audit as work
from semantic.verify_external_semantic_confirmation import equal
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT;prior=work.prior

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==94
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 old=json.loads((prior.ROOT/'summary.json').read_text());clips=json.loads((prior.ROOT/'manifest.json').read_text())['confirmation'];data=json.loads(prior.ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(prior.old.MAPPING_SUMMARY.read_text())['shared_category_mapping'].items()};by=defaultdict(list)
 for a in data['annotations']:by[a['image_id']].append(a)
 annotation={str(c):{'annotations':0,'raster_union_pixels':0,'raster_groups':set(),'strict_pixels':0,'consensus_pixels':0,'native_nonempty_but_128_empty_annotations':0}for c in range(26)}
 for clip in clips:
  for im in clip['frames']:
   count=np.zeros((128,128),np.int32);owner=np.zeros((128,128),np.int16);bits=np.zeros((128,128),np.uint32);crowd=np.zeros((128,128),bool)
   for ann in by[im['id']]:
    raw=masks.decode(ann['segmentation']);hit=np.asarray(Image.fromarray(raw.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST),bool);count+=hit;cid=mapping.get(ann['category_id']);is_crowd=bool(ann.get('iscrowd',0));owner[hit]=-1 if cid is None or is_crowd else cid
    if is_crowd:crowd|=hit
    elif cid is not None:
     bits[hit]|=np.uint32(1<<cid);r=annotation[str(cid)];r['annotations']+=1;r['native_nonempty_but_128_empty_annotations']+=int(raw.any()and not hit.any())
   for c in range(1,26):
    r=annotation[str(c)];union=(bits&np.uint32(1<<c))!=0;n=int(union.sum());r['raster_union_pixels']+=n
    if n:r['raster_groups'].add(clip['group'])
    r['strict_pixels']+=int(((count==1)&(owner==c)).sum());r['consensus_pixels']+=int(((bits==np.uint32(1<<c))&~crowd).sum())
 for r in annotation.values():r['raster_groups']=sorted(r['raster_groups'])
 equal(annotation,s['annotation_coverage']);checks=0
 for policy in prior.POLICIES:
  querycount=np.zeros(26,np.int64);unique={c:set()for c in range(26)};frames={c:set()for c in range(26)};groups={c:set()for c in range(26)};ys=[]
  for clip in clips:
   with np.load(prior.qp(policy,clip['group']))as a:y=a['labels'].copy();xy=a['xy'].copy()
   ys.append(y);querycount+=np.bincount(y,minlength=26)
   for c in np.unique(y):
    for x,v,t in xy[y==c]:unique[int(c)].add((clip['group'],int(t),int(x),int(v)));frames[int(c)].add((clip['group'],int(t)))
    groups[int(c)].add(clip['group'])
  y=np.concatenate(ys)
  for c in range(26):
   r=s['results'][policy]['coverage'][str(c)];assert r['query_count']==int(querycount[c])and r['distinct_pixels']==len(unique[c])and r['frame_count']==len(frames[c])and r['groups']==sorted(groups[c])and r['group_count']==len(groups[c]);assert r['status']==('evaluated'if querycount[c]else'not_evaluated')and r['single_group']==(len(groups[c])==1)and r['under100_distinct_pixels']==(0<len(unique[c])<100)and r['under256_queries']==(0<querycount[c]<256)
  for arm in prior.ARMS:
   for mode,offset in [('raw',0),('calibrated',old['protocol']['models'][arm]['offset'])]:
    per={c:[]for c in range(26)};matrices=[]
    for seed in range(3):
     with np.load(old['results'][policy]['arms'][arm]['runs'][seed]['output_path'])as a:logits=a['logits'].copy();assert np.array_equal(a['labels'],y)
     logits[:,0]+=offset;pred=logits.argmax(1);matrix=np.zeros((26,26),np.int64);np.add.at(matrix,(y,pred),1);matrices.append(matrix)
     for c in range(26):
      tp=int(((pred==c)&(y==c)).sum());fp=int(((pred==c)&(y!=c)).sum());fn=int(((pred!=c)&(y==c)).sum());n=tp+fn;pp=tp+fp
      r={'tp':float(tp),'fp':float(fp),'fn':float(fn),'precision':tp/pp if pp else None,'recall':tp/n if n else None,'f1':2*tp/(2*tp+fp+fn)if n else None};per[c].append(r);equal(r,s['results'][policy]['arms'][arm][mode]['per_class'][str(c)]['per_seed'][seed]);checks+=1
    mean=sum(matrices)/3;expected=s['results'][policy]['arms'][arm][mode];equal(mean.tolist(),expected['three_seed_mean_confusion'])
    for c in range(26):
     row=expected['per_class'][str(c)];averages={k:sum(z[k]for z in per[c])/3 if all(z[k]is not None for z in per[c])else None for k in per[c][0]};equal(averages,row['three_seed_mean'])
     for k,v in s['results'][policy]['coverage'][str(c)].items():equal(v,row[k])
     options=[j for j in range(26)if j!=c and mean[c,j]>0];options=sorted(options,key=lambda j:(-mean[c,j],j))[:3];top=row['top_confusions'];assert[j['predicted_class']for j in top]==options
     for item,j in zip(top,options):assert item['name']==s['class_names'][str(j)];assert abs(item['mean_count']-mean[c,j])<1e-12 and abs(item['fraction_of_true_class']-mean[c,j]/querycount[c])<1e-12
 rows=list(csv.DictReader((ROOT/'per_class.csv').open(encoding='utf-8-sig')));assert len(rows)==624
 for row in rows:
  item=s['results'][row['policy']]['arms'][row['arm']][row['mode']]['per_class'][row['class_id']];assert row['status']==item['status']
  for k in ('query_count','distinct_pixels','frame_count','group_count'):assert int(row[k])==item[k]
  for k,v in item['three_seed_mean'].items():assert(row[k]==''if v is None else abs(float(row[k])-v)<1e-12)
  equal(json.loads(row['top_confusions']),item['top_confusions'])
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'CSV_sha256':file_hash(ROOT/'per_class.csv'),'independent_bitset_pixel_coverage_all320_frames':True,'distinct_pixel_frame_group_counts_checked':True,'all1872_per_seed_class_metrics_checked':checks==1872,'all624_CSV_rows_checked':True,'all_confusions_and_absent_class_nulls_checked':True,'no_training_calibration_or_model_changes':True});print('DONE independent descriptive class audit',checks,len(rows),flush=True)
if __name__=='__main__':main()
