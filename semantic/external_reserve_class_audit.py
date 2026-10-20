"""Post-score descriptive class coverage and confusion; never tune models."""
from collections import defaultdict,Counter
import csv,fcntl,json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as masks
from semantic import external_reserve_confirmation as prior
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/external_reserve_class_audit')

def class_stats(matrix):
 out={}
 for c in range(26):
  tp=float(matrix[c,c]);support=float(matrix[c].sum());predicted=float(matrix[:,c].sum());fp=predicted-tp;fn=support-tp
  out[str(c)]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/predicted if predicted else None,'recall':tp/support if support else None,'f1':2*tp/(support+predicted)if support else None}
 return out

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed descriptive audit: do not repeat')
  s=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text());assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256'];clips=json.loads((prior.ROOT/'manifest.json').read_text())['confirmation'];names={0:'Background',**{r['id']:r['name']for r in json.loads(prior.old.OVIS.read_text())['categories']}}
  paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'manifest.json',prior.ANN,prior.old.OVIS,prior.old.MAPPING_SUMMARY,Path('semantic/external_reserve_class_audit.py'),Path('semantic/OVIS_EXTERNAL_RESERVE_CLASS_AUDIT_PROTOCOL.md')]+[Path(x)for x in s['output_hashes']]
  p={'round':94,'scope':'post-score descriptive only','source_summary_sha256':v['summary_sha256'],'policies':list(prior.POLICIES),'arms':list(prior.ARMS),'input_sha256':{str(x):file_hash(x)for x in paths}};atomic_json(ROOT/'protocol.json',p)
  data=json.loads(prior.ANN.read_bytes());mapping={int(k):int(x)for k,x in json.loads(prior.old.MAPPING_SUMMARY.read_text())['shared_category_mapping'].items()};by=defaultdict(list)
  for a in data['annotations']:by[a['image_id']].append(a)
  annotation_coverage={str(c):{'annotations':0,'raster_union_pixels':0,'raster_groups':set(),'strict_pixels':0,'consensus_pixels':0,'native_nonempty_but_128_empty_annotations':0}for c in range(26)}
  for clip in clips:
   for im in clip['frames']:
    raster_by={}
    for a in by[im['id']]:
     cid=mapping.get(a['category_id'])
     if cid is None or a.get('iscrowd',0):continue
     native=masks.decode(a['segmentation']);raster=np.asarray(Image.fromarray(native.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST),bool);row=annotation_coverage[str(cid)];row['annotations']+=1;row['native_nonempty_but_128_empty_annotations']+=int(native.any()and not raster.any());raster_by[cid]=raster_by.get(cid,np.zeros((128,128),bool))|raster
    maps={'strict':prior.label_map(by[im['id']],mapping,im['height'],im['width']),'consensus':prior.consensus_map(by[im['id']],mapping,im['height'],im['width'])}
    for c,mask in raster_by.items():
     row=annotation_coverage[str(c)];row['raster_union_pixels']+=int(mask.sum())
     if mask.any():row['raster_groups'].add(clip['group'])
     for policy in prior.POLICIES:row[policy+'_pixels']+=int((maps[policy]==c).sum())
  for row in annotation_coverage.values():row['raster_groups']=sorted(row['raster_groups'])
  results={};table=[]
  for policy in prior.POLICIES:
   support={str(c):{'query_count':0,'distinct_pixels':0,'frames':set(),'groups':set()}for c in range(26)}
   for clip in clips:
    with np.load(prior.qp(policy,clip['group']))as a:xy=a['xy'].copy();y=a['labels'].copy()
    for c in np.unique(y):
     keep=y==c;row=support[str(c)];row['query_count']+=int(keep.sum());row['distinct_pixels']+=len(np.unique(xy[keep],axis=0));row['frames'].update((clip['group'],int(t))for t in np.unique(xy[keep,2]));row['groups'].add(clip['group'])
   for row in support.values():
    row['frame_count']=len(row.pop('frames'));row['groups']=sorted(row['groups']);row['group_count']=len(row['groups']);row['status']='not_evaluated'if not row['query_count']else'evaluated';row['single_group']=row['group_count']==1;row['under100_distinct_pixels']=0<row['distinct_pixels']<100;row['under256_queries']=0<row['query_count']<256
   arms={}
   for arm in prior.ARMS:
    modes={}
    for mode,offset in [('raw',0),('calibrated',s['protocol']['models'][arm]['offset'])]:
     stats=[];matrices=[]
     for seed in range(3):
      with np.load(s['results'][policy]['arms'][arm]['runs'][seed]['output_path'])as a:logits=a['logits'].copy();y=a['labels'].copy()
      logits[:,0]+=offset;pred=logits.argmax(1);matrix=np.bincount(y*26+pred,minlength=676).reshape(26,26);matrices.append(matrix);stats.append(class_stats(matrix))
     mean_matrix=np.mean(matrices,axis=0);classes={}
     for c in range(26):
      key=str(c);mean={k:float(np.mean([r[key][k]for r in stats]))if all(r[key][k]is not None for r in stats)else None for k in stats[0][key]};wrong=[(int(j),float(mean_matrix[c,j]))for j in range(26)if j!=c and mean_matrix[c,j]>0];wrong.sort(key=lambda x:(-x[1],x[0]));top=[{'predicted_class':j,'name':names[j],'mean_count':n,'fraction_of_true_class':n/support[key]['query_count']}for j,n in wrong[:3]]
      classes[key]={'name':names[c],**support[key],'per_seed':[r[key]for r in stats],'three_seed_mean':mean,'top_confusions':top};table.append({'policy':policy,'arm':arm,'mode':mode,'class_id':c,'class_name':names[c],**{k:support[key][k]for k in ('status','query_count','distinct_pixels','frame_count','group_count')},**mean,'top_confusions':json.dumps(top,ensure_ascii=False)})
     modes[mode]={'per_class':classes,'three_seed_mean_confusion':mean_matrix.tolist()}
    arms[arm]=modes
   results[policy]={'coverage':support,'arms':arms}
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'class_names':names,'annotation_coverage':annotation_coverage,'results':results,'new_training_calibration_or_model_evaluation':False})
  with(ROOT/'per_class.csv').open('w',newline='',encoding='utf-8-sig')as f:
   writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
  print('DONE descriptive class audit',len(table),'rows',flush=True)
if __name__=='__main__':main()
