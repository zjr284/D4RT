"""Independent per-pixel set-label and frozen head sensitivity replay."""
from collections import defaultdict
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco
import torch
from semantic.external_label_sensitivity import ROOT,OLD,PANEL,ANN,ARMS,MAPPING
from semantic import dino_native_rgb_control as rgb
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.verify_external_semantic_confirmation import metric,equal,linear,independent_features
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 clips=json.loads((ROOT/'manifest.json').read_text())['confirmation'];assert clips==json.loads((OLD/'manifest.json').read_text())['confirmation']
 payload=json.loads(ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(MAPPING.read_text())['shared_category_mapping'].items()};by=defaultdict(list)
 for a in payload['annotations']:by[a['image_id']].append(a)
 packages=[];coverage=[];allold=[]
 for clip in clips:
  xx=[];yy=[]
  for slot,im in enumerate(clip['frames']):
   masks=[];codes=[];crowds=[]
   for a in by[im['id']]:
    m=coco.decode(a['segmentation']);assert m.shape==(im['height'],im['width'])
    masks.append(np.array(Image.fromarray(m.astype('uint8')).resize((128,128),Image.Resampling.NEAREST),bool));codes.append(mapping.get(a['category_id']));crowds.append(bool(a.get('iscrowd',0)))
   labels=np.zeros(128*128,np.int16);old=np.zeros_like(labels)
   # Explicit set construction per pixel differs from production's class planes.
   for index in range(len(labels)):
    hits=[i for i,m in enumerate(masks)if m.ravel()[index]]
    if not hits:continue
    old[index]=codes[hits[0]]if len(hits)==1 and codes[hits[0]]is not None and not crowds[hits[0]]else -1
    known={codes[i]for i in hits if codes[i]is not None and not crowds[i]}
    labels[index]=next(iter(known))if len(known)==1 and not any(crowds[i]for i in hits)else -1
   assert np.array_equal(labels[old>0],old[old>0]);xy=[];target=[]
   for cid in range(2,26):
    pool=np.flatnonzero(labels==cid)
    if not len(pool):continue
    chosen=np.random.default_rng(20261052+im['id']*32+cid).choice(pool,64,replace=len(pool)<64)
    xy.append(np.column_stack((chosen%128,chosen//128,np.full(64,slot))).astype(np.int16));target.append(np.full(64,cid,np.int16))
   xy=np.concatenate(xy)if xy else np.empty((0,3),np.int16);y=np.concatenate(target)if target else np.empty(0,np.int16);xx.append(xy);yy.append(y)
   coverage.append({'group':clip['group'],'time':slot,'image_id':im['id'],'old_nonperson_pixels':int((old>1).sum()),'new_nonperson_pixels':int((labels>1).sum()),'query_count':len(y),'distinct_query_pixels':len(np.unique(xy[:,:2],axis=0)),'query_class_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(y,return_counts=True))}})
  with np.load(ROOT/'features'/f'{clip["group"]}.npz')as a:package={k:a[k].copy()for k in a.files}
  assert np.array_equal(package['xy'],np.concatenate(xx))and np.array_equal(package['labels'],np.concatenate(yy))and int(package['video_id'])==int(clip['group'])
  assert package['decoded'].shape==(len(package['labels']),384)and package['decoded'].dtype==np.float16 and np.isfinite(package['decoded']).all()
  assert file_hash(ROOT/'features'/f'{clip["group"]}.npz')==s['feature_hashes'][str(ROOT/'features'/f'{clip["group"]}.npz')];packages.append(package)
  with np.load(OLD/'features'/f'{clip["group"]}.npz')as a:allold.append({k:a[k].copy()for k in a.files})
  print('verified sensitivity GT',clip['group'],flush=True)
 assert coverage==json.loads((ROOT/'query_coverage.json').read_text())and file_hash(ROOT/'query_coverage.json')==s['query_coverage_sha256']
 assert sum(not r['query_count']for r in coverage)==s['empty_frames']and sum(r['distinct_query_pixels']for r in coverage)==s['distinct_frame_query_pixels']
 model=rgb.prior.resolution.load_dino();first=next(i for i,a in enumerate(packages)if len(a['labels']));replayed=0
 for i,(clip,a,old)in enumerate(zip(clips,packages,allold)):
  if not len(old['xy'])and i!=first:continue
  points=np.concatenate([old['xy'],a['xy']if i==first else np.empty((0,3),np.int16)])
  feats=independent_features(model,[str(PANEL/'images'/im['file_name'])for im in clip['frames']],points)
  assert np.array_equal(feats[:len(old['xy'])],old['decoded']);replayed+=int(bool(len(old['xy'])))
  if i==first:assert np.array_equal(feats[len(old['xy']):],a['decoded'])
 del model
 expectedprefix={c['group']:{'old_query_count':len(a['labels']),'bitwise_equal':True}for c,a in zip(clips,allold)}
 assert expectedprefix==s['old_prefix_checks']==json.loads((ROOT/'old_prefix_checks.json').read_text())
 labels=np.concatenate([a['labels']for a in packages]);times=np.concatenate([a['xy'][:,2]for a in packages]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in packages])
 assert len(labels)==s['total_queries']and {str(int(c)):int(n)for c,n in zip(*np.unique(labels,return_counts=True))}==s['actual_support']
 assert {c['group']:len(a['labels'])for c,a in zip(clips,packages)}==s['query_counts']and [c['group']for c,a in zip(clips,packages)if not len(a['labels'])]==s['empty_groups']
 rebuilt={}
 for arm in ARMS:
  rebuilt[arm]=[];offset=p['selection'][arm]['offset'];assert offset==s['results'][arm]['offset']
  for seed in range(3):
   state=torch.load(checkpoint(arm,seed),map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in packages])
   with np.load(ROOT/arm/'logits'/f'dinov2_{seed}_sensitivity.npz')as a:
    assert np.array_equal(a['logits'],values)and np.array_equal(a['labels'],labels)and np.array_equal(a['times'],times)and np.array_equal(a['video_ids'],ids)
    assert str(a['checkpoint_sha256'])==p['selection'][arm]['checkpoint_sha256'][str(seed)]
   shifted=values.copy();shifted[:,0]+=offset;pred=shifted.argmax(1);per={}
   for clip in clips:
    rows=ids==int(clip['group']);per[clip['group']]=metric(pred[rows],labels[rows])[0]if rows.any()else None
   row={'seed':seed,'raw':metric(values.argmax(1),labels)[0],'calibrated':metric(pred,labels)[0],'per_video':per};equal(row,s['results'][arm]['runs'][seed]);rebuilt[arm].append(row)
   print('verified sensitivity head',arm,seed,flush=True)
 def check(c):
  left,right=c['left'],c['right'];d={}
  for clip in clips:
   g=clip['group'];d[g]=float(np.mean([rebuilt[left][i]['per_video'][g]['macro_f1_present']-rebuilt[right][i]['per_video'][g]['macro_f1_present']for i in range(3)]))if rebuilt[left][0]['per_video'][g]is not None else None
  equal(d,c['per_video_three_seed_mean_differences']);x=np.array([v for v in d.values()if v is not None]);draw=np.random.default_rng(20261055).integers(0,len(x),(2000,len(x)));b=c['conditional_video_bootstrap']
  equal(float(x.mean()),b['mean_per_video_difference']);equal(np.quantile(x[draw].mean(1),[.025,.975]).tolist(),b['percentile_95_interval'])
  assert (int((x>0).sum()),int((x<0).sum()),int((x==0).sum()),len(x))==(b['positive_videos'],b['negative_videos'],b['tied_videos'],b['video_count'])
  pooled=[rebuilt[left][i]['calibrated']['macro_f1_present']-rebuilt[right][i]['calibrated']['macro_f1_present']for i in range(3)];equal(pooled,c['per_seed_pooled_macro_difference']);equal(float(np.mean(pooled)),c['mean_pooled_macro_difference'])
 check(s['primary_contrast'])
 for c in s['secondary_contrasts_unadjusted'].values():check(c)
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'GT_frames_replayed':80,'old_prefix_feature_groups_replayed':replayed,'first_new_feature_group_replayed':clips[first]['group'],'all_twelve_logits_metrics_and_three_bootstraps_replayed':True,'old_files_unchanged':True,'new_confirmation_groups_consumed':0})
 print('DONE independent label sensitivity verification',flush=True)
if __name__=='__main__':main()
