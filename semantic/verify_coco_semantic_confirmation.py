"""Independent COCO polygon/RLE, static features, 42 logits and ten contrasts."""
from collections import defaultdict
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco
import torch
from torch.nn import functional as F
from semantic import coco_semantic_confirmation as work
from semantic.verify_external_semantic_confirmation import metric,equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def independent_feature(model,path,xy,dimension):
 with Image.open(path)as image:array=np.asarray(image.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32)
 video=torch.stack([torch.from_numpy(array).permute(2,0,1)/255]).cuda();mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 with torch.inference_mode():
  image=(F.interpolate(video,size=(448,448),mode='bicubic',align_corners=False)-mean)/std
  with torch.autocast('cuda',dtype=torch.bfloat16):tokens=model.forward_features(image)['x_norm_patchtokens'].reshape(1,32,32,dimension).permute(0,3,1,2)
  coordinates=torch.from_numpy(xy[:,:2]).float().cuda();grid=(((coordinates+.5)/128)*2-1)[None,None];values=F.grid_sample(tokens.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
 return values.half().cpu().numpy()

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==68 and p['image_batch']==1
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 manifest=json.loads((ROOT/'manifest.json').read_text());images=manifest['candidate'];eligible=json.loads((work.PANEL/'frozen_eligibility.json').read_text());assert [im['id']for im in images]==eligible['candidate']and manifest['reserved_image_ids']==eligible['reserve']and not set(eligible['candidate'])&set(eligible['reserve'])
 allrecords={r['image_id']:r for r in json.loads((work.PANEL/'image_records.json').read_text())};payload=json.loads((work.cohort.META/'instances_val2017.json').read_bytes());mapping={int(k):int(v)for k,v in json.loads((work.cohort.META/'summary.json').read_text())['shared_name_mapping'].items()};by=defaultdict(list)
 for ann in payload['annotations']:by[ann['image_id']].append(ann)
 coverage=[];packages={policy:{bone:[]for bone in ('S14','B14')}for policy in work.POLICIES}
 for im in images:
  count=np.zeros((128,128),np.int32);owner=np.zeros_like(count);crowd=np.zeros((128,128),bool);signature=np.zeros((128,128),np.uint32)
  for ann in by[im['id']]:
   seg=ann['segmentation']
   if isinstance(seg,list):rle=coco.merge(coco.frPyObjects(seg,im['height'],im['width']))
   elif isinstance(seg['counts'],list):rle=coco.frPyObjects(seg,im['height'],im['width'])
   else:rle=seg
   native=coco.decode(rle);assert native.shape==(im['height'],im['width']);raster=np.array(Image.fromarray(native.astype('uint8')).resize((128,128),Image.Resampling.NEAREST),bool);count+=raster;cid=mapping.get(ann['category_id']);is_crowd=bool(ann['iscrowd']);owner[raster]=-1 if cid is None or is_crowd else cid
   if is_crowd:crowd|=raster
   elif cid is not None:signature[raster]|=np.uint32(1<<cid)
  strict=np.where(count==0,0,np.where(count==1,owner,-1)).astype(np.int16);consensus=np.full((128,128),-1,np.int16);consensus[count==0]=0
  for cid in range(1,26):consensus[(signature==np.uint32(1<<cid))&~crowd]=cid
  assert np.array_equal(strict[strict>0],consensus[strict>0])
  for policy,labels in [('strict',strict),('consensus',consensus)]:
   points=[];targets=[]
   for cid in range(2,26):
    pool=np.flatnonzero(labels.ravel()==cid)
    if not len(pool):continue
    chosen=np.random.default_rng(20261052+im['id']*32+cid).choice(pool,64,replace=len(pool)<64);points.append(np.column_stack((chosen%128,chosen//128,np.zeros(64))).astype(np.int16));targets.append(np.full(64,cid,np.int16))
   xy=np.concatenate(points)if points else np.empty((0,3),np.int16);target=np.concatenate(targets)if targets else np.empty(0,np.int16)
   with np.load(work.qp(policy,im['id']))as a:assert np.array_equal(xy,a['xy'])and np.array_equal(target,a['labels'])and int(a['image_id'])==im['id']
   coverage.append({'policy':policy,'image_id':im['id'],'nonperson_pixels':int((labels>1).sum()),'ignored_pixels':int((labels<0).sum()),'query_count':len(target),'distinct_query_pixels':len(np.unique(xy[:,:2],axis=0)),'query_class_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(target,return_counts=True))}})
   for bone,dimension in [('S14',384),('B14',768)]:
    with np.load(work.fp(bone,policy,im['id']))as a:record={k:a[k].copy()for k in a.files}
    assert np.array_equal(record['xy'],xy)and np.array_equal(record['labels'],target)and int(record['image_id'])==im['id']and record['decoded'].shape==(len(target),dimension)and record['decoded'].dtype==np.float16 and np.isfinite(record['decoded']).all();packages[policy][bone].append(record)
  print('verified static COCO GT',im['id'],flush=True)
 assert coverage==json.loads((ROOT/'query_coverage.json').read_text())and file_hash(ROOT/'query_coverage.json')==s['query_coverage_sha256']
 for key,folder in [('query_hashes','queries'),('feature_hashes','features')]:
  assert set(s[key])=={str(path)for path in (ROOT/folder).rglob('*.npz')}
  for path,h in s[key].items():assert file_hash(path)==h
 feature_replay={}
 for bone,dimension in [('S14',384),('B14',768)]:
  first={policy:next(i for i,a in enumerate(packages[policy][bone])if len(a['labels']))for policy in work.POLICIES};model=work.rgb.prior.resolution.load_dino()if bone=='S14'else work.load_b14()
  for index in sorted(set(first.values())):
   xy=np.concatenate([packages[a][bone][index]['xy']for a in work.POLICIES]);values=independent_feature(model,allrecords[images[index]['id']]['path'],xy,dimension);offset=0
   for policy in work.POLICIES:
    expected=packages[policy][bone][index]['decoded'];assert np.array_equal(values[offset:offset+len(expected)],expected);offset+=len(expected)
  feature_replay[bone]={policy:images[index]['id']for policy,index in first.items()};del model;torch.cuda.empty_cache()
 contrasts_verified=0
 for policy in work.POLICIES:
  result=s['policies'][policy];base=packages[policy]['S14'];target=np.concatenate([a['labels']for a in base]);ids=np.concatenate([np.full(len(a['labels']),int(a['image_id']),np.int32)for a in base]);rows=[r for r in coverage if r['policy']==policy]
  assert {str(im['id']):len(a['labels'])for im,a in zip(images,base)}==result['query_counts']and [str(im['id'])for im,a in zip(images,base)if not len(a['labels'])]==result['empty_images']and len(target)==result['total_queries']and sum(r['distinct_query_pixels']for r in rows)==result['distinct_image_query_pixels']
  assert {str(int(c)):int(n)for c,n in zip(*np.unique(target,return_counts=True))}==result['actual_support'];rebuilt={}
  for arm in work.ARMS:
   data=packages[policy][arm.split('_')[0]];offset=p['models'][arm]['offset'];assert result['results'][arm]['offset']==offset;rebuilt[arm]=[]
   for seed in range(3):
    path=work.cohort.ckpt(arm,seed);assert file_hash(path)==p['models'][arm]['checkpoint_sha256'][str(seed)];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in data])
    with np.load(ROOT/policy/arm/'logits'/f'seed{seed}.npz')as a:assert np.array_equal(values,a['logits'])and np.array_equal(target,a['labels'])and np.array_equal(ids,a['image_ids'])and str(a['checkpoint_sha256'])==file_hash(path)
    shifted=values.copy();shifted[:,0]+=offset;pred=shifted.argmax(1);per={}
    for im in images:
     ix=ids==im['id'];per[str(im['id'])]=metric(pred[ix],target[ix])[0]if ix.any()else None
    row={'seed':seed,'raw':metric(values.argmax(1),target)[0],'calibrated':metric(pred,target)[0],'per_image':per};equal(row,result['results'][arm]['runs'][seed]);rebuilt[arm].append(row);print('verified static COCO head',policy,arm,seed,flush=True)
  def check(c):
   left,right=c['left'],c['right'];per={}
   for im in images:
    key=str(im['id']);per[key]=float(np.mean([rebuilt[left][seed]['per_image'][key]['macro_f1_present']-rebuilt[right][seed]['per_image'][key]['macro_f1_present']for seed in range(3)]))if rebuilt[left][0]['per_image'][key]is not None else None
   equal(per,c['per_image_three_seed_mean_differences']);d=np.array([x for x in per.values()if x is not None]);draw=np.random.default_rng(20261066).integers(0,len(d),(2000,len(d)));b=c['conditional_image_bootstrap'];equal(float(d.mean()),b['mean_per_image_difference']);equal(np.quantile(d[draw].mean(1),[.025,.975]).tolist(),b['percentile_95_interval'])
   assert (int((d>0).sum()),int((d<0).sum()),int((d==0).sum()),len(d))==(b['positive_images'],b['negative_images'],b['tied_images'],b['image_count'])
   pooled=[rebuilt[left][seed]['calibrated']['macro_f1_present']-rebuilt[right][seed]['calibrated']['macro_f1_present']for seed in range(3)];equal(pooled,c['per_seed_pooled_macro_difference']);equal(float(np.mean(pooled)),c['mean_pooled_macro_difference'])
  check(result['primary_contrast']);contrasts_verified+=1
  for c in result['secondary_contrasts_unadjusted'].values():check(c);contrasts_verified+=1
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert not set(manifest['reserved_image_ids'])&{int(path.stem)for path in (ROOT/'features').rglob('*.npz')}and s['reserved80_not_evaluated']and s['new_training_or_calibration']==0 and s['deployment_unchanged']=='native_gt'
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'images_two_GT_policies_replayed':2*len(images),'first_static_feature_replays':feature_replay,'all42_logits_and_metrics_recomputed':True,'paired_image_bootstraps_recomputed':contrasts_verified,'all_direct_backbone_head_dependency_and_input_fingerprints_unchanged':True,'COCO_reserve80_and_LVVIS40_not_evaluated':True,'no_training_or_calibration':True});print('DONE independent COCO static semantic verification',flush=True)
if __name__=='__main__':main()
