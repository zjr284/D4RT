"""Independent RLE bit-set labels, queries, features, 24 logits and pair replay."""
from collections import defaultdict
import json,subprocess
import numpy as np
from PIL import Image
from pycocotools import mask as coco
import torch
from semantic import external_capacity_confirmation as work
from semantic.verify_external_semantic_confirmation import metric,equal,linear,independent_features
from semantic.verify_dino_backbone_training import independent_b14
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==60
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 # Also bind S14 and backbone code directly to their earlier frozen fingerprints;
 # the main manifest bound these through recipe/source references, not direct paths.
 old=json.loads(__import__('pathlib').Path('semantic_artifacts/external_nonperson_confirmation/protocol.json').read_text());legacy={path:h for path,h in old['input_sha256'].items()if path.startswith('semantic_artifacts/vendor/dinov2/')or path=='semantic_artifacts/weights/dinov2_vits14_pretrain.pth'}
 for path,h in legacy.items():assert file_hash(path)==h
 vendor=work.rgb.prior.resolution.VENDOR;assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()=='7764ea0f912e53c92e82eb78a2a1631e92725fc8';subprocess.run(['git','diff','--quiet','HEAD'],cwd=vendor,check=True)
 clips=json.loads((ROOT/'manifest.json').read_text())['confirmation'];manifest=json.loads((work.PANEL/'manifest.json').read_text());eligible=json.loads((work.PANEL/'frozen_eligibility.json').read_text());assert [c['group']for c in clips]==eligible['accepted_groups']and not set(eligible['accepted_groups'])&set(manifest['excluded_previous90'])
 assert p['selection']==json.loads((work.SOURCE/'frozen_selection.json').read_text())and p['selection']['validation_selected_arm']=='B14_gt32'
 payload=json.loads(work.ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(work.MAPPING.read_text())['shared_category_mapping'].items()};by=defaultdict(list)
 for a in payload['annotations']:by[a['image_id']].append(a)
 coverage=[];packages={policy:{bone:[]for bone in ('S14','B14')}for policy in work.POLICIES}
 for clip in clips:
  points={a:[]for a in work.POLICIES};targets={a:[]for a in work.POLICIES}
  for slot,im in enumerate(clip['frames']):
   cover=np.zeros((128,128),np.int32);owner=np.zeros_like(cover);crowd=np.zeros((128,128),bool);signature=np.zeros((128,128),np.uint32)
   for ann in by[im['id']]:
    mask=coco.decode(ann['segmentation']);assert mask.shape==(im['height'],im['width']);raster=np.array(Image.fromarray(mask.astype('uint8')).resize((128,128),Image.Resampling.NEAREST),bool)
    cover+=raster;cid=mapping.get(ann['category_id']);is_crowd=bool(ann.get('iscrowd',0));owner[raster]=-1 if cid is None or is_crowd else cid
    if is_crowd:crowd|=raster
    elif cid is not None:signature[raster]|=np.uint32(1<<cid)
   strict=np.where(cover==0,0,np.where(cover==1,owner,-1)).astype(np.int16);consensus=np.full((128,128),-1,np.int16);consensus[cover==0]=0
   for cid in range(1,26):consensus[(signature==np.uint32(1<<cid))&~crowd]=cid
   assert np.array_equal(strict[strict>0],consensus[strict>0])
   for policy,labels in [('strict',strict),('consensus',consensus)]:
    xx=[];yy=[]
    for cid in range(2,26):
     pool=np.flatnonzero(labels.ravel()==cid)
     if not len(pool):continue
     chosen=np.random.default_rng(20261052+im['id']*32+cid).choice(pool,64,replace=len(pool)<64)
     xx.append(np.column_stack((chosen%128,chosen//128,np.full(64,slot))).astype(np.int16));yy.append(np.full(64,cid,np.int16))
    xy=np.concatenate(xx)if xx else np.empty((0,3),np.int16);target=np.concatenate(yy)if yy else np.empty(0,np.int16);points[policy].append(xy);targets[policy].append(target)
    coverage.append({'policy':policy,'group':clip['group'],'time':slot,'image_id':im['id'],'nonperson_pixels':int((labels>1).sum()),'ignored_pixels':int((labels<0).sum()),'query_count':len(target),'distinct_query_pixels':len(np.unique(xy[:,:2],axis=0)),'query_class_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(target,return_counts=True))}})
  for policy in work.POLICIES:
   with np.load(work.qp(policy,clip['group']))as a:meta={k:a[k].copy()for k in a.files}
   assert np.array_equal(meta['xy'],np.concatenate(points[policy]))and np.array_equal(meta['labels'],np.concatenate(targets[policy]))and int(meta['video_id'])==int(clip['group'])
   for bone,dim in [('S14',384),('B14',768)]:
    with np.load(work.fp(bone,policy,clip['group']))as a:record={k:a[k].copy()for k in a.files}
    assert all(np.array_equal(record[k],v)for k,v in meta.items())and record['decoded'].shape==(len(meta['labels']),dim)and record['decoded'].dtype==np.float16 and np.isfinite(record['decoded']).all();packages[policy][bone].append(record)
  print('verified new capacity GT',clip['group'],flush=True)
 assert coverage==json.loads((ROOT/'query_coverage.json').read_text())and file_hash(ROOT/'query_coverage.json')==s['query_coverage_sha256']
 for section in ('query_hashes','feature_hashes'):
  for path,h in s[section].items():assert file_hash(path)==h
  assert set(s[section])=={str(path)for path in (ROOT/('queries'if section=='query_hashes'else'features')).rglob('*.npz')}
 featurechecks={}
 for bone in ('S14','B14'):
  firsts={policy:next(i for i,a in enumerate(packages[policy][bone])if len(a['labels']))for policy in work.POLICIES};model=work.rgb.prior.resolution.load_dino()if bone=='S14'else work.load_b14()
  for index in sorted(set(firsts.values())):
   points=np.concatenate([packages[a][bone][index]['xy']for a in work.POLICIES]);paths=[str(work.PANEL/'images'/im['file_name'])for im in clips[index]['frames']]
   values=independent_features(model,paths,points)if bone=='S14'else independent_b14(model,paths,points);offset=0
   for policy in work.POLICIES:
    target=packages[policy][bone][index]['decoded'];assert np.array_equal(values[offset:offset+len(target)],target);offset+=len(target)
  featurechecks[bone]={policy:clips[i]['group']for policy,i in firsts.items()};del model;torch.cuda.empty_cache()
 for policy in work.POLICIES:
  result=s['policies'][policy];rebuilt={};base=packages[policy]['S14'];labels=np.concatenate([a['labels']for a in base]);times=np.concatenate([a['xy'][:,2]for a in base]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in base]);rows=[r for r in coverage if r['policy']==policy]
  assert len(labels)==result['total_queries']and {str(int(c)):int(n)for c,n in zip(*np.unique(labels,return_counts=True))}==result['actual_support']
  assert {c['group']:len(a['labels'])for c,a in zip(clips,base)}==result['query_counts']and [c['group']for c,a in zip(clips,base)if not len(a['labels'])]==result['empty_groups']
  assert sum(not r['query_count']for r in rows)==result['empty_frames']and sum(r['distinct_query_pixels']for r in rows)==result['distinct_frame_query_pixels']
  for arm in work.ARMS:
   data=packages[policy][arm.split('_')[0]];rebuilt[arm]=[];offset=p['selection']['offsets'][arm];assert result['results'][arm]['offset']==offset
   for seed in range(3):
    checkpoint=work.ckpt(arm,seed);assert file_hash(checkpoint)==p['selection']['checkpoint_sha256'][f'{arm}_{seed}'];state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in data])
    with np.load(ROOT/policy/arm/'logits'/f'dinov2_{seed}_confirmation.npz')as a:
     assert np.array_equal(values,a['logits'])and np.array_equal(labels,a['labels'])and np.array_equal(times,a['times'])and np.array_equal(ids,a['video_ids'])and str(a['checkpoint_sha256'])==file_hash(checkpoint)
    shifted=values.copy();shifted[:,0]+=offset;pred=shifted.argmax(1);per={}
    for c in clips:
     ix=ids==int(c['group']);per[c['group']]=metric(pred[ix],labels[ix])[0]if ix.any()else None
    row={'seed':seed,'raw':metric(values.argmax(1),labels)[0],'calibrated':metric(pred,labels)[0],'per_video':per};equal(row,result['results'][arm]['runs'][seed]);rebuilt[arm].append(row)
    print('verified fresh capacity head',policy,arm,seed,flush=True)
  def check(c):
   left,right=c['left'],c['right'];d={}
   for clip in clips:
    g=clip['group'];d[g]=float(np.mean([rebuilt[left][i]['per_video'][g]['macro_f1_present']-rebuilt[right][i]['per_video'][g]['macro_f1_present']for i in range(3)]))if rebuilt[left][0]['per_video'][g]is not None else None
   equal(d,c['per_video_three_seed_mean_differences']);x=np.array([v for v in d.values()if v is not None]);draw=np.random.default_rng(20261060).integers(0,len(x),(2000,len(x)));b=c['conditional_video_bootstrap'];equal(float(x.mean()),b['mean_per_video_difference']);equal(np.quantile(x[draw].mean(1),[.025,.975]).tolist(),b['percentile_95_interval'])
   assert (int((x>0).sum()),int((x<0).sum()),int((x==0).sum()),len(x))==(b['positive_videos'],b['negative_videos'],b['tied_videos'],b['video_count'])
   pooled=[rebuilt[left][i]['calibrated']['macro_f1_present']-rebuilt[right][i]['calibrated']['macro_f1_present']for i in range(3)];equal(pooled,c['per_seed_pooled_macro_difference']);equal(float(np.mean(pooled)),c['mean_pooled_macro_difference'])
  check(result['primary_contrast'])
  for c in result['secondary_contrasts_unadjusted'].values():check(c)
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 for path,h in legacy.items():assert file_hash(path)==h
 assert s['model_or_calibration_updates']==0 and not s['old40_reserve_evaluated']and s['deployment_unchanged']=='native_gt'
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'new_group_count':40,'GT_frames_two_policies_replayed':640,'feature_first_effective_group_replays':featurechecks,'logit_packages_recomputed':24,'all_six_policy_contrast_bootstraps_recomputed':True,'direct_additional_legacy_backbone_fingerprints_verified':len(legacy),'main_freeze_S14_weight_and_vendor_were_indirect_recipe_references':True,'all_old_inputs_models_offsets_unchanged':True,'original40_reserve_unprocessed':True})
 print('DONE independent fresh external capacity confirmation verification',flush=True)
if __name__=='__main__':main()
