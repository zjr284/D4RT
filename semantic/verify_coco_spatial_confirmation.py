"""Independent fresh40 GT, uniformXY, logits and28 image-bootstrap audits."""
from collections import defaultdict
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco
import torch
from semantic import coco_spatial_confirmation as work
from semantic.verify_coco_semantic_confirmation import independent_feature
from semantic.verify_coco_uniform_queries import independent_metrics
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==80 and p['primary_model_fixed']=='B14_aux_sam'and p['primary_modes']==['spatial_calibration','old_calibration']and p['bootstrap_seed']==20261080 and p['bootstrap_repetitions']==2000
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 resources=json.loads((work.resources.ROOT/'summary.json').read_text());assert p['models']==resources['protocol']['models'];manifest=json.loads((ROOT/'manifest.json').read_text());images=manifest['candidate'];eligible=json.loads((work.resources.ROOT/'frozen_eligibility.json').read_text());assert [im['id']for im in images]==eligible['candidate']and len(images)==40 and manifest['metadata_only_reserve_ids']==eligible['metadata_only_reserve'];assert not set(eligible['candidate'])&set(manifest['excluded_previously_evaluated_COCO_ids'])
 source=json.loads((work.resources.calibration.ROOT/'frozen_selection.json').read_text())
 for arm,config in p['models'].items():assert config['old_offset']==source['old_offsets'][arm]and config['spatial_offset']==source['spatial_offsets'][arm]and config['checkpoint_sha256']==source['models'][arm]['checkpoint_sha256']
 payload=json.loads((work.prior.cohort.META/'instances_val2017.json').read_bytes());mapping={int(k):int(v)for k,v in json.loads((work.prior.cohort.META/'summary.json').read_text())['shared_name_mapping'].items()};by=defaultdict(list)
 for ann in payload['annotations']:by[ann['image_id']].append(ann)
 coverage=[];targets={policy:[]for policy in work.POLICIES};packages={bone:[]for bone in ('S14','B14')};coordinates=[]
 for im in images:
  count=np.zeros((128,128),np.int32);owner=np.zeros_like(count);crowd=np.zeros((128,128),bool);bits=np.zeros((128,128),np.uint32)
  for ann in by[im['id']]:
   seg=ann['segmentation']
   if isinstance(seg,list):rle=coco.merge(coco.frPyObjects(seg,im['height'],im['width']))
   elif isinstance(seg['counts'],list):rle=coco.frPyObjects(seg,im['height'],im['width'])
   else:rle=seg
   native=coco.decode(rle);assert native.shape==(im['height'],im['width']);mask=np.asarray(Image.fromarray(native.astype('uint8')).resize((128,128),Image.Resampling.NEAREST),bool);count+=mask;cid=mapping.get(ann['category_id']);is_crowd=bool(ann['iscrowd']);owner[mask]=-1 if cid is None or is_crowd else cid
   if is_crowd:crowd|=mask
   elif cid is not None:bits[mask]|=np.uint32(1<<cid)
  strict=np.where(count==0,0,np.where(count==1,owner,-1)).astype(np.int16);consensus=np.full((128,128),-1,np.int16);consensus[count==0]=0
  for cid in range(1,26):consensus[(bits==np.uint32(1<<cid))&~crowd]=cid
  selected=np.random.default_rng(20261077+im['id']).choice(16384,1024,replace=False);xy=np.column_stack((selected%128,selected//128,np.zeros(1024))).astype(np.int16);assert len(np.unique(xy[:,:2],axis=0))==1024;coordinates.append(xy)
  with np.load(work.qp(im['id']))as a:
   assert np.array_equal(xy,a['xy'])and int(a['image_id'])==im['id']
   for policy,labels in [('strict',strict),('consensus',consensus)]:
    target=labels[xy[:,1],xy[:,0]];assert np.array_equal(target,a[policy]);targets[policy].append(target);coverage.append({'image_id':im['id'],'policy':policy,'valid_query_count':int((target>=0).sum()),'ignored_query_count':int((target<0).sum()),'target_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(target,return_counts=True))}})
  for bone,dim in [('S14',384),('B14',768)]:
   with np.load(work.fp(bone,im['id']))as a:assert np.array_equal(a['xy'],xy)and int(a['image_id'])==im['id']and a['decoded'].shape==(1024,dim)and a['decoded'].dtype==np.float16 and np.isfinite(a['decoded']).all();packages[bone].append(a['decoded'].copy())
  print('verified fresh spatial COCO two GT policies/XY',im['id'],flush=True)
 assert coverage==json.loads((ROOT/'coverage.json').read_text())and file_hash(ROOT/'coverage.json')==s['coverage_sha256'];targets={k:np.concatenate(v)for k,v in targets.items()};ids=np.repeat([im['id']for im in images],1024).astype(np.int32);records={r['image_id']:r for r in json.loads((work.resources.ROOT/'image_records.json').read_text())}
 for key,folder in [('query_hashes','queries'),('feature_hashes','features')]:
  assert set(s[key])=={str(path)for path in (ROOT/folder).rglob('*.npz')}
  for path,h in s[key].items():assert file_hash(path)==h
 for bone,dimension in [('S14',384),('B14',768)]:
  model=work.prior.rgb.prior.resolution.load_dino()if bone=='S14'else work.prior.load_b14();feature=independent_feature(model,records[images[0]['id']]['path'],coordinates[0],dimension);assert np.array_equal(feature,packages[bone][0]);del model;torch.cuda.empty_cache()
 rebuilt={}
 for arm in work.ARMS:
  config=p['models'][arm];offsets={'raw':0.,'old_calibration':config['old_offset'],'spatial_calibration':config['spatial_offset']};assert offsets==s['results'][arm]['offsets'];rebuilt[arm]=[]
  for seed in range(3):
   path=work.prior.ckpt(arm,seed);assert file_hash(path)==config['checkpoint_sha256'][str(seed)];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a,state)for a in packages[arm.split('_')[0]]]);r=s['results'][arm]['runs'][seed];assert file_hash(r['output_path'])==r['output_sha256']
   with np.load(r['output_path'])as a:assert np.array_equal(values,a['logits'])and np.array_equal(ids,a['image_ids'])and str(a['checkpoint_sha256'])==file_hash(path)
   result={}
   for policy,target in targets.items():
    modes={}
    for mode,offset in offsets.items():shifted=values.copy();shifted[:,0]+=offset;modes[mode]=shifted.argmax(-1)
    expected={mode:independent_metrics(pred,target)for mode,pred in modes.items()};expected['per_image']={str(im['id']):{mode:independent_metrics(pred[ids==im['id']],target[ids==im['id']])for mode,pred in modes.items()}for im in images};equal(expected,r['policies'][policy]);result[policy]=expected
   rebuilt[arm].append(result);print('verified fresh spatial COCO21 raw logits/three offsets',arm,seed,flush=True)
 contrasts_verified=0
 for policy in work.POLICIES:
  group=s['contrasts'][policy];primary=group['primary_contrast'];assert(primary['arm'],primary['left'],primary['right'])==('B14_aux_sam','spatial_calibration','old_calibration');expected_secondary={a+'_spatial_minus_'+right for a in work.ARMS for right in ('old_calibration','raw')if(a,right)!=('B14_aux_sam','old_calibration')};assert set(group['secondary_contrasts_unadjusted'])==expected_secondary
  for c in [primary,*group['secondary_contrasts_unadjusted'].values()]:
   arm,left,right=c['arm'],c['left'],c['right'];per={}
   for im in images:
    key=str(im['id']);rows=[rebuilt[arm][seed][policy]['per_image'][key]for seed in range(3)];per[key]=float(np.mean([row[left]['macro_f1_present']-row[right]['macro_f1_present']for row in rows]))if rows[0][left]is not None else None
   pooled=[rebuilt[arm][seed][policy][left]['macro_f1_present']-rebuilt[arm][seed][policy][right]['macro_f1_present']for seed in range(3)];equal(per,c['per_image_three_seed_mean_macro_difference']);equal(pooled,c['per_seed_pooled_macro_difference']);equal(float(np.mean(pooled)),c['mean_pooled_macro_difference']);x=np.array([v for v in per.values()if v is not None]);draw=np.random.default_rng(20261080).integers(0,len(x),(2000,len(x)));b=c['conditional_image_bootstrap'];equal(float(x.mean()),b['mean_per_image_macro_difference']);equal(np.quantile(x[draw].mean(1),[.025,.975]).tolist(),b['percentile_95_interval']);assert(b['positive_images'],b['negative_images'],b['tied_images'],b['valid_images'])==(int((x>0).sum()),int((x<0).sum()),int((x==0).sum()),len(x));contrasts_verified+=1
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert not set(manifest['metadata_only_reserve_ids'])&{int(path.stem)for path in (ROOT/'features').rglob('*.npz')}and s['metadata_reserve40_and_LVVIS40_not_evaluated']and s['new_training_or_calibration_updates']==0 and s['confirmation_images_fixed_before_evaluation']==40 and s['total_coordinates']==40960 and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all40_uniformXY_and80_GT_policy_maps_recomputed':True,'both_first250205_static_feature_replays_bitwise_equal':True,'all21_logits_and126_policy_offset_metric_sets_recomputed':True,'all_image_pair_bootstraps_recomputed':contrasts_verified,'frozen_source_only_old_and_spatial_offset_provenances_verified':True,'all_frozen_inputs_unchanged':True,'metadata_COCO_reserve40_and_LVVIS40_not_evaluated':True,'no_training_recalibration_or_deployment_change':True});print('DONE independent fresh COCO spatial calibration confirmation',contrasts_verified,flush=True)
if __name__=='__main__':main()
