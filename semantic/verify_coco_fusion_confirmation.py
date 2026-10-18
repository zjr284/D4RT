"""Independent fresh40 GT, uniformXY, logits and28 image-bootstrap audits."""
from collections import defaultdict
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco
import torch
from semantic import coco_fusion_confirmation as work
from semantic.verify_coco_semantic_confirmation import independent_feature
from semantic.verify_coco_uniform_queries import independent_metrics
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from semantic.verify_segformer_feature_control import manual_maps,sample
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text()) and p['round']==86 and p['primary_model_fixed']=='b14_ade_decoder' and p['primary_reference']=='b14_zero' and p['primary_profile']=='spatial_profile' and p['bootstrap_seed']==20261086 and p['bootstrap_repetitions']==2000
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 resource=json.loads((work.resources.ROOT/'summary.json').read_text());assert p['models']==resource['protocol']['models']
 manifest=json.loads((ROOT/'manifest.json').read_text());images=manifest['candidate'];eligible=json.loads((work.resources.ROOT/'frozen_eligibility.json').read_text());assert [im['id']for im in images]==eligible['candidate'] and len(images)==40 and not(set(eligible['candidate'])&set(manifest['excluded_previously_evaluated_COCO_ids']))
 source=json.loads((work.models.ROOT/'summary.json').read_text());assert source['validation_selected_arm']=='b14_ade_decoder' and source['eligible_for_separately_preregistered_confirmation']==['b14_ade_decoder']
 stats=json.loads((work.models.prior.ROOT/'statistics_manifest.json').read_text());assert stats==p['statistics']
 for arm,config in p['models'].items():assert config['offsets']==source['results'][arm]['offsets']
 payload=json.loads((work.dino.cohort.META/'instances_val2017.json').read_bytes());mapping={int(k):int(v)for k,v in json.loads((work.dino.cohort.META/'summary.json').read_text())['shared_name_mapping'].items()};by=defaultdict(list)
 for ann in payload['annotations']:by[ann['image_id']].append(ann)
 coverage=[];targets={policy:[]for policy in work.POLICIES};packages={bone:[]for bone in work.BONES};coordinates=[]
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
  for bone,dim in [(bone,768 if bone=='B14'else 256)for bone in work.BONES]:
   with np.load(work.fp(bone,im['id']))as a:assert np.array_equal(a['xy'],xy)and int(a['image_id'])==im['id']and a['decoded'].shape==(1024,dim)and a['decoded'].dtype==np.float16 and np.isfinite(a['decoded']).all();packages[bone].append(a['decoded'].copy())
  print('verified fresh spatial COCO two GT policies/XY',im['id'],flush=True)
 assert coverage==json.loads((ROOT/'coverage.json').read_text())and file_hash(ROOT/'coverage.json')==s['coverage_sha256'];targets={k:np.concatenate(v)for k,v in targets.items()};ids=np.repeat([im['id']for im in images],1024).astype(np.int32);records={r['image_id']:r for r in json.loads((work.resources.ROOT/'image_records.json').read_text())}
 for key,folder in [('query_hashes','queries'),('feature_hashes','features')]:
  assert set(s[key])=={str(path)for path in (ROOT/folder).rglob('*.npz')}
  for path,h in s[key].items():assert file_hash(path)==h
 first=images[0]['id'];model=work.dino.load_b14();rebuilt_feature=independent_feature(model,records[first]['path'],coordinates[0],768);assert np.array_equal(rebuilt_feature,packages['B14'][0]);del model;torch.cuda.empty_cache()
 with Image.open(records[first]['path'])as im:array=np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32)
 video=(torch.from_numpy(array).permute(2,0,1)[None]/255).cuda()
 for alias in ('mit','ade'):
  model=work.segmentation.load_official(alias);maps=manual_maps(model,alias,video,'fp32')
  for branch,value in maps.items():assert np.array_equal(sample(value,coordinates[0]),packages[branch+'_fp32'][0])
  del model;torch.cuda.empty_cache()
 normalized={}
 for bone in work.BONES:
  raw='b14_gt32'if bone=='B14'else bone;record=stats[raw];assert file_hash(record['path'])==record['sha256']
  with np.load(record['path'])as a:mean=a['mean'].copy();scale=a['scale'].copy()
  normalized[bone]=[]
  for x in packages[bone]:
   value=x.astype(np.float32);np.subtract(value,mean,out=value);np.divide(value,scale,out=value);normalized[bone].append(value)
 rebuilt={}
 for arm in work.ARMS:
  if arm==work.models.REFERENCE:inputs=normalized['B14']
  else:
   branch={'b14_mit':'mit_encoder_fp32','b14_ade_encoder':'ade_encoder_fp32','b14_ade_decoder':'ade_decoder_fp32'}.get(arm)
   extra=normalized[branch]if branch else[np.zeros((len(x),256),np.float32)for x in normalized['B14']]
   inputs=[np.column_stack((x,y))for x,y in zip(normalized['B14'],extra)]
  config=p['models'][arm];offsets=config['offsets'];assert offsets==s['results'][arm]['offsets'];rebuilt[arm]=[]
  for seed in range(3):
   path=work.models.ckpt(arm,seed);assert file_hash(path)==config['checkpoint_sha256'][str(seed)];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(x,state)for x in inputs]);r=s['results'][arm]['runs'][seed];assert file_hash(r['output_path'])==r['output_sha256']
   with np.load(r['output_path'])as a:assert np.array_equal(values,a['logits']) and np.array_equal(ids,a['image_ids']) and str(a['checkpoint_sha256'])==file_hash(path)
   result={}
   for policy,target in targets.items():
    modes={}
    for profile,offset in offsets.items():shifted=values.copy();shifted[:,0]+=offset;modes[profile]=shifted.argmax(-1)
    metrics={profile:independent_metrics(pred,target)for profile,pred in modes.items()};metrics['per_image']={str(im['id']):{profile:independent_metrics(pred[ids==im['id']],target[ids==im['id']])for profile,pred in modes.items()}for im in images};equal(metrics,r['policies'][policy]);result[policy]=metrics
   rebuilt[arm].append(result);print('verified full frozen fusion logits/profiles',arm,seed,flush=True)
 primary=('b14_ade_decoder','b14_zero','spatial_profile','spatial_profile');pairs=[('b14_ade_decoder','z_b14_gt32'),('b14_ade_encoder','b14_mit')]+[(arm,'b14_zero')for arm in ('b14_mit','b14_ade_encoder','b14_ade_decoder')]+[(arm,'z_b14_gt32')for arm in ('b14_zero','b14_mit','b14_ade_encoder','b14_ade_decoder')]
 secondary=[]
 for left,right in pairs:
  for profile in ('raw','balanced_profile','spatial_profile'):
   spec=(left,right,profile,profile)
   if spec!=primary and spec not in secondary:secondary.append(spec)
 secondary.append(('b14_ade_decoder','b14_ade_decoder','spatial_profile','raw'));count=0
 for policy in work.POLICIES:
  group=s['contrasts'][policy];expected_keys={'__'.join(spec)for spec in secondary};assert set(group['secondary_contrasts_unadjusted'])==expected_keys
  entries=[(primary,group['primary_contrast'])]+[(spec,group['secondary_contrasts_unadjusted']['__'.join(spec)])for spec in secondary]
  for spec,c in entries:
   left,right,lp,rp=spec;assert(c['left_arm'],c['right_arm'],c['left_profile'],c['right_profile'])==spec
   per={}
   for im in images:
    key=str(im['id']);rows=[(rebuilt[left][seed][policy]['per_image'][key][lp],rebuilt[right][seed][policy]['per_image'][key][rp])for seed in range(3)]
    per[key]=float(np.mean([a['macro_f1_present']-b['macro_f1_present']for a,b in rows]))if rows[0][0]is not None else None
   pooled=[rebuilt[left][seed][policy][lp]['macro_f1_present']-rebuilt[right][seed][policy][rp]['macro_f1_present']for seed in range(3)];equal(per,c['per_image_three_seed_mean_macro_difference']);equal(pooled,c['per_seed_pooled_macro_difference']);equal(float(np.mean(pooled)),c['mean_pooled_macro_difference'])
   x=np.array([v for v in per.values()if v is not None]);draw=np.random.default_rng(20261086).integers(0,len(x),(2000,len(x)));b=c['conditional_image_bootstrap'];equal(float(x.mean()),b['mean_per_image_macro_difference']);equal(np.quantile(x[draw].mean(1),[.025,.975]).tolist(),b['percentile_95_interval']);assert(b['positive_images'],b['negative_images'],b['tied_images'],b['valid_images'])==(int((x>0).sum()),int((x<0).sum()),int((x==0).sum()),len(x));count+=1
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['total_coordinates']==40960 and s['confirmation_images_fixed_before_evaluation']==40 and s['new_training_or_calibration_updates']==0 and s['deployment_unchanged']=='native_gt'
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all40_uniformXY_and80_GT_policy_maps_independently_recomputed':True,'all160_feature_metadata_and_hashes_verified':True,'first222235_B14_patch_and_MIT_ADE_manual_multiscale_replays_bitwise_equal':True,'all15_logits_and90_policy_profile_metrics_with_per_image_metrics_recomputed':True,'all_predefined_image_pair_bootstraps_recomputed':count,'source_selection_statistics_and_three_profiles_frozen':True,'all_frozen_inputs_unchanged':True,'no_training_recalibration_or_deployment_change':True})
 print('DONE independent frozen complementary feature confirmation',count,flush=True)
if __name__=='__main__':main()
