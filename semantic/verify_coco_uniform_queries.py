"""Independent GT bitsets, uniformXY, source features and all21 frozen logits."""
from collections import defaultdict
import inspect,json
import numpy as np
from PIL import Image
from pycocotools import mask as coco
import torch
from torch.nn import functional as F
from semantic import coco_uniform_queries as work
from semantic.verify_coco_semantic_confirmation import independent_feature
from semantic.verify_external_semantic_confirmation import metric,equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def independent_metrics(pred,target):
 valid=target>=0;pred=pred[valid];target=target[valid]
 if not len(target):return None
 matrix=np.bincount(target.astype(np.int64)*26+pred,minlength=676).reshape(26,26);support=matrix.sum(1);predicted=matrix.sum(0);present=np.flatnonzero(support);f1={str(int(c)):float(2*matrix[c,c]/(support[c]+predicted[c]))for c in present};fg_values=[f1[str(int(c))]for c in present if c>0];out={'accuracy':float(np.trace(matrix)/matrix.sum()),'macro_f1_present':float(np.mean(list(f1.values()))),'foreground_macro_f1_present':float(np.mean(fg_values))if fg_values else None,'support':{str(int(c)):int(support[c])for c in present},'per_class_f1':f1};n=matrix.sum();bg=matrix[0].sum();fg=matrix[1:].sum();nonperson=target>1
 out.update(valid_query_count=int(n),background_queries=int(bg),foreground_queries=int(fg),background_to_foreground_rate=float(matrix[0,1:].sum()/bg)if bg else None,foreground_to_background_rate=float(matrix[1:,0].sum()/fg)if fg else None,foreground_class_confusion_rate=float((matrix[1:,1:].sum()-np.trace(matrix[1:,1:]))/fg)if fg else None,predicted_class_counts=matrix.sum(0).tolist(),nonperson_foreground_conditional=metric(pred[nonperson],target[nonperson])[0]if nonperson.any()else None);return out

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 check=independent_metrics(np.array([0,1]),np.array([0,0]));assert check['foreground_macro_f1_present']is None and check['foreground_to_background_rate']is None and check['background_to_foreground_rate']==.5 and abs(check['macro_f1_present']-2/3)<1e-12;assert independent_metrics(np.array([0]),np.array([-1]))is None
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==77 and p['query_coordinates_independent_of_GT']and p['queries_per_image']==1024
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 amendment=json.loads((ROOT/'import_amendment.json').read_text());assert file_hash(ROOT/'import_amendment.json')==s['runtime_import_amendment_sha256']and amendment['original_protocol_sha256']==file_hash(ROOT/'protocol.json')and amendment['original_frozen_source_sha256']==file_hash('semantic/coco_uniform_queries.py')and amendment['recovery_source_sha256']==file_hash('semantic/coco_uniform_queries_resume.py')and amendment['torch_functional_sha256']==file_hash(inspect.getfile(F))and amendment['initial_failure_sha256']==file_hash(ROOT/'failure.json')and amendment['logits_before_recovery']==amendment['training_or_calibration_updates']==0
 for key,hashes in [('query_sha256',s['query_hashes']),('feature_sha256',s['feature_hashes'])]:assert amendment[key]==hashes
 images=json.loads((ROOT/'manifest.json').read_text())['candidate'];eligible=json.loads((work.prior.PANEL/'frozen_eligibility.json').read_text());assert [im['id']for im in images]==eligible['candidate'];payload=json.loads((work.prior.cohort.META/'instances_val2017.json').read_bytes());mapping={int(k):int(v)for k,v in json.loads((work.prior.cohort.META/'summary.json').read_text())['shared_name_mapping'].items()};by=defaultdict(list)
 for ann in payload['annotations']:by[ann['image_id']].append(ann)
 coverage=[];targets={policy:[]for policy in work.POLICIES};packages={bone:[]for bone in ('S14','B14')};coordinates=[]
 for im in images:
  count=np.zeros((128,128),np.int32);owner=np.zeros_like(count);crowd=np.zeros((128,128),bool);bits=np.zeros((128,128),np.uint32)
  for ann in by[im['id']]:
   seg=ann['segmentation']
   if isinstance(seg,list):rle=coco.merge(coco.frPyObjects(seg,im['height'],im['width']))
   elif isinstance(seg['counts'],list):rle=coco.frPyObjects(seg,im['height'],im['width'])
   else:rle=seg
   mask=np.asarray(Image.fromarray(coco.decode(rle).astype('uint8')).resize((128,128),Image.Resampling.NEAREST),bool);count+=mask;cid=mapping.get(ann['category_id']);is_crowd=bool(ann['iscrowd']);owner[mask]=-1 if cid is None or is_crowd else cid
   if is_crowd:crowd|=mask
   elif cid is not None:bits[mask]|=np.uint32(1<<cid)
  strict=np.where(count==0,0,np.where(count==1,owner,-1)).astype(np.int16);consensus=np.full((128,128),-1,np.int16);consensus[count==0]=0
  for cid in range(1,26):consensus[(bits==np.uint32(1<<cid))&~crowd]=cid
  chosen=np.random.default_rng(20261077+im['id']).choice(16384,1024,replace=False);xy=np.column_stack((chosen%128,chosen//128,np.zeros(1024))).astype(np.int16);assert len(np.unique(xy[:,:2],axis=0))==1024;coordinates.append(xy)
  with np.load(work.qp(im['id']))as a:
   assert np.array_equal(xy,a['xy'])and int(a['image_id'])==im['id']
   for policy,labels in [('strict',strict),('consensus',consensus)]:
    target=labels[xy[:,1],xy[:,0]];assert np.array_equal(target,a[policy]);targets[policy].append(target);coverage.append({'image_id':im['id'],'policy':policy,'valid_query_count':int((target>=0).sum()),'ignored_query_count':int((target<0).sum()),'target_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(target,return_counts=True))}})
  for bone,dim in [('S14',384),('B14',768)]:
   with np.load(work.fp(bone,im['id']))as a:assert np.array_equal(a['xy'],xy)and int(a['image_id'])==im['id']and a['decoded'].shape==(1024,dim)and a['decoded'].dtype==np.float16 and np.isfinite(a['decoded']).all();packages[bone].append(a['decoded'].copy())
  print('verified uniform XY and two GT policies',im['id'],flush=True)
 assert coverage==json.loads((ROOT/'coverage.json').read_text())and file_hash(ROOT/'coverage.json')==s['coverage_sha256'];targets={k:np.concatenate(v)for k,v in targets.items()};ids=np.repeat([im['id']for im in images],1024).astype(np.int32);records={a['image_id']:a for a in json.loads((work.prior.PANEL/'image_records.json').read_text())}
 for key,folder in [('query_hashes','queries'),('feature_hashes','features')]:
  assert set(s[key])=={str(path)for path in (ROOT/folder).rglob('*.npz')}
  for path,h in s[key].items():assert file_hash(path)==h
 for bone,dimension in [('S14',384),('B14',768)]:
  model=work.prior.rgb.prior.resolution.load_dino()if bone=='S14'else work.prior.load_b14();features=independent_feature(model,records[images[0]['id']]['path'],coordinates[0],dimension);assert np.array_equal(features,packages[bone][0]);del model;torch.cuda.empty_cache()
 for arm in work.ARMS:
  offset=p['models'][arm]['offset'];assert s['results'][arm]['offset']==offset
  for seed in range(3):
   path=work.prior.ckpt(arm,seed);assert file_hash(path)==p['models'][arm]['checkpoint_sha256'][str(seed)];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a,state)for a in packages[arm.split('_')[0]]]);r=s['results'][arm]['runs'][seed];assert file_hash(r['output_path'])==r['output_sha256']
   with np.load(r['output_path'])as a:assert np.array_equal(values,a['logits'])and np.array_equal(ids,a['image_ids'])and str(a['checkpoint_sha256'])==file_hash(path)
   shifted=values.copy();shifted[:,0]+=offset;raw=values.argmax(-1);cal=shifted.argmax(-1)
   for policy,target in targets.items():
    expected={'raw':independent_metrics(raw,target),'calibrated':independent_metrics(cal,target),'per_image':{str(im['id']):{'raw':independent_metrics(raw[ids==im['id']],target[ids==im['id']]),'calibrated':independent_metrics(cal[ids==im['id']],target[ids==im['id']])}for im in images}};equal(expected,r['policies'][policy])
   print('verified uniform query head/raw-cal metrics',arm,seed,flush=True)
 contrasts={}
 for left,right in [('B14_class_video','S14_gt32'),('B14_aux_sam','B14_extra_gt'),('B14_aux_sam','B14_gt32'),('B14_weak','B14_gt256')]:
  contrasts[left+'_minus_'+right]={policy:{key:[s['results'][left]['runs'][seed]['policies'][policy]['calibrated'][key]-s['results'][right]['runs'][seed]['policies'][policy]['calibrated'][key]for seed in range(3)]for key in ('accuracy','macro_f1_present','foreground_macro_f1_present','background_to_foreground_rate','foreground_to_background_rate')}for policy in work.POLICIES}
 equal(contrasts,s['descriptive_contrasts'])
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['total_coordinates']==81920 and s['new_confirmation_groups']==0 and s['LVVIS40_not_evaluated']and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all80_uniform_XY_sets_independent_of_GT_recomputed':True,'all160_GT_query_targets_and_supports_recomputed':True,'both_first_image_frozen_features_recomputed':images[0]['id'],'all21_logits_and_two_policy_raw_calibrated_metrics_recomputed':True,'all4_descriptive_contrasts_recomputed':True,'background_only_and_all_ignored_metric_boundaries_explicitly_checked':True,'explicit_import_only_recovery_provenance_and_unchanged_queries_features_verified':True,'all_frozen_inputs_unchanged':True,'no_new_confirmation_or_calibration_or_deployment_change':True});print('DONE independent uniform query diagnostic',flush=True)
if __name__=='__main__':main()
