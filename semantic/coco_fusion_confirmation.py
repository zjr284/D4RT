"""Frozen complementary feature confirmation; no selection from these image scores."""
from collections import defaultdict
import fcntl,gc,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pycocotools.coco import COCO
from semantic import coco_sam_auxiliary as dino
from semantic import coco_fusion_cohort as resources
from semantic import feature_fusion_control as models
from semantic import segformer_precision_features as segmentation
from semantic.coco_semantic_adapter import semantic_maps
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.background_calibration import predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics
ROOT=Path('semantic_artifacts/coco_fusion_confirmation');ARMS=models.ARMS;POLICIES=('strict','consensus');BONES=('B14','mit_encoder_fp32','ade_encoder_fp32','ade_decoder_fp32');PROFILES=('raw','balanced_profile','spatial_profile')
def qp(im):return ROOT/'queries'/f'{im}.npz'
def fp(bone,im):return ROOT/'features'/bone/f'{im}.npz'

def freeze():
 s=json.loads((resources.ROOT/'summary.json').read_text());assert file_hash(resources.ROOT/'summary.json')==json.loads((resources.ROOT/'verification.json').read_text())['summary_sha256']
 m=json.loads((resources.ROOT/'manifest.json').read_text());eligible=json.loads((resources.ROOT/'frozen_eligibility.json').read_text());assert file_hash(resources.ROOT/'frozen_eligibility.json')==s['frozen_eligibility_sha256']
 images=[im for im in m['candidate']if im['id']in eligible['candidate']];assert [im['id']for im in images]==eligible['candidate']
 records={r['image_id']:r for r in json.loads((resources.ROOT/'image_records.json').read_text())}
 selection=s['protocol']['models'];assert s['protocol']['primary_model_fixed']=='b14_ade_decoder' and s['protocol']['primary_reference']=='b14_zero'
 source=json.loads((models.ROOT/'summary.json').read_text());assert source['validation_selected_arm']=='b14_ade_decoder' and source['eligible_for_separately_preregistered_confirmation']==['b14_ade_decoder']
 stats=json.loads((models.prior.ROOT/'statistics_manifest.json').read_text());legacy=json.loads((models.prior.prior.ROOT/'summary.json').read_text())
 paths=[Path(x)for x in {**s['protocol']['input_sha256'],**legacy['protocol']['input_sha256']}]
 paths.extend(resources.ROOT/name for name in ('summary.json','verification.json','manifest.json','image_records.json','frozen_eligibility.json','rgb_review.json'))
 paths.extend([Path('semantic/coco_fusion_confirmation.py'),Path('semantic/OVIS_COCO_FUSION_CONFIRMATION_PROTOCOL.md'),Path('semantic/coco_semantic_adapter.py'),Path('semantic/external_coco_adapter.py'),Path('semantic/external_label_sensitivity.py'),Path('semantic/segformer_precision_features.py'),Path('semantic/segformer_features.py'),Path('semantic/feature_fusion_control.py'),Path('semantic/feature_standardization_control.py'),dino.cohort.META/'instances_val2017.json',dino.cohort.META/'summary.json',dino.WEIGHTS,models.prior.ROOT/'statistics_manifest.json'])
 assert file_hash(dino.WEIGHTS)=='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
 paths.extend(sorted(dino.VENDOR.rglob('*.py')));paths.extend(Path(records[im['id']]['path'])for im in images)
 for row in stats.values():assert file_hash(row['path'])==row['sha256'];paths.append(Path(row['path']))
 for arm in ARMS:
  assert selection[arm]['offsets']==source['results'][arm]['offsets']
  for seed in range(3):assert file_hash(models.ckpt(arm,seed))==selection[arm]['checkpoint_sha256'][str(seed)];paths.append(models.ckpt(arm,seed))
 p={'round':86,'models':selection,'statistics':stats,'primary_model_fixed':'b14_ade_decoder','primary_reference':'b14_zero','primary_profile':'spatial_profile','bootstrap_seed':20261086,'bootstrap_repetitions':2000,'query_seed_base':20261077,'queries_per_image':1024,'image_batch':1,'training_updates':0,'calibration_updates':0,'input_sha256':{str(path):file_hash(path)for path in sorted(set(paths))}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',{'candidate':images,'excluded_previously_evaluated_COCO_ids':m['excluded_previously_evaluated_COCO_ids'],'activated_original_metadata_reserve':True})
 return images,records,p

def build(images):
 if(ROOT/'coverage.json').exists():return json.loads((ROOT/'coverage.json').read_text())
 api=COCO(str(dino.cohort.META/'instances_val2017.json'));mapping={int(k):int(v)for k,v in json.loads((dino.cohort.META/'summary.json').read_text())['shared_name_mapping'].items()};coverage=[]
 for im in images:
  chosen=np.random.default_rng(20261077+im['id']).choice(16384,1024,replace=False);xy=np.column_stack((chosen%128,chosen//128,np.zeros(1024))).astype(np.int16);maps=semantic_maps(api,im,mapping);package={'xy':xy,'image_id':np.int64(im['id'])}
  for policy,labels in maps.items():
   target=labels[xy[:,1],xy[:,0]];package[policy]=target;coverage.append({'image_id':im['id'],'policy':policy,'valid_query_count':int((target>=0).sum()),'ignored_query_count':int((target<0).sum()),'target_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(target,return_counts=True))}})
  save_npz(qp(im['id']),**package)
 atomic_json(ROOT/'coverage.json',coverage);return coverage

def cache(images,records):
 for alias in ('B14','mit','ade'):
  branches=('B14',)if alias=='B14'else(('mit_encoder_fp32',)if alias=='mit'else('ade_encoder_fp32','ade_decoder_fp32'))
  tasks=[im for im in images if not all(fp(branch,im['id']).exists()for branch in branches)]
  if not tasks:continue
  model=dino.load_b14()if alias=='B14'else segmentation.load_official(alias)
  for im in tasks:
   with np.load(qp(im['id']))as a:xy=a['xy'].copy()
   video=dino.rgb.load_native_video({'frames':[records[im['id']]['path']]}).cuda()
   if alias=='B14':
    maps=dino.tokens_b14(model,video)
    with torch.inference_mode():values={'B14':patch_at_queries(maps,xy).half().cpu().numpy()}
   else:
    maps=segmentation.feature_maps(model,alias,video,'fp32');values={branch:segmentation.at_queries(maps[branch.replace('_fp32','')],xy).half().cpu().numpy()for branch in branches}
   for branch,decoded in values.items():
    assert decoded.shape==(1024,768 if branch=='B14'else 256)and np.isfinite(decoded).all()
    save_npz(fp(branch,im['id']),decoded=decoded,xy=xy,image_id=np.int64(im['id']))
   print('frozen new image feature',alias,im['id'],flush=True)
  del model;gc.collect();torch.cuda.empty_cache()

def metric(pred,target):
 valid=target>=0;pred=pred[valid];target=target[valid]
 if not len(target):return None
 r=classification_metrics(pred,target,26);bg=target==0;fg=target>0;nonperson=target>1;r.update(valid_query_count=len(target),background_queries=int(bg.sum()),foreground_queries=int(fg.sum()),background_to_foreground_rate=float((pred[bg]!=0).mean())if bg.any()else None,foreground_to_background_rate=float((pred[fg]==0).mean())if fg.any()else None,foreground_class_confusion_rate=float(((pred[fg]>0)&(pred[fg]!=target[fg])).mean())if fg.any()else None,predicted_class_counts=np.bincount(pred,minlength=26).tolist(),nonperson_foreground_conditional=classification_metrics(pred[nonperson],target[nonperson],26)if nonperson.any()else None);return r

def inputs(arm,packages,stats):
 def normalized(bone,raw):
  with np.load(stats[raw]['path'])as a:mu=a['mean'].copy();scale=a['scale'].copy()
  return [(x.astype(np.float32)-mu)/scale for x in packages[bone]]
 base=normalized('B14','b14_gt32')
 if arm==models.REFERENCE:return base
 if arm=='b14_zero':extra=[np.zeros((len(x),256),np.float32)for x in base]
 else:
  raw={'b14_mit':'mit_encoder_fp32','b14_ade_encoder':'ade_encoder_fp32','b14_ade_decoder':'ade_decoder_fp32'}[arm];extra=normalized(raw,raw)
 return [np.concatenate([a,b],axis=1)for a,b in zip(base,extra)]

def contrast(left,right,left_profile,right_profile,results,images,policy,p):
 l=results[left]['runs'];r=results[right]['runs'];pooled=[l[seed]['policies'][policy][left_profile]['macro_f1_present']-r[seed]['policies'][policy][right_profile]['macro_f1_present']for seed in range(3)];per={}
 for im in images:
  key=str(im['id']);valid=l[0]['policies'][policy]['per_image'][key][left_profile]is not None
  per[key]=float(np.mean([l[seed]['policies'][policy]['per_image'][key][left_profile]['macro_f1_present']-r[seed]['policies'][policy]['per_image'][key][right_profile]['macro_f1_present']for seed in range(3)]))if valid else None
 x=np.array([v for v in per.values()if v is not None]);draw=np.random.default_rng(p['bootstrap_seed']).integers(0,len(x),(p['bootstrap_repetitions'],len(x)))
 return {'left_arm':left,'right_arm':right,'left_profile':left_profile,'right_profile':right_profile,'per_seed_pooled_macro_difference':pooled,'mean_pooled_macro_difference':float(np.mean(pooled)),'per_image_three_seed_mean_macro_difference':per,'conditional_image_bootstrap':{'mean_per_image_macro_difference':float(x.mean()),'positive_images':int((x>0).sum()),'negative_images':int((x<0).sum()),'tied_images':int((x==0).sum()),'valid_images':len(x),'percentile_95_interval':np.quantile(x[draw].mean(1),[.025,.975]).tolist(),'note':'conditional image-equal macro difference, not pooled-query macro CI/video/AP'}}

def contrast_specs():
 primary=('b14_ade_decoder','b14_zero','spatial_profile','spatial_profile')
 pairs=[('b14_ade_decoder',models.REFERENCE),('b14_ade_encoder','b14_mit')]+[(arm,'b14_zero')for arm in models.NEW[1:]]+[(arm,models.REFERENCE)for arm in models.NEW]
 secondary=[]
 for left,right in pairs:
  for profile in PROFILES:
   spec=(left,right,profile,profile)
   if spec!=primary and spec not in secondary:secondary.append(spec)
 secondary.append(('b14_ade_decoder','b14_ade_decoder','spatial_profile','raw'))
 return primary,secondary

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed frozen feature confirmation exists')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  images,records,p=freeze();build(images);cache(images,records);packages={bone:[]for bone in BONES};targets={policy:[]for policy in POLICIES}
  for im in images:
   with np.load(qp(im['id']))as a:
    for policy in POLICIES:targets[policy].append(a[policy].copy())
   for bone in BONES:
    with np.load(fp(bone,im['id']))as a:packages[bone].append(a['decoded'].copy())
  targets={key:np.concatenate(v)for key,v in targets.items()};ids=np.repeat([im['id']for im in images],1024).astype(np.int32);results={}
  for arm in ARMS:
   x=inputs(arm,packages,p['statistics']);offsets=p['models'][arm]['offsets'];runs=[]
   for seed in range(3):
    checkpoint=models.ckpt(arm,seed);state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model'];weight=state['classifier.weight'].cuda();bias=state['classifier.bias'].cuda()
    with torch.inference_mode():values=np.concatenate([F.linear(torch.from_numpy(a).cuda(),weight,bias).cpu().numpy()for a in x])
    path=ROOT/'logits'/f'{arm}_{seed}.npz';save_npz(path,logits=values,image_ids=ids,checkpoint_sha256=np.array(file_hash(checkpoint)));policies={}
    for policy,target in targets.items():
     modes={profile:predictions(values,offset)for profile,offset in offsets.items()};policies[policy]={profile:metric(pred,target)for profile,pred in modes.items()};policies[policy]['per_image']={str(im['id']):{profile:metric(pred[ids==im['id']],target[ids==im['id']])for profile,pred in modes.items()}for im in images}
    runs.append({'seed':seed,'policies':policies,'output_path':str(path),'output_sha256':file_hash(path)});print('frozen fusion confirmation',arm,seed,{mode:policies['strict'][mode]['macro_f1_present']for mode in offsets},flush=True)
   results[arm]={'offsets':offsets,'runs':runs}
  primary,secondary=contrast_specs();contrasts={}
  for policy in POLICIES:
   contrasts[policy]={'primary_contrast':contrast(*primary,results,images,policy,p),'secondary_contrasts_unadjusted':{'__'.join(spec):contrast(*spec,results,images,policy,p)for spec in secondary}}
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'coverage_sha256':file_hash(ROOT/'coverage.json'),'query_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'queries').glob('*.npz'))},'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'features').rglob('*.npz'))},'results':results,'contrasts':contrasts,'total_coordinates':len(images)*1024,'confirmation_images_fixed_before_evaluation':len(images),'new_training_or_calibration_updates':0,'deployment_unchanged':'native_gt'})
  print('DONE frozen complementary feature confirmation',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
