"""Frozen source-only spatial calibration confirmation on new COCO40 images."""
from collections import defaultdict
import fcntl,gc,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from pycocotools.coco import COCO
from semantic import coco_sam_auxiliary as prior
from semantic import coco_spatial_cohort as resources
from semantic.coco_semantic_adapter import semantic_maps
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.background_calibration import predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics
ROOT=Path('semantic_artifacts/coco_spatial_confirmation');ARMS=prior.ARMS;POLICIES=('strict','consensus')
def qp(im):return ROOT/'queries'/f'{im}.npz'
def fp(bone,im):return ROOT/'features'/bone/f'{im}.npz'

def freeze():
 s=json.loads((resources.ROOT/'summary.json').read_text());assert file_hash(resources.ROOT/'summary.json')==json.loads((resources.ROOT/'verification.json').read_text())['summary_sha256'];manifest=json.loads((resources.ROOT/'manifest.json').read_text());elig=json.loads((resources.ROOT/'frozen_eligibility.json').read_text());assert file_hash(resources.ROOT/'frozen_eligibility.json')==s['frozen_eligibility_sha256'];images=[im for im in manifest['candidate']if im['id']in elig['candidate']];assert [im['id']for im in images]==elig['candidate'];records={r['image_id']:r for r in json.loads((resources.ROOT/'image_records.json').read_text())};selection=s['protocol']['models'];assert s['protocol']['primary_model_fixed']=='B14_aux_sam'
 source=json.loads((resources.calibration.ROOT/'summary.json').read_text());assert file_hash(resources.calibration.ROOT/'summary.json')==json.loads((resources.calibration.ROOT/'verification.json').read_text())['summary_sha256'];frozen=json.loads((resources.calibration.ROOT/'frozen_selection.json').read_text());assert file_hash(resources.calibration.ROOT/'frozen_selection.json')==source['frozen_selection_sha256']
 for arm in ARMS:assert selection[arm]['old_offset']==frozen['old_offsets'][arm]and selection[arm]['spatial_offset']==frozen['spatial_offsets'][arm]and selection[arm]['checkpoint_sha256']==frozen['models'][arm]['checkpoint_sha256']
 paths=[resources.ROOT/'summary.json',resources.ROOT/'verification.json',resources.ROOT/'manifest.json',resources.ROOT/'image_records.json',resources.ROOT/'frozen_eligibility.json',resources.ROOT/'rgb_review.json',resources.calibration.ROOT/'summary.json',resources.calibration.ROOT/'verification.json',resources.calibration.ROOT/'frozen_selection.json',prior.cohort.META/'instances_val2017.json',prior.cohort.META/'summary.json',prior.WEIGHTS,prior.rgb.prior.resolution.WEIGHTS,Path('semantic/coco_spatial_confirmation.py'),Path('semantic/OVIS_COCO_SPATIAL_CONFIRMATION_PROTOCOL.md'),Path('semantic/coco_semantic_adapter.py'),Path('semantic/external_coco_adapter.py'),Path('semantic/external_label_sensitivity.py'),Path('semantic/dino_native_rgb_control.py'),Path('semantic/dino_resolution_control.py'),Path('semantic/dino_backbone_resource.py'),Path('semantic/semantic_feature_control.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py')]+sorted(prior.VENDOR.rglob('*.py'))+[Path(records[im['id']]['path'])for im in images]
 for path in [prior.WEIGHTS,prior.rgb.prior.resolution.WEIGHTS]+list(prior.VENDOR.rglob('*.py')):assert file_hash(path)==source['protocol']['input_sha256'][str(path)]
 for arm in ARMS:
  for seed in range(3):path=prior.ckpt(arm,seed);assert file_hash(path)==selection[arm]['checkpoint_sha256'][str(seed)];paths.append(path)
 check=metric(np.array([0,1]),np.array([0,0]));assert check['foreground_macro_f1_present']is None and metric(np.array([0]),np.array([-1]))is None
 p={'round':80,'models':selection,'primary_model_fixed':'B14_aux_sam','primary_modes':['spatial_calibration','old_calibration'],'bootstrap_seed':20261080,'bootstrap_repetitions':2000,'query_seed_base':20261077,'queries_per_image':1024,'image_batch':1,'training_updates':0,'calibration_updates':0,'input_sha256':{str(path):file_hash(path)for path in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',{'candidate':images,'metadata_only_reserve_ids':elig['metadata_only_reserve'],'excluded_previously_evaluated_COCO_ids':manifest['excluded_previously_evaluated_COCO_ids']});return images,records,p

def build(images):
 if(ROOT/'coverage.json').exists():return json.loads((ROOT/'coverage.json').read_text())
 api=COCO(str(prior.cohort.META/'instances_val2017.json'));mapping={int(k):int(v)for k,v in json.loads((prior.cohort.META/'summary.json').read_text())['shared_name_mapping'].items()};coverage=[]
 for im in images:
  chosen=np.random.default_rng(20261077+im['id']).choice(16384,1024,replace=False);xy=np.column_stack((chosen%128,chosen//128,np.zeros(1024))).astype(np.int16);maps=semantic_maps(api,im,mapping);package={'xy':xy,'image_id':np.int64(im['id'])}
  for policy,labels in maps.items():
   target=labels[xy[:,1],xy[:,0]];package[policy]=target;coverage.append({'image_id':im['id'],'policy':policy,'valid_query_count':int((target>=0).sum()),'ignored_query_count':int((target<0).sum()),'target_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(target,return_counts=True))}})
  save_npz(qp(im['id']),**package)
 atomic_json(ROOT/'coverage.json',coverage);return coverage

def cache(images,records):
 for bone in ('S14','B14'):
  tasks=[im for im in images if not fp(bone,im['id']).exists()]
  if not tasks:continue
  model=prior.rgb.prior.resolution.load_dino()if bone=='S14'else prior.load_b14();dim=384 if bone=='S14'else 768
  for im in tasks:
   with np.load(qp(im['id']))as a:xy=a['xy'].copy()
   video=prior.rgb.load_native_video({'frames':[records[im['id']]['path']]}).cuda();tokens=prior.rgb.prior.resolution.extract_tokens(model,video,448)if bone=='S14'else prior.tokens_b14(model,video)
   with torch.inference_mode():decoded=patch_at_queries(tokens,xy).half().cpu().numpy()
   assert decoded.shape==(1024,dim)and np.isfinite(decoded).all();save_npz(fp(bone,im['id']),decoded=decoded,xy=xy,image_id=np.int64(im['id']));print('uniform query features',bone,im['id'],flush=True)
  del model;gc.collect();torch.cuda.empty_cache()

def metric(pred,target):
 valid=target>=0;pred=pred[valid];target=target[valid]
 if not len(target):return None
 r=classification_metrics(pred,target,26);bg=target==0;fg=target>0;nonperson=target>1;r.update(valid_query_count=len(target),background_queries=int(bg.sum()),foreground_queries=int(fg.sum()),background_to_foreground_rate=float((pred[bg]!=0).mean())if bg.any()else None,foreground_to_background_rate=float((pred[fg]==0).mean())if fg.any()else None,foreground_class_confusion_rate=float(((pred[fg]>0)&(pred[fg]!=target[fg])).mean())if fg.any()else None,predicted_class_counts=np.bincount(pred,minlength=26).tolist(),nonperson_foreground_conditional=classification_metrics(pred[nonperson],target[nonperson],26)if nonperson.any()else None);return r

def contrast(arm,left,right,results,images,policy,p):
 runs=results[arm]['runs'];pooled=[r['policies'][policy][left]['macro_f1_present']-r['policies'][policy][right]['macro_f1_present']for r in runs];per={}
 for im in images:
  key=str(im['id']);valid=runs[0]['policies'][policy]['per_image'][key][left]is not None;per[key]=float(np.mean([r['policies'][policy]['per_image'][key][left]['macro_f1_present']-r['policies'][policy]['per_image'][key][right]['macro_f1_present']for r in runs]))if valid else None
 x=np.array([v for v in per.values()if v is not None]);draw=np.random.default_rng(p['bootstrap_seed']).integers(0,len(x),(p['bootstrap_repetitions'],len(x)));return {'arm':arm,'left':left,'right':right,'per_seed_pooled_macro_difference':pooled,'mean_pooled_macro_difference':float(np.mean(pooled)),'per_image_three_seed_mean_macro_difference':per,'conditional_image_bootstrap':{'mean_per_image_macro_difference':float(x.mean()),'positive_images':int((x>0).sum()),'negative_images':int((x<0).sum()),'tied_images':int((x==0).sum()),'valid_images':len(x),'percentile_95_interval':np.quantile(x[draw].mean(1),[.025,.975]).tolist(),'note':'conditional image-equal macro difference; not pooled-query macro CI or video/OVIS AP'}}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed spatial calibration confirmation exists')
  torch.backends.cuda.matmul.allow_tf32=False;images,records,p=freeze();coverage=build(images);cache(images,records);bybone={};targets={policy:[]for policy in POLICIES}
  for im in images:
   with np.load(qp(im['id']))as a:
    for policy in POLICIES:targets[policy].append(a[policy].copy())
  targets={key:np.concatenate(value)for key,value in targets.items()};ids=np.repeat([im['id']for im in images],1024).astype(np.int32);results={}
  for bone in ('S14','B14'):
   bybone[bone]=[]
   for im in images:
    with np.load(fp(bone,im['id']))as a:bybone[bone].append(a['decoded'].copy())
  for arm in ARMS:
   settings=p['models'][arm];offsets={'raw':0.,'old_calibration':settings['old_offset'],'spatial_calibration':settings['spatial_offset']};runs=[]
   for seed in range(3):
    state=torch.load(prior.ckpt(arm,seed),map_location='cpu',weights_only=True)['model'];weight=state['classifier.weight'].cuda();bias=state['classifier.bias'].cuda()
    with torch.inference_mode():values=np.concatenate([F.linear(torch.from_numpy(x).float().cuda(),weight,bias).cpu().numpy()for x in bybone[arm.split('_')[0]]])
    path=ROOT/'logits'/f'{arm}_{seed}.npz';save_npz(path,logits=values,image_ids=ids,checkpoint_sha256=np.array(file_hash(prior.ckpt(arm,seed))));policies={}
    for policy,target in targets.items():
     modes={name:predictions(values,offset)for name,offset in offsets.items()};policies[policy]={name:metric(pred,target)for name,pred in modes.items()};policies[policy]['per_image']={str(im['id']):{name:metric(pred[ids==im['id']],target[ids==im['id']])for name,pred in modes.items()}for im in images}
    runs.append({'seed':seed,'policies':policies,'output_path':str(path),'output_sha256':file_hash(path)});print('new COCO spatial calibration',arm,seed,{name:policies['strict'][name]['macro_f1_present']for name in offsets},flush=True)
   results[arm]={'offsets':offsets,'runs':runs}
  contrasts={}
  for policy in POLICIES:
   primary=contrast('B14_aux_sam','spatial_calibration','old_calibration',results,images,policy,p);secondary={}
   for arm in ARMS:
    for right in ('old_calibration','raw'):
     if arm=='B14_aux_sam'and right=='old_calibration':continue
     secondary[arm+'_spatial_minus_'+right]=contrast(arm,'spatial_calibration',right,results,images,policy,p)
   contrasts[policy]={'primary_contrast':primary,'secondary_contrasts_unadjusted':secondary}
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'coverage_sha256':file_hash(ROOT/'coverage.json'),'query_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'queries').glob('*.npz'))},'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'features').rglob('*.npz'))},'results':results,'contrasts':contrasts,'total_coordinates':len(images)*1024,'confirmation_images_fixed_before_evaluation':len(images),'new_training_or_calibration_updates':0,'metadata_reserve40_and_LVVIS40_not_evaluated':True,'deployment_unchanged':'native_gt'});print('DONE fresh COCO spatial calibration confirmation',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
