"""Frozen existing-image uniform-query diagnostics, not fresh confirmation."""
from collections import defaultdict
import fcntl,gc,json
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from semantic import coco_sam_auxiliary as prior
from semantic import dino_b14_multilayer as latest
from semantic.coco_semantic_adapter import semantic_maps
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.background_calibration import predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics
ROOT=Path('semantic_artifacts/coco_uniform_queries');ARMS=prior.ARMS;POLICIES=('strict','consensus')
def qp(im):return ROOT/'queries'/f'{im}.npz'
def fp(bone,im):return ROOT/'features'/bone/f'{im}.npz'

def freeze():
 old=json.loads((prior.ROOT/'summary.json').read_text());assert file_hash(prior.ROOT/'summary.json')==json.loads((prior.ROOT/'verification.json').read_text())['summary_sha256'];assert file_hash(latest.ROOT/'summary.json')==json.loads((latest.ROOT/'verification.json').read_text())['summary_sha256'];m=json.loads((prior.PANEL/'manifest.json').read_text());eligible=json.loads((prior.PANEL/'frozen_eligibility.json').read_text());images=m['candidate'];assert [im['id']for im in images]==eligible['candidate']and len(images)==80;records={r['image_id']:r for r in json.loads((prior.PANEL/'image_records.json').read_text())};selection=old['protocol']['models'];paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',latest.ROOT/'summary.json',latest.ROOT/'verification.json',prior.PANEL/'manifest.json',prior.PANEL/'frozen_eligibility.json',prior.PANEL/'image_records.json',prior.cohort.META/'instances_val2017.json',prior.cohort.META/'summary.json',prior.WEIGHTS,prior.rgb.prior.resolution.WEIGHTS,Path('semantic/coco_uniform_queries.py'),Path('semantic/OVIS_COCO_UNIFORM_QUERY_PROTOCOL.md')]+[Path(k)for k in old['protocol']['input_sha256']if k.startswith('semantic/')and k.endswith('.py')]+sorted(prior.VENDOR.rglob('*.py'))+[Path(records[im['id']]['path'])for im in images]
 for arm in ARMS:
  for seed in range(3):path=prior.ckpt(arm,seed);assert file_hash(path)==selection[arm]['checkpoint_sha256'][str(seed)];paths.append(path)
 for path in [prior.WEIGHTS,prior.rgb.prior.resolution.WEIGHTS]+list(prior.VENDOR.rglob('*.py')):assert file_hash(path)==old['protocol']['input_sha256'][str(path)]
 p={'round':77,'analysis':'descriptive on already observed candidate80, not new confirmation','models':selection,'query_seed_base':20261077,'queries_per_image':1024,'uniform_without_replacement':True,'query_coordinates_independent_of_GT':True,'policies':list(POLICIES),'include_background_and_person':True,'training_updates':0,'calibration_updates':0,'input_sha256':{str(path):file_hash(path)for path in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',{'candidate':images,'LVVIS40_unscored':True});return images,records,p

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

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed uniform-query diagnostic exists')
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
   offset=p['models'][arm]['offset'];runs=[]
   for seed in range(3):
    state=torch.load(prior.ckpt(arm,seed),map_location='cpu',weights_only=True)['model'];weight=state['classifier.weight'].cuda();bias=state['classifier.bias'].cuda()
    with torch.inference_mode():values=np.concatenate([F.linear(torch.from_numpy(x).float().cuda(),weight,bias).cpu().numpy()for x in bybone[arm.split('_')[0]]])
    path=ROOT/'logits'/f'{arm}_{seed}.npz';save_npz(path,logits=values,image_ids=ids,checkpoint_sha256=np.array(file_hash(prior.ckpt(arm,seed))));raw=predictions(values,0);cal=predictions(values,offset);policies={}
    for policy,target in targets.items():policies[policy]={'raw':metric(raw,target),'calibrated':metric(cal,target),'per_image':{str(im['id']):{'raw':metric(raw[ids==im['id']],target[ids==im['id']]),'calibrated':metric(cal[ids==im['id']],target[ids==im['id']])}for im in images}}
    runs.append({'seed':seed,'policies':policies,'output_path':str(path),'output_sha256':file_hash(path)});print('uniform diagnostic',arm,seed,'strict acc',policies['strict']['calibrated']['accuracy'],'BGtoFG',policies['strict']['calibrated']['background_to_foreground_rate'],flush=True)
   results[arm]={'offset':offset,'runs':runs}
  contrasts={}
  for left,right in [('B14_class_video','S14_gt32'),('B14_aux_sam','B14_extra_gt'),('B14_aux_sam','B14_gt32'),('B14_weak','B14_gt256')]:
   contrasts[left+'_minus_'+right]={policy:{key:[results[left]['runs'][seed]['policies'][policy]['calibrated'][key]-results[right]['runs'][seed]['policies'][policy]['calibrated'][key]for seed in range(3)]for key in ('accuracy','macro_f1_present','foreground_macro_f1_present','background_to_foreground_rate','foreground_to_background_rate')}for policy in POLICIES}
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'coverage_sha256':file_hash(ROOT/'coverage.json'),'query_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'queries').glob('*.npz'))},'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'features').rglob('*.npz'))},'results':results,'descriptive_contrasts':contrasts,'total_coordinates':81920,'new_confirmation_groups':0,'LVVIS40_not_evaluated':True,'deployment_unchanged':'native_gt'});print('DONE existing-image uniform query diagnostic',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
