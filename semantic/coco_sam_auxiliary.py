"""Prespecified frozen SAM auxiliary confirmation on original COCO reserve80."""
from collections import defaultdict
import fcntl,gc,importlib.metadata,inspect,json,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import __version__ as pillow_version
from pycocotools.coco import COCO
from semantic import coco_cohort_segments as cohort
from semantic import dino_b14_auxiliary as auxiliary
from semantic import dino_native_rgb_control as rgb
from semantic.dino_backbone_resource import load_b14,tokens_b14,WEIGHTS,VENDOR
from semantic.coco_semantic_adapter import semantic_maps
from semantic.external_nonperson_confirmation import nonperson_queries
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.background_calibration import predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics
ROOT=Path('semantic_artifacts/coco_sam_auxiliary');PANEL=cohort.ROOT;POLICIES=('strict','consensus');ARMS=('B14_aux_sam','B14_extra_gt','B14_gt32','B14_weak','B14_gt256','B14_class_video','S14_gt32')

def ckpt(arm,seed):
 if arm in ('B14_aux_sam','B14_extra_gt','B14_gt32','B14_weak'):return auxiliary.ckpt({'B14_aux_sam':'aux_sam','B14_extra_gt':'extra_gt','B14_gt32':'gt32','B14_weak':'weak'}[arm],seed)
 return cohort.ckpt(arm,seed)

def qp(policy,im):return ROOT/'queries'/policy/f'{im}.npz'
def fp(bone,policy,im):return ROOT/'features'/bone/policy/f'{im}.npz'
def freeze():
 v=json.loads((PANEL/'verification.json').read_text());assert file_hash(PANEL/'summary.json')==v['summary_sha256'];elig=json.loads((PANEL/'frozen_eligibility.json').read_text());manifest=json.loads((PANEL/'manifest.json').read_text());images=[im for im in manifest['reserve']if im['id']in elig['reserve']];assert len(images)==80 and [im['id']for im in images]==elig['reserve']and not set(elig['candidate'])&set(elig['reserve'])
 sv=json.loads((auxiliary.ROOT/'verification.json').read_text());assert file_hash(auxiliary.ROOT/'summary.json')==sv['summary_sha256'];selected=json.loads((auxiliary.ROOT/'frozen_selection.json').read_text());assert selected['validation_selected_teacher_recipe']=='aux_sam'and selected['teacher_matched_GT_reference']=='extra_gt'and selected['oracle_excluded_from_selection']
 oldroot=Path('semantic_artifacts/coco_semantic_confirmation');old=json.loads((oldroot/'summary.json').read_text());assert file_hash(oldroot/'summary.json')==json.loads((oldroot/'verification.json').read_text())['summary_sha256']and old['reserved80_not_evaluated']
 records={r['image_id']:r for r in json.loads((PANEL/'image_records.json').read_text())};selection={}
 for arm in ARMS:
  source=selected['offsets'][{'B14_aux_sam':'aux_sam','B14_extra_gt':'extra_gt','B14_gt32':'gt32','B14_weak':'weak'}[arm]]if arm in ('B14_aux_sam','B14_extra_gt','B14_gt32','B14_weak')else old['protocol']['models'][arm]['offset'];selection[arm]={'offset':source,'checkpoint_sha256':{str(seed):file_hash(ckpt(arm,seed))for seed in range(3)}}
 assert {a:selection[a]['offset']for a in ARMS}=={'B14_aux_sam':-2.25,'B14_extra_gt':-2.25,'B14_gt32':-2.0,'B14_weak':-2.75,'B14_gt256':-2.0,'B14_class_video':-2.0,'S14_gt32':-1.25}
 for bonepath,h in [(WEIGHTS,'0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'),(rgb.prior.resolution.WEIGHTS,'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9')]:assert file_hash(bonepath)==h
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()=='7764ea0f912e53c92e82eb78a2a1631e92725fc8';subprocess.run(['git','diff','--quiet','HEAD'],cwd=VENDOR,check=True)
 paths=[PANEL/'summary.json',PANEL/'verification.json',PANEL/'frozen_eligibility.json',PANEL/'manifest.json',PANEL/'image_records.json',cohort.META/'instances_val2017.json',cohort.META/'summary.json',auxiliary.ROOT/'summary.json',auxiliary.ROOT/'verification.json',auxiliary.ROOT/'frozen_selection.json',oldroot/'summary.json',oldroot/'verification.json',WEIGHTS,rgb.prior.resolution.WEIGHTS,Path('semantic/coco_sam_auxiliary.py'),Path('semantic/OVIS_COCO_SAM_AUXILIARY_PROTOCOL.md'),Path('semantic/coco_semantic_adapter.py'),Path('semantic/external_nonperson_confirmation.py'),Path('semantic/external_coco_adapter.py'),Path('semantic/external_label_sensitivity.py'),Path('semantic/dino_native_rgb_control.py'),Path('semantic/dino_resolution_control.py'),Path('semantic/dino_backbone_resource.py'),Path('semantic/semantic_feature_control.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py'),Path(inspect.getfile(COCO))]+sorted(VENDOR.rglob('*.py'))+[Path(records[im['id']]['path'])for im in images]
 for arm in ARMS:
  for seed in range(3):paths.append(ckpt(arm,seed))
 pre=json.loads(Path('semantic_artifacts/dino_backbone_recovery/resource_preflight.json').read_text());assert (torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))==(pre['torch_version'],pre['cuda_version'],pre['device'])
 p={'round':72,'static_image_source_only':True,'cohort_role':'original_frozen_COCO_reserve80','models':selection,'source_validation_selected_teacher':'B14_aux_sam','oracle_excluded':True,'primary':['B14_aux_sam','B14_extra_gt'],'secondary':[['B14_aux_sam','B14_gt32'],['B14_weak','B14_gt256'],['B14_aux_sam','B14_weak'],['B14_class_video','S14_gt32']],'bootstrap_seed':20261072,'bootstrap_repetitions':2000,'query_seed_base':20261052,'per_image_per_class_queries':64,'image_batch':1,'training_updates':0,'calibration_updates':0,'torch':torch.__version__,'pillow':pillow_version,'pycocotools':importlib.metadata.version('pycocotools'),'input_sha256':{str(x):file_hash(x)for x in paths}}
 for name,data in [('protocol.json',p),('manifest.json',{'candidate':images,'previously_evaluated_candidate_image_ids':elig['candidate'],'reserved_image_ids':[],'LVVIS40_unscored':True})]:
  path=ROOT/name
  if path.exists():assert json.loads(path.read_text())==data
  else:atomic_json(path,data)
 return images,records,p

def build_queries(images):
 if(ROOT/'query_coverage.json').exists():return json.loads((ROOT/'query_coverage.json').read_text())
 api=COCO(str(cohort.META/'instances_val2017.json'));mapping={int(k):int(v)for k,v in json.loads((cohort.META/'summary.json').read_text())['shared_name_mapping'].items()};coverage=[]
 for im in images:
  maps=semantic_maps(api,im,mapping);strict=maps['strict'];assert np.array_equal(strict[strict>0],maps['consensus'][strict>0])
  for policy,labels in maps.items():
   xy,y=nonperson_queries(labels,im['id'],0);save_npz(qp(policy,im['id']),xy=xy,labels=y,image_id=np.int64(im['id']));coverage.append({'policy':policy,'image_id':im['id'],'nonperson_pixels':int((labels>1).sum()),'ignored_pixels':int((labels<0).sum()),'query_count':len(y),'distinct_query_pixels':len(np.unique(xy[:,:2],axis=0)),'query_class_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(y,return_counts=True))}})
 atomic_json(ROOT/'query_coverage.json',coverage);return coverage

def cache(images,records):
 for bone in ('S14','B14'):
  tasks=[im for im in images if any(not fp(bone,policy,im['id']).exists()for policy in POLICIES)]
  if not tasks:continue
  model=rgb.prior.resolution.load_dino()if bone=='S14'else load_b14();dim=384 if bone=='S14'else 768
  for im in tasks:
   metas={}
   for policy in POLICIES:
    with np.load(qp(policy,im['id']))as a:metas[policy]={k:a[k].copy()for k in a.files}
   points=np.concatenate([metas[a]['xy']for a in POLICIES])
   if len(points):
    video=rgb.load_native_video({'frames':[records[im['id']]['path']]}).cuda();tokens=rgb.prior.resolution.extract_tokens(model,video,448)if bone=='S14'else tokens_b14(model,video)
    with torch.inference_mode():features=patch_at_queries(tokens,points).half().cpu().numpy()
    assert features.shape==(len(points),dim)and np.isfinite(features).all();del video,tokens
   else:features=np.empty((0,dim),np.float16)
   offset=0
   for policy in POLICIES:
    n=len(metas[policy]['labels']);package=dict(metas[policy],decoded=features[offset:offset+n]);offset+=n;path=fp(bone,policy,im['id'])
    if path.exists():
     with np.load(path)as a:assert all(np.array_equal(a[k],value)for k,value in package.items())
    else:save_npz(path,**package)
   print('COCO features',bone,im['id'],len(points),flush=True)
  del model;gc.collect();torch.cuda.empty_cache()

def head(arm,seed,policy,packages):
 path=ROOT/policy/arm/'logits'/f'seed{seed}.npz';h=file_hash(ckpt(arm,seed))
 if path.exists():
  with np.load(path)as a:out={k:a[k].copy()for k in a.files}
  assert str(out['checkpoint_sha256'])==h;return out
 state=torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model'];weight=state['classifier.weight'].cuda();bias=state['classifier.bias'].cuda()
 with torch.inference_mode():values=[torch.nn.functional.linear(torch.from_numpy(a['decoded']).float().cuda(),weight,bias).cpu().numpy()for a in packages]
 out={'logits':np.concatenate(values),'labels':np.concatenate([a['labels']for a in packages]),'image_ids':np.concatenate([np.full(len(a['labels']),int(a['image_id']),np.int32)for a in packages]),'checkpoint_sha256':np.array(h)};save_npz(path,**out);return out

def contrast(left,right,results,images,p):
 pooled=[results[left]['runs'][i]['calibrated']['macro_f1_present']-results[right]['runs'][i]['calibrated']['macro_f1_present']for i in range(3)];per={}
 for im in images:
  key=str(im['id']);a=results[left]['runs'][0]['per_image'][key]
  per[key]=float(np.mean([results[left]['runs'][i]['per_image'][key]['macro_f1_present']-results[right]['runs'][i]['per_image'][key]['macro_f1_present']for i in range(3)]))if a is not None else None
 x=np.array([v for v in per.values()if v is not None]);draw=np.random.default_rng(p['bootstrap_seed']).integers(0,len(x),(p['bootstrap_repetitions'],len(x)));ci=np.quantile(x[draw].mean(1),[.025,.975]).tolist()
 return {'left':left,'right':right,'per_seed_pooled_macro_difference':pooled,'mean_pooled_macro_difference':float(np.mean(pooled)),'per_image_three_seed_mean_differences':per,'conditional_image_bootstrap':{'mean_per_image_difference':float(x.mean()),'positive_images':int((x>0).sum()),'negative_images':int((x<0).sum()),'tied_images':int((x==0).sum()),'image_count':len(x),'percentile_95_interval':ci,'note':'conditional image-equal difference, not pooled-query macro interval or temporal-video evidence'}}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed COCO confirmation exists')
  torch.backends.cuda.matmul.allow_tf32=False;images,records,p=freeze();coverage=build_queries(images);cache(images,records);allresults={}
  for policy in POLICIES:
   bybone={}
   for bone in ('S14','B14'):
    bybone[bone]=[]
    for im in images:
     with np.load(fp(bone,policy,im['id']))as a:bybone[bone].append({k:a[k].copy()for k in a.files})
   results={}
   for arm in ARMS:
    packages=bybone[arm.split('_')[0]];offset=p['models'][arm]['offset'];runs=[]
    for seed in range(3):
     out=head(arm,seed,policy,packages);pred=predictions(out['logits'],offset);per={}
     for im in images:
      rows=out['image_ids']==im['id'];per[str(im['id'])]=classification_metrics(pred[rows],out['labels'][rows],26)if rows.any()else None
     runs.append({'seed':seed,'raw':classification_metrics(predictions(out['logits'],0),out['labels'],26),'calibrated':classification_metrics(pred,out['labels'],26),'per_image':per});print('COCO scores',policy,arm,seed,runs[-1]['calibrated']['macro_f1_present'],flush=True)
    results[arm]={'offset':offset,'runs':runs}
   labels=np.concatenate([a['labels']for a in packages]);counts={str(im['id']):len(a['labels'])for im,a in zip(images,packages)};rows=[r for r in coverage if r['policy']==policy]
   allresults[policy]={'results':results,'primary_contrast':contrast(*p['primary'],results,images,p),'secondary_contrasts_unadjusted':{a+'_minus_'+b:contrast(a,b,results,images,p)for a,b in p['secondary']},'actual_support':{str(int(c)):int(n)for c,n in zip(*np.unique(labels,return_counts=True))},'total_queries':len(labels),'query_counts':counts,'empty_images':[i for i,n in counts.items()if not n],'distinct_image_query_pixels':sum(r['distinct_query_pixels']for r in rows)}
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  manifest=json.loads((ROOT/'manifest.json').read_text());observed={int(path.stem)for path in (ROOT/'features').rglob('*.npz')};assert observed=={im['id']for im in images}and not observed&set(manifest['previously_evaluated_candidate_image_ids'])
  atomic_json(ROOT/'summary.json',{'protocol':p,'policies':allresults,'query_coverage_sha256':file_hash(ROOT/'query_coverage.json'),'query_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'queries').rglob('*.npz'))},'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'features').rglob('*.npz'))},'original_COCO_reserve80_evaluated':True,'LVVIS40_not_evaluated':True,'new_training_or_calibration':0,'deployment_unchanged':'native_gt'});print('DONE COCO reserved SAM auxiliary confirmation',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
