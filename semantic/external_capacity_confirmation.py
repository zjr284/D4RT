"""Fresh metadata-enriched external frozen S14/B14 confirmation, two GT policies."""
from collections import defaultdict
import fcntl,gc,json
from pathlib import Path
import numpy as np
import torch
from semantic import dino_native_rgb_control as rgb
from semantic.dino_backbone_resource import load_b14,tokens_b14,WEIGHTS
from semantic.dino_backbone_training import ARMS,ckpt
from semantic.external_coco_adapter import ANN,label_map
from semantic.external_label_sensitivity import consensus_map
from semantic.external_nonperson_confirmation import nonperson_queries
from semantic.external_semantic_confirmation import contrast
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.background_calibration import predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics
ROOT=Path('semantic_artifacts/external_capacity_confirmation')
PANEL=Path('semantic_artifacts/external_capacity_panel')
SOURCE=Path('semantic_artifacts/dino_backbone_training')
MAPPING=Path('semantic_artifacts/external_coco_adapter/summary.json')
POLICIES=('strict','consensus')
def qp(policy,group):return ROOT/'queries'/policy/f'{group}.npz'
def fp(bone,policy,group):return ROOT/'features'/bone/policy/f'{group}.npz'
def freeze():
 panel=json.loads((PANEL/'summary.json').read_text());v=json.loads((PANEL/'verification.json').read_text());assert file_hash(PANEL/'summary.json')==v['summary_sha256']
 frozen=json.loads((SOURCE/'frozen_selection.json').read_text());assert frozen==panel['protocol']['selection']and frozen['validation_selected_arm']=='B14_gt32'
 eligible=json.loads((PANEL/'frozen_eligibility.json').read_text());manifest=json.loads((PANEL/'manifest.json').read_text());clips=[c for c in manifest['panel']if c['group']in eligible['accepted_groups']];assert [c['group']for c in clips]==eligible['accepted_groups']and len(clips)==40
 inputs=dict(panel['protocol']['input_sha256']);paths=[PANEL/'summary.json',PANEL/'verification.json',PANEL/'frozen_eligibility.json',PANEL/'manifest.json',PANEL/'image_records.json',SOURCE/'verification.json',MAPPING,Path('semantic/external_capacity_confirmation.py'),Path('semantic/OVIS_EXTERNAL_CAPACITY_CONFIRMATION_PROTOCOL.md'),Path('semantic/external_label_sensitivity.py'),Path('semantic/external_nonperson_confirmation.py'),Path('semantic/external_coco_adapter.py'),Path('semantic/dino_backbone_resource.py')]
 for c in clips:paths.extend(PANEL/'images'/im['file_name']for im in c['frames'])
 inputs.update({str(x):file_hash(x)for x in paths})
 for path,h in inputs.items():assert file_hash(path)==h
 pre=json.loads(Path('semantic_artifacts/dino_backbone_recovery/resource_preflight.json').read_text());assert (torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))==(pre['torch_version'],pre['cuda_version'],pre['device'])
 p={'round':60,'selection':frozen,'policies':list(POLICIES),'primary':['B14_gt32','S14_gt32'],'secondary':[['B14_gt256','S14_gt256'],['B14_gt32','S14_gt256']],'bootstrap_seed':20261060,'bootstrap_repetitions':2000,'query_seed_base':20261052,'per_class_per_frame_queries':64,'training_updates':0,'calibration_updates':0,'input_sha256':inputs}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',{'confirmation':clips});return clips,p

def queries(clips):
 if(ROOT/'query_coverage.json').exists():return json.loads((ROOT/'query_coverage.json').read_text())
 payload=json.loads(ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(MAPPING.read_text())['shared_category_mapping'].items()};by=defaultdict(list)
 for a in payload['annotations']:by[a['image_id']].append(a)
 coverage=[]
 for clip in clips:
  points={a:[]for a in POLICIES};targets={a:[]for a in POLICIES}
  for slot,im in enumerate(clip['frames']):
   strict=label_map(by[im['id']],mapping,im['height'],im['width']);new=consensus_map(by[im['id']],mapping,im['height'],im['width']);assert np.array_equal(strict[strict>0],new[strict>0])
   for policy,labels in [('strict',strict),('consensus',new)]:
    xy,y=nonperson_queries(labels,im['id'],slot);points[policy].append(xy);targets[policy].append(y);coverage.append({'policy':policy,'group':clip['group'],'time':slot,'image_id':im['id'],'nonperson_pixels':int((labels>1).sum()),'ignored_pixels':int((labels<0).sum()),'query_count':len(y),'distinct_query_pixels':len(np.unique(xy[:,:2],axis=0)),'query_class_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(y,return_counts=True))}})
  for policy in POLICIES:save_npz(qp(policy,clip['group']),xy=np.concatenate(points[policy]),labels=np.concatenate(targets[policy]),video_id=np.int64(int(clip['group'])))
 atomic_json(ROOT/'query_coverage.json',coverage);return coverage

def cache(clips):
 for bone in ('S14','B14'):
  tasks=[c for c in clips if any(not fp(bone,policy,c['group']).exists()for policy in POLICIES)]
  if not tasks:continue
  model=rgb.prior.resolution.load_dino()if bone=='S14'else load_b14();dimension=384 if bone=='S14'else 768
  for clip in tasks:
   meta={}
   for policy in POLICIES:
    with np.load(qp(policy,clip['group']))as a:meta[policy]={k:a[k].copy()for k in a.files}
   joined=np.concatenate([meta[a]['xy']for a in POLICIES])
   if len(joined):
    video=rgb.load_native_video({'frames':[str(PANEL/'images'/im['file_name'])for im in clip['frames']]}).cuda();tokens=rgb.prior.resolution.extract_tokens(model,video,448)if bone=='S14'else tokens_b14(model,video)
    with torch.inference_mode():features=patch_at_queries(tokens,joined).half().cpu().numpy()
    assert features.shape==(len(joined),dimension)and np.isfinite(features).all();del video,tokens
   else:features=np.empty((0,dimension),np.float16)
   start=0
   for policy in POLICIES:
    n=len(meta[policy]['labels']);path=fp(bone,policy,clip['group']);package=dict(meta[policy],decoded=features[start:start+n]);start+=n
    if path.exists():
     with np.load(path)as a:assert all(np.array_equal(a[k],value)for k,value in package.items())
    else:save_npz(path,**package)
   print('external capacity features',bone,clip['group'],len(joined),flush=True)
  del model;gc.collect();torch.cuda.empty_cache()

def head(arm,seed,policy,packages):
 path=ROOT/policy/arm/'logits'/f'dinov2_{seed}_confirmation.npz';h=file_hash(ckpt(arm,seed))
 if path.exists():
  with np.load(path)as a:out={k:a[k].copy()for k in a.files}
  assert str(out['checkpoint_sha256'])==h;return out
 state=torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model'];weight=state['classifier.weight'].cuda();bias=state['classifier.bias'].cuda()
 with torch.inference_mode():values=[torch.nn.functional.linear(torch.from_numpy(c['decoded']).float().cuda(),weight,bias).cpu().numpy()for c in packages]
 out={'logits':np.concatenate(values),'labels':np.concatenate([c['labels']for c in packages]),'times':np.concatenate([c['xy'][:,2]for c in packages]),'video_ids':np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int32)for c in packages]),'checkpoint_sha256':np.array(h)};save_npz(path,**out);return out

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed external capacity confirmation exists')
  torch.backends.cuda.matmul.allow_tf32=False;clips,p=freeze();coverage=queries(clips);cache(clips);allresults={}
  for policy in POLICIES:
   results={};bybone={}
   for bone in ('S14','B14'):
    bybone[bone]=[]
    for c in clips:
     with np.load(fp(bone,policy,c['group']))as a:bybone[bone].append({k:a[k].copy()for k in a.files})
   for arm in ARMS:
    packages=bybone[arm.split('_')[0]];runs=[];offset=p['selection']['offsets'][arm]
    for seed in range(3):
     out=head(arm,seed,policy,packages);pred=predictions(out['logits'],offset);per={}
     for c in clips:
      rows=out['video_ids']==int(c['group']);per[c['group']]=classification_metrics(pred[rows],out['labels'][rows],26)if rows.any()else None
     runs.append({'seed':seed,'raw':classification_metrics(predictions(out['logits'],0),out['labels'],26),'calibrated':classification_metrics(pred,out['labels'],26),'per_video':per});print('external capacity scores',policy,arm,seed,runs[-1]['calibrated']['macro_f1_present'],flush=True)
    results[arm]={'offset':offset,'runs':runs}
   target=np.concatenate([c['labels']for c in packages]);counts={c['group']:len(a['labels'])for c,a in zip(clips,packages)};rows=[r for r in coverage if r['policy']==policy]
   allresults[policy]={'results':results,'primary_contrast':contrast(*p['primary'],results,clips,p),'secondary_contrasts_unadjusted':{a+'_minus_'+b:contrast(a,b,results,clips,p)for a,b in p['secondary']},'actual_support':{str(int(c)):int(n)for c,n in zip(*np.unique(target,return_counts=True))},'total_queries':len(target),'query_counts':counts,'empty_groups':[g for g,n in counts.items()if not n],'empty_frames':sum(not r['query_count']for r in rows),'distinct_frame_query_pixels':sum(r['distinct_query_pixels']for r in rows)}
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'policies':allresults,'query_coverage_sha256':file_hash(ROOT/'query_coverage.json'),'query_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'queries').rglob('*.npz'))},'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'features').rglob('*.npz'))},'model_or_calibration_updates':0,'old40_reserve_evaluated':False,'deployment_unchanged':'native_gt'})
  print('DONE fresh external capacity confirmation',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
