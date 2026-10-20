"""Frozen original reserve40 confirmation: six arms, both fixed GT policies."""
from collections import defaultdict
import fcntl,gc,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import external_reserve_readiness as ready
from semantic import external_semantic_confirmation as old
from semantic.external_coco_adapter import ANN,label_map,queries
from semantic.external_label_sensitivity import consensus_map
from semantic.dino_backbone_resource import load_b14,tokens_b14
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.background_calibration import predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics
ROOT=Path('semantic_artifacts/external_reserve_confirmation');POLICIES=('strict','consensus');ARMS=ready.ARMS

def qp(policy,group):return ROOT/'queries'/policy/(group+'.npz')
def fp(bone,policy,group):return ROOT/'features'/bone/policy/(group+'.npz')
def verify_inputs(p):
 for name,h in p['input_sha256'].items():assert file_hash(name)==h,name

def freeze():
 prior=json.loads((ready.ROOT/'summary.json').read_text());v=json.loads((ready.ROOT/'verification.json').read_text());assert file_hash(ready.ROOT/'summary.json')==v['summary_sha256']
 eligible=json.loads((ready.ROOT/'frozen_eligibility.json').read_text());clips=json.loads((ready.ROOT/'manifest.json').read_text())['confirmation'];assert [c['group']for c in clips]==eligible['accepted_groups']and len(clips)==40
 p0=prior['protocol'];paths=[ready.ROOT/n for n in ('summary.json','verification.json','manifest.json','frozen_eligibility.json','rgb_review.json','materials.json')]+[ANN,old.MAPPING_SUMMARY,Path('semantic/external_reserve_confirmation.py'),Path('semantic/OVIS_EXTERNAL_RESERVE_CONFIRMATION_PROTOCOL.md'),Path('semantic/external_coco_adapter.py'),Path('semantic/external_label_sensitivity.py'),Path('semantic/dino_backbone_resource.py'),Path('semantic/dino_resolution_control.py'),Path('semantic/semantic_feature_control.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py')]
 inputs=dict(p0['input_sha256']);inputs.update({str(x):file_hash(x)for x in paths})
 p={'round':93,'models':p0['models'],'statistics':p0['statistics'],'primary':p0['primary'],'secondary':p0['secondary'],'policies':list(POLICIES),'primary_policy':'strict','query_seed_base':20261041,'queries_per_nonempty_frame':256,'bootstrap_seed':20261093,'bootstrap_repetitions':2000,'training_updates':0,'calibration_updates':0,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'input_sha256':inputs}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 verify_inputs(p);atomic_json(ROOT/'manifest.json',{'confirmation':clips});return clips,p

def build_queries(clips):
 if(ROOT/'coverage.json').exists():return
 data=json.loads(ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(old.MAPPING_SUMMARY.read_text())['shared_category_mapping'].items()};by=defaultdict(list)
 for a in data['annotations']:by[a['image_id']].append(a)
 coverage=[]
 for clip in clips:
  points={k:[]for k in POLICIES};labels={k:[]for k in POLICIES}
  for slot,im in enumerate(clip['frames']):
   for policy,fn in [('strict',label_map),('consensus',consensus_map)]:
    raster=fn(by[im['id']],mapping,im['height'],im['width']);xy,y=queries(raster,im['id'],slot);points[policy].append(xy);labels[policy].append(y)
    coverage.append({'group':clip['group'],'time':slot,'image_id':im['id'],'policy':policy,'query_count':len(y),'distinct_pixels':len(np.unique(xy[:,:2],axis=0)),'foreground_pixels':int((raster>0).sum()),'ignored_pixels':int((raster<0).sum()),'support':{str(int(k)):int(v)for k,v in zip(*np.unique(y,return_counts=True))}})
  for policy in POLICIES:save_npz(qp(policy,clip['group']),xy=np.concatenate(points[policy]).astype(np.int64),labels=np.concatenate(labels[policy]).astype(np.int64))
 atomic_json(ROOT/'coverage.json',coverage);print('queries frozen',len(clips),flush=True)

def cache(clips):
 for bone in ('S14','B14'):
  tasks=[c for c in clips if not all(fp(bone,p,c['group']).exists()for p in POLICIES)]
  if not tasks:continue
  model=old.rgb.prior.resolution.load_dino()if bone=='S14'else load_b14()
  for clip in tasks:
   points={}
   for policy in POLICIES:
    with np.load(qp(policy,clip['group']))as a:points[policy]=a['xy'].copy()
   if any(len(x)for x in points.values()):
    video=old.rgb.load_native_video({'frames':[str(old.RGB/'images'/im['file_name'])for im in clip['frames']]}).cuda()
    tokens=old.rgb.prior.resolution.extract_tokens(model,video,448)if bone=='S14'else tokens_b14(model,video)
    with torch.inference_mode():values={p:patch_at_queries(tokens,xy).half().cpu().numpy()if len(xy)else np.empty((0,384 if bone=='S14'else 768),np.float16)for p,xy in points.items()}
    del video,tokens
   else:values={p:np.empty((0,384 if bone=='S14'else 768),np.float16)for p in POLICIES}
   for policy,x in values.items():
    assert np.isfinite(x).all();save_npz(fp(bone,policy,clip['group']),decoded=x,xy=points[policy])
   print('cached',bone,clip['group'],{p:len(x)for p,x in points.items()},flush=True)
  del model;gc.collect();torch.cuda.empty_cache()

def evaluate(clips,p):
 with np.load(p['statistics']['path'])as a:mu=a['mean'].copy();scale=a['scale'].copy()
 results={}
 for policy in POLICIES:
  targets=[];group_ids=[];xs={b:[]for b in ('S14','B14')}
  for clip in clips:
   with np.load(qp(policy,clip['group']))as a:y=a['labels'].copy();targets.append(y);group_ids.extend([int(clip['group'])]*len(y))
   for bone in xs:
    with np.load(fp(bone,policy,clip['group']))as a:xs[bone].append(a['decoded'].copy())
  y=np.concatenate(targets);ids=np.array(group_ids,np.int64)
  if not len(y):results[policy]={'no_evaluable_queries':True};continue
  policy_results={}
  for arm in ARMS:
   bone='B14'if arm.startswith('B14')else'S14';runs=[]
   for seed in range(3):
    path=ready.ckpt(arm,seed);state=torch.load(path,map_location='cpu',weights_only=True)['model'];w=state['classifier.weight'].cuda();b=state['classifier.bias'].cuda();pieces=[]
    with torch.inference_mode():
     for x in xs[bone]:
      x=x.astype(np.float32)
      if arm=='B14_z':x=(x-mu)/scale
      pieces.append(F.linear(torch.from_numpy(x).cuda(),w,b).cpu().numpy())
    logits=np.concatenate(pieces);out=ROOT/'logits'/policy/(arm+f'_{seed}.npz');save_npz(out,logits=logits,labels=y,group_ids=ids)
    modes={}
    for mode,offset in [('raw',0),('calibrated',p['models'][arm]['offset'])]:
     pred=predictions(logits,offset);modes[mode]={'pooled':classification_metrics(pred,y,26),'per_video':{c['group']:classification_metrics(pred[ids==int(c['group'])],y[ids==int(c['group'])],26)if np.any(ids==int(c['group']))else None for c in clips},'confusion':np.bincount(y*26+pred,minlength=26*26).reshape(26,26).tolist()}
    runs.append({'seed':seed,'metrics':modes,'output_path':str(out),'output_sha256':file_hash(out)});print('evaluated',policy,arm,seed,modes['calibrated']['pooled']['macro_f1_present'],flush=True)
   policy_results[arm]={'runs':runs}
  # Reuse the historical contrast implementation with a pure data-shape adapter.
  adapted={a:{'runs':[{'calibrated':r['metrics']['calibrated']['pooled'],'per_video':r['metrics']['calibrated']['per_video']}for r in val['runs']]}for a,val in policy_results.items()}
  results[policy]={'arms':policy_results,'primary_contrast':old.contrast(*p['primary'],adapted,clips,p),'secondary_contrasts_unadjusted':{a+'_minus_'+b:old.contrast(a,b,adapted,clips,p)for a,b in p['secondary']},'total_queries':len(y),'query_bearing_groups':int(sum(bool(len(t))for t in targets)),'empty_groups':[c['group']for c,t in zip(clips,targets)if not len(t)]}
 return results

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed reserve confirmation: inspect, do not repeat')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  clips,p=freeze();build_queries(clips);cache(clips);results=evaluate(clips,p);verify_inputs(p)
  hashes={str(x):file_hash(x)for folder in ('queries','features','logits')for x in sorted((ROOT/folder).rglob('*.npz'))}
  atomic_json(ROOT/'summary.json',{'protocol':p,'results':results,'output_hashes':hashes,'coverage_sha256':file_hash(ROOT/'coverage.json'),'total_candidate_groups':40,'total_frames':320,'deployment_unchanged':'native_gt','limits':['GT-conditioned mapped foreground only','directory group not verified track identity','metadata and RGB previously observed','SAM/STCN-assisted annotations and backbone pretraining overlap unknown']})
  print('DONE original reserve confirmation',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
