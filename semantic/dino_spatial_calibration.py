"""Validation-only calibration from GT-independent uniform source-image queries."""
import fcntl,gc,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import coco_uniform_queries as prior
from semantic import dino_backbone_training as capacity
from semantic.ytvis import load_dataset
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.background_calibration import evaluate,pick_offset
from semantic.run import atomic_json,classification_metrics
from semantic.oracle_probe import file_hash
ROOT=Path('semantic_artifacts/dino_spatial_calibration');ARMS=prior.ARMS;ANNOTATIONS=Path('semantic_artifacts/ovis/annotations_train.json')
def qp(vid):return ROOT/'queries'/f'{vid}.npz'
def fp(bone,vid):return ROOT/'features'/bone/f'{vid}.npz'

def freeze():
 old=json.loads((prior.ROOT/'summary.json').read_text());assert file_hash(prior.ROOT/'summary.json')==json.loads((prior.ROOT/'verification.json').read_text())['summary_sha256'];m=json.loads((capacity.ROOT/'manifest.json').read_text());clips=m['val'];assert len(clips)==20 and clips[0]['video_id']==258;models=old['protocol']['models'];paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',capacity.ROOT/'manifest.json',capacity.ROOT/'summary.json',capacity.ROOT/'verification.json',ANNOTATIONS,Path('semantic/dino_spatial_calibration.py'),Path('semantic/OVIS_DINO_SPATIAL_CALIBRATION_PROTOCOL.md'),Path('semantic/ytvis.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py'),prior.prior.WEIGHTS,prior.prior.rgb.prior.resolution.WEIGHTS]+sorted(prior.prior.VENDOR.rglob('*.py'))
 paths.extend(Path(k)for k in old['protocol']['input_sha256']if k.startswith('semantic/')and k.endswith('.py'))
 for c in clips:paths.extend(Path(frame)for frame in c['frames'])
 for arm in ARMS:
  for seed in range(3):path=prior.prior.ckpt(arm,seed);assert file_hash(path)==models[arm]['checkpoint_sha256'][str(seed)];paths.append(path)
 for path in [prior.prior.WEIGHTS,prior.prior.rgb.prior.resolution.WEIGHTS]+list(prior.prior.VENDOR.rglob('*.py')):assert file_hash(path)==old['protocol']['input_sha256'][str(path)]
 grid=json.loads((prior.prior.auxiliary.ROOT/'summary.json').read_text())['protocol']['calibration_grid'];p={'round':78,'source_validation_only':True,'models':models,'primary_model_fixed':'B14_aux_sam','video_ids':[c['video_id']for c in clips],'query_seed_base':20261078,'queries_per_source_frame':256,'source_frames':8,'calibration_grid':grid,'offset_tie_break':'abs smaller then negative first','model_selection':False,'training_updates':0,'input_sha256':{str(path):file_hash(path)for path in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',{'val':clips});return clips,p

def build(clips):
 if(ROOT/'coverage.json').exists():return json.loads((ROOT/'coverage.json').read_text())
 ds=load_dataset('ovis',ANNOTATIONS,Path('semantic_artifacts/ovis'));coverage=[]
 for c in clips:
  points=[];targets=[]
  for t in range(8):
   selected=np.random.default_rng(20261078+c['video_id']*16+t).choice(16384,256,replace=False);xy=np.column_stack((selected%128,selected//128,np.full(256,t))).astype(np.int16);labels=ds.gt_label_map(c,t)[xy[:,1],xy[:,0]];points.append(xy);targets.append(labels);coverage.append({'video_id':c['video_id'],'source_time':t,'valid_queries':int((labels>=0).sum()),'ignored_queries':int((labels<0).sum()),'support':{str(int(k)):int(v)for k,v in zip(*np.unique(labels,return_counts=True))}})
  save_npz(qp(c['video_id']),xy=np.concatenate(points),labels=np.concatenate(targets),video_id=np.int64(c['video_id']))
 atomic_json(ROOT/'coverage.json',coverage);return coverage

def cache(clips):
 for bone in ('S14','B14'):
  tasks=[c for c in clips if not fp(bone,c['video_id']).exists()]
  if not tasks:continue
  model=prior.prior.rgb.prior.resolution.load_dino()if bone=='S14'else prior.prior.load_b14();dim=384 if bone=='S14'else 768
  for c in tasks:
   with np.load(qp(c['video_id']))as a:xy=a['xy'].copy()
   video=capacity.rgb.load_native_video(c).cuda();tokens=prior.prior.rgb.prior.resolution.extract_tokens(model,video,448)if bone=='S14'else prior.prior.tokens_b14(model,video)
   with torch.inference_mode():decoded=patch_at_queries(tokens,xy).half().cpu().numpy()
   assert decoded.shape==(2048,dim)and np.isfinite(decoded).all();save_npz(fp(bone,c['video_id']),decoded=decoded,xy=xy,video_id=np.int64(c['video_id']));print('spatial calibration features',bone,c['video_id'],flush=True)
  del model;gc.collect();torch.cuda.empty_cache()

def diagnose(out,offset):
 result=evaluate(out,offset);matrix=np.asarray(result['confusion_matrix']);bg=matrix[0].sum();fg=matrix[1:].sum();result['background_to_foreground_rate']=float(matrix[0,1:].sum()/bg)if bg else None;result['foreground_class_confusion_rate']=float((matrix[1:,1:].sum()-np.trace(matrix[1:,1:]))/fg)if fg else None;return result

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed spatial calibration exists')
  torch.backends.cuda.matmul.allow_tf32=False;clips,p=freeze();coverage=build(clips);cache(clips);labels=[];times=[];ids=[];bybone={bone:[]for bone in ('S14','B14')}
  for c in clips:
   with np.load(qp(c['video_id']))as a:labels.append(a['labels'].copy());times.append(a['xy'][:,2].copy());ids.append(np.full(2048,c['video_id'],np.int32))
   for bone in bybone:
    with np.load(fp(bone,c['video_id']))as a:bybone[bone].append(a['decoded'].copy())
  labels=np.concatenate(labels);times=np.concatenate(times);ids=np.concatenate(ids);valid=labels>=0;results={};offsets={};curves={}
  for arm in ARMS:
   outs=[];allouts=[]
   for seed in range(3):
    path=prior.prior.ckpt(arm,seed);state=torch.load(path,map_location='cpu',weights_only=True)['model'];weight=state['classifier.weight'].cuda();bias=state['classifier.bias'].cuda()
    with torch.inference_mode():values=np.concatenate([F.linear(torch.from_numpy(a).float().cuda(),weight,bias).cpu().numpy()for a in bybone[arm.split('_')[0]]])
    outpath=ROOT/'logits'/f'{arm}_{seed}.npz';save_npz(outpath,logits=values,labels=labels,times=times,video_ids=ids,valid_mask=valid,checkpoint_sha256=np.array(file_hash(path)));allouts.append(outpath);outs.append({'logits':values[valid],'labels':labels[valid],'times':times[valid],'video_ids':ids[valid]})
   offsets[arm],curves[arm]=pick_offset(outs,p['calibration_grid']);old=p['models'][arm]['offset'];runs=[]
   for seed,out in enumerate(outs):
    metrics={name:diagnose(out,offset)for name,offset in [('raw',0),('old_calibration',old),('spatial_calibration',offsets[arm])]};runs.append({'seed':seed,'metrics':metrics,'output_path':str(allouts[seed]),'output_sha256':file_hash(allouts[seed])})
   results[arm]={'old_offset':old,'new_spatial_offset':offsets[arm],'runs':runs};print('source spatial offset',arm,old,'to',offsets[arm],flush=True)
  selection={'primary_model_fixed':p['primary_model_fixed'],'model_selection':False,'old_offsets':{a:p['models'][a]['offset']for a in ARMS},'spatial_offsets':offsets,'curves':curves,'models':p['models'],'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'results':results,'coverage_sha256':file_hash(ROOT/'coverage.json'),'query_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'queries').glob('*.npz'))},'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'features').rglob('*.npz'))},'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'total_coordinates':40960,'valid_queries':int(valid.sum()),'ignored_queries':int((~valid).sum()),'new_confirmation_groups':0,'LVVIS40_not_evaluated':True,'deployment_unchanged':'native_gt'});print('DONE source-only uniform spatial calibration',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
