"""Validation-only frozen S14/B14 x source GT batch capacity control."""
from collections import OrderedDict
import fcntl,gc,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_native_rgb_control as rgb
from semantic import dino_source_batch_control as batch32
from semantic.dino_backbone_resource import ROOT,WEIGHTS,VENDOR,load_b14,tokens_b14
from semantic.dino_low_batch_confirmation import checkpoint as old_checkpoint
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.semantic_feature_control import save_npz,infer,patch_at_queries
from semantic.background_calibration import evaluate,pick_offset
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ARMS=('S14_gt256','S14_gt32','B14_gt256','B14_gt32')
def ckpt(arm,seed):return old_checkpoint('source_gt'if arm.endswith('256')else'source_gt32',seed)if arm.startswith('S14')else ROOT/arm/'runs'/f'linear_seed{seed}.pt'
def fp(split,vid):return ROOT/'features'/split/f'{vid}.npz'
def read(split,manifest):
 out=[]
 for clip in sorted(manifest[split],key=lambda c:str(c['video_id'])):
  with np.load(fp(split,clip['video_id']))as a:out.append({k:a[k].copy()for k in a.files})
 return out

def freeze():
 pre=json.loads((ROOT/'resource_preflight.json').read_text());assert pre['weight_sha256']==file_hash(WEIGHTS)and pre['source_commit']=='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
 assert torch.__version__==pre['torch_version']and torch.version.cuda==pre['cuda_version']and torch.cuda.get_device_name(0)==pre['device']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()==pre['source_commit'];subprocess.run(['git','diff','--quiet','HEAD'],cwd=VENDOR,check=True)
 old=json.loads((batch32.ROOT/'summary.json').read_text());v=json.loads((batch32.ROOT/'verification.json').read_text());assert file_hash(batch32.ROOT/'summary.json')==v['summary_sha256']
 m={split:old['manifest'][split]for split in ('train','val')};assert len(m['train'])==80 and len(m['val'])==20
 atomic_json(ROOT/'manifest.json',m)
 paths=[ROOT/'resource_preflight.json',ROOT/'acquisition.json',batch32.ROOT/'summary.json',batch32.ROOT/'verification.json',ROOT/'manifest.json',WEIGHTS,Path('semantic/dino_backbone_resource.py'),Path('semantic/dino_backbone_capacity.py'),Path('semantic/OVIS_DINO_BACKBONE_CAPACITY_PROTOCOL.md')]
 paths+=sorted(VENDOR.rglob('*.py'))
 for split in m:
  for c in m[split]:
   paths.append(rgb.feature_path('native_sam',split,c['video_id']));paths.extend(Path(p)for p in (c['frames'][:1]if split=='train'else c['frames']))
 for a in ARMS[:2]:
  for seed in range(3):paths.extend([ckpt(a,seed),ckpt(a,seed).with_suffix('.json')])
 p={'round':56,'validation_only':True,'seeds':[0,1,2],'arms':list(ARMS),'primary':['B14_gt256','S14_gt256'],'steps':2000,'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.01},'head_parameters':{'S14':10010,'B14':19994},'feature_dimensions':{'S14':384,'B14':768},'training_queries':20480,'validation_queries':40960,'calibration_grid':old['protocol']['calibration_grid'],'fixed_offsets':{'S14_gt256':old['results']['source_gt256']['offset'],'S14_gt32':old['results']['source_gt32']['offset']},'TF32_disabled':True,'new_heads':6,'confirmation_evaluation':False,'input_sha256':{str(x):file_hash(x)for x in paths}}
 path=ROOT/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==p
 else:atomic_json(path,p)
 return m,p,old

def cache(manifest):
 tasks=[(split,c)for split in manifest for c in manifest[split]if not fp(split,c['video_id']).exists()]
 if not tasks:return
 model=load_b14();cost=[]
 for split,clip in tasks:
  with np.load(rgb.feature_path('native_sam',split,clip['video_id']))as a:meta={k:a[k].copy()for k in ('labels','xy','t_tgt','t_cam','video_id')}
  if split=='train':
   ix=meta['xy'][:,2]==0
   for k in ('labels','xy','t_tgt','t_cam'):meta[k]=meta[k][ix]
   assert len(meta['labels'])==256 and np.all(meta['xy'][:,2]==0)
  video=rgb.load_native_video({'frames':clip['frames'][:1]if split=='train'else clip['frames']}).cuda();torch.cuda.synchronize();start=time.perf_counter();tokens=tokens_b14(model,video)
  with torch.inference_mode():features=patch_at_queries(tokens,meta['xy']).half().cpu().numpy()
  torch.cuda.synchronize();assert features.shape==(len(meta['labels']),768)and np.isfinite(features).all();save_npz(fp(split,clip['video_id']),decoded=features,**meta)
  cost.append({'split':split,'video_id':clip['video_id'],'seconds':time.perf_counter()-start});print('B14 cached',split,clip['video_id'],flush=True);del tokens,video
 atomic_json(ROOT/'extraction_cost.json',cost);del model;gc.collect();torch.cuda.empty_cache()

def train(arm,seed,clips,val,p):
 path=ckpt(arm,seed);report=path.with_suffix('.json')
 if report.exists():
  r=json.loads(report.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();assert sum(x.numel()for x in model.parameters())==19994
 features=torch.from_numpy(np.concatenate([c['decoded']for c in clips])).float().cuda();labels=np.concatenate([c['labels']for c in clips]);times=np.concatenate([c['xy'][:,2]for c in clips]);assert np.all(times==0)and len(labels)==20480
 sampler=Sampler(labels,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 count=128 if arm.endswith('256')else 16;hist=torch.zeros(26,dtype=torch.long,device='cuda');ih=hashlib.sha256();best=-1;history=[]
 for step in range(1,2001):
  fg,bg=sampler.draw(gen,count=count);ix=torch.cat([fg,bg]);ih.update(ix.cpu().numpy().tobytes());hist+=torch.bincount(sampler.labels[ix],minlength=26)
  model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(features[ix]),sampler.labels[ix]);loss.backward();optimizer.step()
  if step%50==0:
   value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 assert int(hist[0])==2000*count and int(hist.sum())==4000*count
 reference=json.loads(ckpt('S14_gt256'if count==128 else'S14_gt32',seed).with_suffix('.json').read_text())
 assert hist.cpu().tolist()==reference['sampled_class_counts']
 if count==16:assert ih.hexdigest()==reference['sampled_indices_sha256']
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},temp);temp.replace(path)
 r={'arm':arm,'seed':seed,'protocol':p,'best_step':best_step,'best_val':best,'history':history,'sampled_class_counts':hist.cpu().tolist(),'sampled_indices_sha256':ih.hexdigest(),'source_exposures':4000*count,'parameter_count':19994,'checkpoint_sha256':file_hash(path)};atomic_json(report,r)
 print('B14 trained',arm,seed,best_step,best,flush=True);del model,features;gc.collect();torch.cuda.empty_cache();return r

def infer_b14(arm,seed,clips):
 path=ROOT/arm/'logits'/f'dinov2_{seed}_val.npz';h=file_hash(ckpt(arm,seed))
 if path.exists():
  with np.load(path)as a:out={k:a[k].copy()for k in a.files}
  assert str(out['checkpoint_sha256'])==h;return out
 model=SemanticHead(hidden_dim=768,num_classes=26).cuda().eval();model.load_state_dict(torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model'])
 with torch.inference_mode():values=[model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in clips]
 out={'logits':np.concatenate(values),'labels':np.concatenate([c['labels']for c in clips]),'times':np.concatenate([c['xy'][:,2]for c in clips]),'video_ids':np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int32)for c in clips]),'checkpoint_sha256':np.array(h)};save_npz(path,**out);del model;return out

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (ROOT/'summary.json').exists():raise RuntimeError('Completed capacity control exists')
  torch.backends.cuda.matmul.allow_tf32=False;m,p,previous=freeze();cache(m);trainset=read('train',m);val=read('val',m);oldval=rgb.read('native_sam','val',m)
  assert sum(len(c['labels'])for c in trainset)==20480 and sum(len(c['labels'])for c in val)==40960
  reports={a:[train(a,s,trainset,val,p)for s in range(3)]for a in ARMS[2:]};offsets=dict(p['fixed_offsets']);curves={};results={}
  for arm in ARMS:
   outputs=[infer('dinov2',s,'val',oldval,checkpoint_path=ckpt(arm,s),output_root=ROOT/arm)if arm.startswith('S14')else infer_b14(arm,s,val)for s in range(3)]
   if arm.startswith('B14'):offsets[arm],curves[arm]=pick_offset(outputs,p['calibration_grid'])
   runs=[{'seed':s,'raw_validation':evaluate(outputs[s],0),'calibrated_validation':evaluate(outputs[s],offsets[arm])}for s in range(3)]
   if arm.startswith('S14'):
    key='source_gt256'if arm.endswith('256')else'source_gt32'
    for s in range(3):assert runs[s]==previous['results'][key]['runs'][s]
   results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))};print('capacity validation',arm,results[arm]['mean_calibrated_validation_macro'],flush=True)
  selected=max(ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro']);selection={'validation_selected_arm':selected,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{a}_{s}':file_hash(ckpt(a,s))for a in ARMS for s in range(3)},'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'validation_selected_arm':selected,'feature_hashes':{str(fp(split,c['video_id'])):file_hash(fp(split,c['video_id']))for split in m for c in m[split]},'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'confirmation_evaluated':False,'deployment_unchanged':'native_gt'})
  print('DONE B14 capacity exploration',selected,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
