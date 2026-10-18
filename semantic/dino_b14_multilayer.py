"""Single preregistered last-four normalized patch mean at fixed768 dimensions."""
import fcntl,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_backbone_training as prior
from semantic import dino_b14_projection_control as projection
from semantic.dino_backbone_resource import WEIGHTS,VENDOR,load_b14
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_b14_multilayer');ARMS=('final','mean_last4')
def ckpt(arm,seed):return prior.ckpt('B14_gt32',seed)if arm=='final'else ROOT/'mean_last4'/'runs'/f'linear_seed{seed}.pt'
def fp(split,vid):return ROOT/'features'/split/f'{vid}.npz'

def freeze():
 sv=json.loads((prior.ROOT/'verification.json').read_text());s=json.loads((prior.ROOT/'summary.json').read_text());assert file_hash(prior.ROOT/'summary.json')==sv['summary_sha256'];assert file_hash(projection.ROOT/'summary.json')==json.loads((projection.ROOT/'verification.json').read_text())['summary_sha256'];m=json.loads((prior.ROOT/'manifest.json').read_text());assert m['train'][0]['video_id']==562 and m['val'][0]['video_id']==258
 paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'manifest.json',prior.ROOT/'frozen_selection.json',projection.ROOT/'summary.json',projection.ROOT/'verification.json',WEIGHTS,Path('semantic/dino_b14_multilayer.py'),Path('semantic/OVIS_DINO_B14_MULTILAYER_PROTOCOL.md'),Path('semantic/dino_backbone_resource.py'),Path('semantic/dino_native_rgb_control.py'),Path('semantic/semantic_feature_control.py'),Path('semantic/model.py'),Path('semantic/sampling_ablation.py'),Path('semantic/head_budget_probe.py'),Path('semantic/background_calibration.py')]+sorted(VENDOR.rglob('*.py'))
 assert file_hash(WEIGHTS)=='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
 for split in m:
  for c in m[split]:paths.append(prior.fp(split,c['video_id']));paths.extend(Path(x)for x in(c['frames'][:1]if split=='train'else c['frames']))
 for seed in range(3):paths.extend([ckpt('final',seed),ckpt('final',seed).with_suffix('.json')])
 p={'round':76,'arms':list(ARMS),'new_heads':3,'feature_dimension':768,'head_parameters':19994,'layer_recipe':'last4 blocks each official norm then FP32 mean, no renorm','training_source_frames':1,'validation_source_frames':8,'steps':2000,'batch':32,'exposures':64000,'fixed_final_offset':s['results']['B14_gt32']['offset'],'calibration_grid':s['protocol']['calibration_grid'],'prior_global_source_validation':.8326561544706612,'input_sha256':{str(path):file_hash(path)for path in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p,s

@torch.inference_mode()
def tokens(model,video):
 mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];image=(F.interpolate(video.float(),size=(448,448),mode='bicubic',align_corners=False)-mean)/std
 with torch.autocast('cuda',dtype=torch.bfloat16):layers=model.get_intermediate_layers(image,n=4,norm=True,reshape=False)
 assert len(layers)==4 and all(x.shape==(len(video),1024,768)for x in layers);pooled=torch.stack([x.float()for x in layers]).mean(0);assert pooled.dtype==torch.float32 and torch.isfinite(pooled).all()
 return pooled.reshape(len(video),32,32,768).permute(0,3,1,2),layers[-1].reshape(len(video),32,32,768).permute(0,3,1,2)

def cache(m):
 tasks=[(split,c)for split in m for c in m[split]if not fp(split,c['video_id']).exists()];cost=json.loads((ROOT/'extraction.json').read_text())if(ROOT/'extraction.json').exists()else[]
 if not tasks:return
 model=load_b14()
 for split,clip in tasks:
  with np.load(prior.fp(split,clip['video_id']))as a:old={k:a[k].copy()for k in a.files}
  frames=clip['frames'][:1]if split=='train'else clip['frames'];video=prior.rgb.load_native_video({'frames':frames}).cuda();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();pooled,last=tokens(model,video);decoded=patch_at_queries(pooled,old['xy']).half().cpu().numpy();last_values=patch_at_queries(last,old['xy']).half().cpu().numpy();assert np.array_equal(last_values,old['decoded']);save_npz(fp(split,clip['video_id']),decoded=decoded,**{k:v for k,v in old.items()if k!='decoded'});torch.cuda.synchronize();cost.append({'split':split,'video_id':clip['video_id'],'last_layer_bitwise_equal':True,'last_layer_values_sha256':hashlib.sha256(last_values.tobytes()).hexdigest(),'seconds':time.perf_counter()-t,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'pooled_patch_dtype':str(pooled.dtype)});atomic_json(ROOT/'extraction.json',cost);print('multilayer features',split,clip['video_id'],flush=True);del video,pooled,last,decoded,last_values
 del model;gc.collect();torch.cuda.empty_cache()

def read(split,m):
 data=[]
 for c in sorted(m[split],key=lambda c:str(c['video_id'])):
  with np.load(fp(split,c['video_id']))as a:data.append({k:a[k].copy()for k in a.files})
 return data

def train(seed,base,val,p):
 path=ckpt('mean_last4',seed);report=path.with_suffix('.json')
 if report.exists():
  r=json.loads(report.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();x=torch.from_numpy(np.concatenate([a['decoded']for a in base])).float().cuda();y=np.concatenate([a['labels']for a in base]);times=np.concatenate([a['xy'][:,2]for a in base]);assert (times==0).all()and len(y)==20480;sampler=Sampler(y,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];hist=torch.zeros(26,dtype=torch.long,device='cuda');ih=hashlib.sha256();best=-1
 for step in range(1,2001):
  f,b=sampler.draw(gen,count=16);ix=torch.cat([f,b]);ih.update(ix.cpu().numpy().tobytes());hist+=torch.bincount(sampler.labels[ix],minlength=26);model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ix]),sampler.labels[ix]);loss.backward();optimizer.step()
  if step%50==0:
   value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 reference=json.loads(ckpt('final',seed).with_suffix('.json').read_text());assert ih.hexdigest()==reference['sampled_indices_sha256']and hist.cpu().tolist()==reference['sampled_class_counts'];path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},tmp);tmp.replace(path);r={'seed':seed,'protocol':p,'best_step':best_step,'best_val':best,'history':history,'sampled_indices_sha256':ih.hexdigest(),'sampled_class_counts':hist.cpu().tolist(),'parameter_count':19994,'checkpoint_sha256':file_hash(path)};atomic_json(report,r);print('trained mean_last4',seed,best_step,best,flush=True);del model,x;gc.collect();torch.cuda.empty_cache();return r

def infer(arm,seed,val):
 path=ROOT/arm/'logits'/f'seed{seed}_val.npz';h=file_hash(ckpt(arm,seed))
 if path.exists():
  with np.load(path)as a:out={k:a[k].copy()for k in a.files}
  assert str(out['checkpoint_sha256'])==h;return out
 model=SemanticHead(hidden_dim=768,num_classes=26).cuda().eval();model.load_state_dict(torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model'])
 with torch.inference_mode():values=[model(torch.from_numpy(a['decoded']).float().cuda()).cpu().numpy()for a in val]
 out={'logits':np.concatenate(values),'labels':np.concatenate([a['labels']for a in val]),'times':np.concatenate([a['xy'][:,2]for a in val]),'video_ids':np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in val]),'checkpoint_sha256':np.array(h)};save_npz(path,**out);del model;return out

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed multilayer feature control exists')
  torch.backends.cuda.matmul.allow_tf32=False;m,p,old=freeze();cache(m);base=read('train',m);val=read('val',m);oldval=prior.read('val',m);reports=[train(seed,base,val,p)for seed in range(3)];results={};offsets={'final':p['fixed_final_offset']};curves={}
  for arm in ARMS:
   outputs=[infer(arm,seed,val if arm=='mean_last4'else oldval)for seed in range(3)]
   if arm=='mean_last4':offsets[arm],curves[arm]=pick_offset(outputs,p['calibration_grid'])
   runs=[{'seed':seed,'raw_validation':evaluate(outputs[seed],0),'calibrated_validation':evaluate(outputs[seed],offsets[arm])}for seed in range(3)]
   if arm=='final':assert runs==old['results']['B14_gt32']['runs']
   results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([a['calibrated_validation']['macro_f1_present']for a in runs]))};print('multilayer validation',arm,results[arm]['mean_calibrated_validation_macro'],flush=True)
  differences=[results['mean_last4']['runs'][seed]['calibrated_validation']['macro_f1_present']-results['final']['runs'][seed]['calibrated_validation']['macro_f1_present']for seed in range(3)];selected=max(ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro']);selection={'validation_selected_arm':selected,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{a}_{seed}':file_hash(ckpt(a,seed))for a in ARMS for seed in range(3)},'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'mean_last4_minus_final':{'per_seed_validation_macro_difference':differences,'mean':float(np.mean(differences))},'validation_selected_arm':selected,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'features').rglob('*.npz'))},'extraction_sha256':file_hash(ROOT/'extraction.json'),'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE B14 last4 mean feature control',selected,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
