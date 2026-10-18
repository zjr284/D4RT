"""Validation-only S14/B14 x complete first-frame GT / SAM-label scheme."""
import fcntl,gc,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_native_rgb_control as rgb
from semantic import dino_backbone_training as prior
from semantic import dino_class_video_sampling as globalchoice
from semantic.dino_backbone_resource import WEIGHTS,VENDOR,load_b14,tokens_b14
from semantic.dino_low_batch_confirmation import checkpoint as oldckpt
from semantic.semantic_feature_control import patch_at_queries,save_npz
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.model import SemanticHead
from semantic.background_calibration import evaluate,pick_offset
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_b14_sam_control');ARMS=('S14_gt','S14_sam','B14_gt','B14_sam')
def ckpt(arm,seed):
 if arm=='S14_gt':return oldckpt('source_gt',seed)
 if arm=='S14_sam':return oldckpt('native_sam',seed)
 if arm=='B14_gt':return prior.ckpt('B14_gt256',seed)
 return ROOT/arm/'runs'/f'linear_seed{seed}.pt'
def fp(vid):return ROOT/'features'/'train'/f'{vid}.npz'
def freeze():
 original=json.loads((rgb.ROOT/'summary.json').read_text());b14=json.loads((prior.ROOT/'summary.json').read_text());m=json.loads((prior.ROOT/'manifest.json').read_text());selection=json.loads((globalchoice.ROOT/'frozen_selection.json').read_text());candidate=json.loads((globalchoice.ROOT/'summary.json').read_text())
 for root in (rgb.ROOT,prior.ROOT,globalchoice.ROOT):assert file_hash(root/'summary.json')==json.loads((root/'verification.json').read_text())['summary_sha256']
 paths=[rgb.ROOT/'summary.json',rgb.ROOT/'verification.json',prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'manifest.json',globalchoice.ROOT/'summary.json',globalchoice.ROOT/'verification.json',globalchoice.ROOT/'frozen_selection.json',WEIGHTS,rgb.prior.resolution.WEIGHTS,Path('semantic/dino_b14_sam_control.py'),Path('semantic/OVIS_DINO_B14_SAM_PROTOCOL.md'),Path('semantic/dino_backbone_resource.py'),Path('semantic/dino_native_rgb_control.py'),Path('semantic/dino_resolution_control.py'),Path('semantic/semantic_feature_control.py'),Path('semantic/sampling_ablation.py'),Path('semantic/model.py'),Path('semantic/head_budget_probe.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py')]+sorted(VENDOR.rglob('*.py'))
 for split in m:
  for c in m[split]:
   paths.extend([rgb.feature_path('native_sam',split,c['video_id']),prior.fp(split,c['video_id'])])
   if split=='train':paths.extend(Path(f)for f in c['frames'])
 for arm in ARMS[:3]:
  for seed in range(3):paths.extend([ckpt(arm,seed),ckpt(arm,seed).with_suffix('.json')])
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()=='7764ea0f912e53c92e82eb78a2a1631e92725fc8';subprocess.run(['git','diff','--quiet','HEAD'],cwd=VENDOR,check=True)
 p={'round':69,'arms':list(ARMS),'seeds':[0,1,2],'new_heads':3,'steps':2000,'head_parameters':19994,'batch':256,'total_exposures':512000,'teacher_queries':162560,'GT_anchor_queries':20480,'calibration_grid':b14['protocol']['calibration_grid'],'fixed_offsets':{'S14_gt':original['results']['native_gt']['offset'],'S14_sam':original['results']['native_sam']['offset'],'B14_gt':b14['results']['B14_gt256']['offset']},'validation_only':True,'current_global_validation_candidate':selection['validation_selected_arm'],'current_global_validation_macro':candidate['results'][selection['validation_selected_arm']]['mean_calibrated_validation_macro'],'input_sha256':{str(x):file_hash(x)for x in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p,original,b14

def cache(m):
 tasks=[c for c in m['train']if not fp(c['video_id']).exists()]
 if not tasks:return
 model=load_b14()
 for c in tasks:
  with np.load(rgb.feature_path('native_sam','train',c['video_id']))as a:meta={k:a[k].copy()for k in ('labels','xy','t_tgt','t_cam','video_id')}
  source=meta['xy'][:,2]==0
  with np.load(prior.fp('train',c['video_id']))as a:
   assert np.array_equal(meta['labels'][source],a['labels'])and np.array_equal(meta['xy'][source],a['xy']);anchors=a['decoded'].copy()
  video=rgb.load_native_video(c).cuda();tokens=tokens_b14(model,video)
  with torch.inference_mode():features=patch_at_queries(tokens,meta['xy']).half().cpu().numpy()
  features[source]=anchors;assert np.isfinite(features).all();save_npz(fp(c['video_id']),decoded=features,**meta);del tokens,video;print('B14 SAM teacher features',c['video_id'],len(features),flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
def read_train(m):
 out=[]
 for c in sorted(m['train'],key=lambda c:str(c['video_id'])):
  with np.load(fp(c['video_id']))as a:out.append({k:a[k].copy()for k in a.files})
 return out

def train(seed,source,val,p):
 path=ckpt('B14_sam',seed);rp=path.with_suffix('.json')
 if rp.exists():
  r=json.loads(rp.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();features=torch.from_numpy(np.concatenate([a['decoded']for a in source])).float().cuda();labels=np.concatenate([a['labels']for a in source]);times=np.concatenate([a['xy'][:,2]for a in source]);assert len(labels)==162560 and int((times==0).sum())==20480
 sampler=Sampler(labels,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);hist=torch.zeros(8*26,dtype=torch.long,device='cuda');ih=hashlib.sha256();seen=np.zeros(len(labels),bool);best=-1;history=[]
 for step in range(1,2001):
  fg,bg=sampler.draw(gen);ix=torch.cat([fg,bg]);idx=ix.cpu().numpy();ih.update(idx.tobytes());seen[idx]=True;hist+=torch.bincount(sampler.times[ix]*26+sampler.labels[ix],minlength=8*26)
  model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(features[ix]),sampler.labels[ix]);loss.backward();optimizer.step()
  if step%50==0:
   value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 counts=hist.reshape(8,26).cpu().numpy();old=json.loads(ckpt('S14_sam',seed).with_suffix('.json').read_text());assert counts.sum(0).tolist()==old['sampled_class_counts']and counts.sum()==512000 and counts[:,0].sum()==256000
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},temp);temp.replace(path)
 r={'seed':seed,'protocol':p,'best_step':best_step,'best_val':best,'history':history,'parameter_count':19994,'sampled_indices_sha256':ih.hexdigest(),'sampled_time_class_counts':counts.tolist(),'sampled_class_counts':counts.sum(0).tolist(),'GT_anchor_exposures':int(counts[0].sum()),'unique_sampled_queries':int(seen.sum()),'unique_sampled_GT_anchor_queries':int(seen[times==0].sum()),'checkpoint_sha256':file_hash(path)};atomic_json(rp,r);print('trained B14 SAM',seed,best_step,best,'GTanchor visits',r['GT_anchor_exposures'],flush=True);del model,features;gc.collect();torch.cuda.empty_cache();return r

def infer(arm,seed,val):
 path=ROOT/arm/'logits'/f'seed{seed}_val.npz';h=file_hash(ckpt(arm,seed))
 if path.exists():
  with np.load(path)as a:out={k:a[k].copy()for k in a.files}
  assert str(out['checkpoint_sha256'])==h;return out
 model=SemanticHead(hidden_dim=384 if arm.startswith('S14')else 768,num_classes=26).cuda().eval();model.load_state_dict(torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model'])
 with torch.inference_mode():values=[model(torch.from_numpy(a['decoded']).float().cuda()).cpu().numpy()for a in val]
 out={'logits':np.concatenate(values),'labels':np.concatenate([a['labels']for a in val]),'times':np.concatenate([a['xy'][:,2]for a in val]),'video_ids':np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in val]),'checkpoint_sha256':np.array(h)};save_npz(path,**out);del model;return out

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed B14 SAM control exists')
  torch.backends.cuda.matmul.allow_tf32=False;m,p,original,b14=freeze();cache(m);source=read_train(m);valbig=prior.read('val',m);valsmall=rgb.read('native_sam','val',m);reports=[train(seed,source,valbig,p)for seed in range(3)];results={};offsets=dict(p['fixed_offsets']);curves={}
  for arm in ARMS:
   outputs=[infer(arm,seed,valsmall if arm.startswith('S14')else valbig)for seed in range(3)]
   if arm=='B14_sam':offsets[arm],curves[arm]=pick_offset(outputs,p['calibration_grid'])
   runs=[{'seed':seed,'raw_validation':evaluate(outputs[seed],0),'calibrated_validation':evaluate(outputs[seed],offsets[arm])}for seed in range(3)]
   if arm!='B14_sam':
    reference=b14['results']['B14_gt256']['runs']if arm=='B14_gt'else original['results']['native_gt'if arm=='S14_gt'else'native_sam']['runs']
    for seed in range(3):
     for key in ('raw_validation','calibrated_validation'):assert runs[seed][key]==reference[seed][key]
   results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))};print('GT/SAM factorial source validation',arm,results[arm]['mean_calibrated_validation_macro'],flush=True)
  contrasts={}
  terms={'SAM_B14':[('B14_sam',1),('B14_gt',-1)],'SAM_S14':[('S14_sam',1),('S14_gt',-1)],'interaction':[('B14_sam',1),('B14_gt',-1),('S14_sam',-1),('S14_gt',1)]}
  for name,term in terms.items():
   values=[sum(weight*results[a]['runs'][seed]['calibrated_validation']['macro_f1_present']for a,weight in term)for seed in range(3)];contrasts[name]={'per_seed_validation_macro_difference':values,'mean':float(np.mean(values))}
  selected=max(ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro']);selection={'within_factorial_validation_choice':selected,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{a}_{seed}':file_hash(ckpt(a,seed))for a in ARMS for seed in range(3)},'new_SAM_exceeds_current_global_validation_candidate':results['B14_sam']['mean_calibrated_validation_macro']>p['current_global_validation_macro'],'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'validation_contrasts':contrasts,'feature_hashes':{str(fp(c['video_id'])):file_hash(fp(c['video_id']))for c in m['train']},'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'within_factorial_validation_choice':selected,'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE B14 SAM source-only factorial',selected,contrasts,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
