"""GT-priority one-way raw-gradient projection, distinct from symmetric PCGrad."""
import fcntl,gc,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_b14_auxiliary as prior
from semantic import dino_b14_gradient_audit as audit
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_b14_projection_control');ARMS=('gt32','extra_gt','aux_sam','project_extra_gt','project_sam');NEW=ARMS[-2:]

def project(base,supplement):
 if base.shape!=supplement.shape or not torch.isfinite(base).all()or not torch.isfinite(supplement).all():raise ValueError('Matching finite gradient vectors required')
 dot=torch.dot(base,supplement);norm2=torch.dot(base,base);alpha=dot.clamp_max(0)/(norm2+1e-12);return supplement-alpha*base,alpha

def ckpt(arm,seed):return prior.ckpt(arm,seed)if arm not in NEW else ROOT/arm/'runs'/f'linear_seed{seed}.pt'
def freeze():
 v=json.loads((audit.ROOT/'verification.json').read_text());assert file_hash(audit.ROOT/'summary.json')==v['summary_sha256'];old=json.loads((prior.ROOT/'summary.json').read_text());assert file_hash(prior.ROOT/'summary.json')==json.loads((prior.ROOT/'verification.json').read_text())['summary_sha256'];m=json.loads((prior.ROOT/'manifest.json').read_text());paths=[audit.ROOT/'summary.json',audit.ROOT/'verification.json',prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'manifest.json',Path('semantic/dino_b14_projection_control.py'),Path('semantic/OVIS_DINO_B14_PROJECTION_PROTOCOL.md'),Path('tests/test_dino_b14_projection.py')]
 for path,h in old['protocol']['input_sha256'].items():assert file_hash(path)==h;paths.append(Path(path))
 for a in ARMS[:3]:
  for seed in range(3):paths.extend([ckpt(a,seed),ckpt(a,seed).with_suffix('.json')])
 run=subprocess.run(['.venv/bin/python','tests/test_dino_b14_projection.py'],capture_output=True,text=True,check=True);(ROOT/'unit_checks.log').write_text(run.stdout+run.stderr);assert 'PASS' in run.stdout
 p={'round':75,'arms':list(ARMS),'new_arms':list(NEW),'steps':2000,'base_batch':32,'supplement_batch':32,'base_GT_exposures':64000,'total_exposures':128000,'head_parameters':19994,'eps':1e-12,'calibration_grid':old['protocol']['calibration_grid'],'fixed_offsets':{a:old['results'][a]['offset']for a in ARMS[:3]},'source_validation_only':True,'unit_checks_sha256':file_hash(ROOT/'unit_checks.log'),'input_sha256':{str(path):file_hash(path)for path in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p,old

def train(arm,seed,base,later,val,p):
 path=ckpt(arm,seed);report=path.with_suffix('.json')
 if report.exists():
  r=json.loads(report.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();params=list(model.parameters());bx=torch.from_numpy(np.concatenate([a['decoded']for a in base])).float().cuda();by=np.concatenate([a['labels']for a in base]);bs=Sampler(by,np.zeros(len(by),np.int64),'baseline');data=base if arm=='project_extra_gt'else later;sx=torch.from_numpy(np.concatenate([a['decoded']for a in data])).float().cuda();sy=np.concatenate([a['labels']for a in data]);st=np.concatenate([a['xy'][:,2]for a in data]);ss=Sampler(sy,st,'baseline');bg=torch.Generator(device='cuda').manual_seed(seed);sg=torch.Generator(device='cuda').manual_seed(20261071+seed);optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.01);hashes=[hashlib.sha256(),hashlib.sha256()];hist=torch.zeros(26,dtype=torch.long,device='cuda');supphist=torch.zeros(208,dtype=torch.long,device='cuda');history=[];diagnostics=[];best=-1;prefix=[{k:v.detach().cpu().clone()for k,v in model.state_dict().items()}]
 for step in range(1,2001):
  f,b=bs.draw(bg,count=16);bi=torch.cat([f,b]);f,b=ss.draw(sg,count=16);si=torch.cat([f,b]);hashes[0].update(bi.cpu().numpy().tobytes());hashes[1].update(si.cpu().numpy().tobytes());hist+=torch.bincount(bs.labels[bi],minlength=26);supphist+=torch.bincount(ss.times[si]*26+ss.labels[si],minlength=208)
  model.train();optimizer.zero_grad(set_to_none=True);lb=F.cross_entropy(model(bx[bi]),bs.labels[bi]);ls=F.cross_entropy(model(sx[si]),ss.labels[si]);gb=torch.cat([g.reshape(-1)for g in torch.autograd.grad(lb,params)]);gs=torch.cat([g.reshape(-1)for g in torch.autograd.grad(ls,params)]);projected,alpha=project(gb,gs);combined=gb+projected;before=torch.cat([q.detach().flatten()for q in params]);offset=0
  for q in params:q.grad=combined[offset:offset+q.numel()].reshape_as(q);offset+=q.numel()
  optimizer.step();after=torch.cat([q.detach().flatten()for q in params]);base2=torch.dot(gb,gb);supp2=torch.dot(gs,gs);post2=torch.dot(projected,projected);dot=torch.dot(gs,gb);postdot=torch.dot(projected,gb);scale=(base2*post2).sqrt().clamp_min(1e-12);assert postdot>=-2e-5*scale
  diagnostics.append([step,float(dot),float(base2),float(supp2),float(alpha),float(postdot),float(post2),float(torch.dot(before-after,gb))])
  if step<=3:prefix.append({k:v.detach().cpu().clone()for k,v in model.state_dict().items()})
  if step%50==0:
   value=score(model,val);history.append({'step':step,'val_macro_f1':value,'base_loss':float(lb.detach()),'supplement_loss':float(ls.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 original='extra_gt'if arm=='project_extra_gt'else'aux_sam';reference=json.loads(prior.ckpt(original,seed).with_suffix('.json').read_text());assert hashes[0].hexdigest()==reference['base_indices_sha256']and hashes[1].hexdigest()==reference['supplement_indices_sha256']and hist.cpu().tolist()==reference['base_GT_class_counts']and supphist.reshape(8,26).cpu().tolist()==reference['supplement_sampling_time_class_counts']
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},tmp);tmp.replace(path);prefixpath=path.with_suffix('.prefix.pt');torch.save(prefix,prefixpath);diagnosticpath=path.with_suffix('.diagnostics.npz');save_npz(diagnosticpath,values=np.asarray(diagnostics,np.float64));d=np.asarray(diagnostics);r={'arm':arm,'seed':seed,'protocol':p,'best_step':best_step,'best_val':best,'history':history,'base_indices_sha256':hashes[0].hexdigest(),'supplement_indices_sha256':hashes[1].hexdigest(),'base_GT_class_counts':hist.cpu().tolist(),'supplement_time_class_counts':supphist.reshape(8,26).cpu().tolist(),'projection_applied_steps':int((d[:,1]<0).sum()),'post_projection_negative_dot_beyond_tolerance_steps':0,'actual_AdamW_descent_dot_base_negative_steps':int((d[:,7]<0).sum()),'diagnostic_sha256':file_hash(diagnosticpath),'prefix_sha256':file_hash(prefixpath),'parameter_count':19994,'checkpoint_sha256':file_hash(path)};atomic_json(report,r);print('trained projection',arm,seed,best_step,best,'projected',r['projection_applied_steps'],'actual adverse',r['actual_AdamW_descent_dot_base_negative_steps'],flush=True);del model,bx,sx;gc.collect();torch.cuda.empty_cache();return r

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
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed projection control exists')
  torch.backends.cuda.matmul.allow_tf32=False;m,p,old=freeze();base,later=prior.pools(m);val=prior.prior.prior.prior.read('val',m);reports={a:[train(a,seed,base,later,val,p)for seed in range(3)]for a in NEW};offsets=dict(p['fixed_offsets']);curves={};results={}
  for arm in ARMS:
   outputs=[infer(arm,seed,val)for seed in range(3)]
   if arm in NEW:offsets[arm],curves[arm]=pick_offset(outputs,p['calibration_grid'])
   runs=[{'seed':seed,'raw_validation':evaluate(outputs[seed],0),'calibrated_validation':evaluate(outputs[seed],offsets[arm])}for seed in range(3)]
   if arm not in NEW:assert runs==old['results'][arm]['runs']
   results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))};print('projection validation',arm,results[arm]['mean_calibrated_validation_macro'],flush=True)
  contrasts={}
  for left,right in [('project_sam','aux_sam'),('project_extra_gt','extra_gt'),('project_sam','project_extra_gt'),('project_sam','gt32')]:
   values=[results[left]['runs'][seed]['calibrated_validation']['macro_f1_present']-results[right]['runs'][seed]['calibrated_validation']['macro_f1_present']for seed in range(3)];contrasts[left+'_minus_'+right]={'per_seed_validation_macro_difference':values,'mean':float(np.mean(values))}
  selected=max(ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro']);selection={'validation_selected_arm':selected,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{a}_{seed}':file_hash(ckpt(a,seed))for a in ARMS for seed in range(3)},'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'validation_contrasts':contrasts,'validation_selected_arm':selected,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE GT-priority projection control',selected,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
