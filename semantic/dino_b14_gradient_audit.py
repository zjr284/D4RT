"""Static CE-gradient alignment; not an AdamW trajectory or new confirmation."""
import fcntl,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_b14_auxiliary as prior
from semantic.sampling_ablation import Sampler
from semantic.semantic_feature_control import save_npz
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_b14_gradient_audit');STATES=('gt32','extra_gt','aux_sam');CONDITIONS=('base_GT','extra_GT','later_SAM','later_oracle');N=200

def freeze():
 s=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text());assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256'];m=json.loads((prior.ROOT/'manifest.json').read_text());paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'manifest.json',prior.ROOT/'frozen_selection.json',Path('semantic/dino_b14_gradient_audit.py'),Path('semantic/OVIS_DINO_B14_GRADIENT_PROTOCOL.md'),Path('semantic/sampling_ablation.py')]
 for state in STATES:
  for seed in range(3):paths.append(prior.ckpt(state,seed))
 for path in s['protocol']['input_sha256']:assert file_hash(path)==s['protocol']['input_sha256'][path];paths.append(Path(path))
 p={'round':74,'states':list(STATES),'conditions':list(CONDITIONS),'batch_size':32,'replayed_steps':2000,'analyzed_prefix_steps':N,'gradient_parameters':19994,'training_updates':0,'calibration_updates':0,'gradient':'raw CE.mean Euclidean gradient, no AdamW preconditioning or weight decay','input_sha256':{str(path):file_hash(path)for path in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p

def samples(base,later,seed):
 by=np.concatenate([a['labels']for a in base]);sy=np.concatenate([a['labels']for a in later]);st=np.concatenate([a['xy'][:,2]for a in later]);bs=Sampler(by,np.zeros(len(by),np.int64),'baseline');ss=Sampler(sy,st,'baseline');gens=[torch.Generator(device='cuda').manual_seed(k)for k in(seed,20261071+seed,20261071+seed)];hashes=[hashlib.sha256()for _ in range(3)];prefix=[[],[],[]]
 for step in range(2000):
  for i,(sampler,gen)in enumerate(zip((bs,bs,ss),gens)):
   f,b=sampler.draw(gen,count=16);ix=torch.cat([f,b]).cpu().numpy();hashes[i].update(ix.tobytes())
   if step<N:prefix[i].append(ix)
 reference=[json.loads(prior.ckpt(arm,seed).with_suffix('.json').read_text())for arm in ('extra_gt','aux_sam')];assert hashes[0].hexdigest()==reference[0]['base_indices_sha256']==reference[1]['base_indices_sha256']and hashes[1].hexdigest()==reference[0]['supplement_indices_sha256']and hashes[2].hexdigest()==reference[1]['supplement_indices_sha256'];a=[np.stack(x)for x in prefix];save_npz(ROOT/'indices'/f'seed{seed}.npz',base=a[0],extra=a[1],teacher=a[2]);return a,[h.hexdigest()for h in hashes]

def gradient(x,y,weight,bias):
 logits=F.linear(x,weight,bias);prob=logits.softmax(-1);error=prob-F.one_hot(y,26);error/=len(y);g=torch.cat([(error.T@x).reshape(-1),error.sum(0)]);return g,F.cross_entropy(logits,y)

def aggregate(a):
 result={'base_mean_loss':float(a['losses'][:,0].mean()),'base_mean_gradient_norm':float(a['norms'][:,0].mean()),'oracle_SAM_mean_relative_gradient_change':float(a['oracle_SAM_relative_gradient_change'].mean())};mean=a['mean_gradients'];result['supplement_vs_base']={}
 for j,name in enumerate(CONDITIONS[1:]):
  b=mean[0];c=mean[j+1];result['supplement_vs_base'][name]={'mean_loss':float(a['losses'][:,j+1].mean()),'mean_batch_cosine':float(a['cosines'][:,j].mean()),'negative_dot_fraction':float((a['dots'][:,j]<0).mean()),'mean_supplement_to_base_norm_ratio':float(a['ratios'][:,j].mean()),'supplement_norm_exceeds_base_fraction':float((a['norms'][:,j+1]>a['norms'][:,0]).mean()),'cosine_of_mean_gradients':float(np.dot(b,c)/max(float(np.linalg.norm(b)*np.linalg.norm(c)),1e-12))}
 return result

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed gradient audit exists')
  torch.backends.cuda.matmul.allow_tf32=False;m,p=freeze();base,later=prior.pools(m);bx=torch.from_numpy(np.concatenate([a['decoded']for a in base])).float().cuda();by=torch.tensor(np.concatenate([a['labels']for a in base]),device='cuda',dtype=torch.long);sx=torch.from_numpy(np.concatenate([a['decoded']for a in later])).float().cuda();sy=torch.tensor(np.concatenate([a['labels']for a in later]),device='cuda',dtype=torch.long);truth=torch.tensor(np.concatenate([a['truth']for a in later]),device='cuda',dtype=torch.long);results={};samplers={}
  for seed in range(3):
   ix,hashes=samples(base,later,seed);samplers[str(seed)]={'full_2000_step_hashes':hashes,'analyzed_teacher_target_changes':int((sy[torch.from_numpy(ix[2]).cuda()]!=truth[torch.from_numpy(ix[2]).cuda()]).sum())};indices=[torch.from_numpy(a).cuda()for a in ix]
   for state in STATES:
    checkpoint=torch.load(prior.ckpt(state,seed),map_location='cpu',weights_only=True)['model'];weight=checkpoint['classifier.weight'].cuda();bias=checkpoint['classifier.bias'].cuda();losses=[];norms=[];cos=[];dots=[];ratios=[];relative=[];means=torch.zeros(4,19994,device='cuda')
    with torch.inference_mode():
     for step in range(N):
      bi,ei,ti=(a[step]for a in indices);sets=((bx[bi],by[bi]),(bx[ei],by[ei]),(sx[ti],sy[ti]),(sx[ti],truth[ti]));out=[gradient(x,y,weight,bias)for x,y in sets];g=torch.stack([a[0]for a in out]);n=torch.linalg.vector_norm(g,dim=1);d=(g[1:]*g[0]).sum(1);means+=g/N;losses.append(torch.stack([a[1]for a in out]).cpu().numpy());norms.append(n.cpu().numpy());dots.append(d.cpu().numpy());cos.append((d/(n[1:]*n[0]).clamp_min(1e-12)).cpu().numpy());ratios.append((n[1:]/n[0].clamp_min(1e-12)).cpu().numpy());relative.append(float(torch.linalg.vector_norm(g[3]-g[2])/n[2].clamp_min(1e-12)))
    arrays={key:np.asarray(value)for key,value in [('losses',losses),('norms',norms),('cosines',cos),('dots',dots),('ratios',ratios),('oracle_SAM_relative_gradient_change',relative)]};arrays['mean_gradients']=means.cpu().numpy();path=ROOT/'gradients'/f'{state}_{seed}.npz';save_npz(path,**arrays);results[f'{state}_{seed}']={'state':state,'seed':seed,'statistics':aggregate(arrays),'output_path':str(path),'output_sha256':file_hash(path)};print('gradient audit',state,seed,results[f'{state}_{seed}']['statistics'],flush=True)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'sampler_replays':samplers,'results':results,'new_confirmation_groups':0,'deployment_unchanged':'native_gt','optimizer_causal_claim':False});print('DONE frozen B14 gradient audit',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
