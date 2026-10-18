"""Source-only B14 auxiliary labels with fixed original GT32 anchor batches."""
import fcntl,gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_b14_anchor_oracle as prior
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_b14_auxiliary');ARMS=('gt32','extra_gt','aux_sam','aux_oracle','weak');NEW=ARMS[1:4];ELIGIBLE=('gt32','extra_gt','aux_sam','weak')
def ckpt(arm,seed):
 if arm in('gt32','weak'):return prior.ckpt(arm,seed)
 return ROOT/arm/'runs'/f'linear_seed{seed}.pt'
def freeze():
 old=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text());assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256'];m=json.loads((prior.ROOT/'manifest.json').read_text())
 paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'manifest.json',prior.ROOT/'frozen_selection.json',Path('semantic/dino_b14_auxiliary.py'),Path('semantic/OVIS_DINO_B14_AUXILIARY_PROTOCOL.md'),Path('semantic/sampling_ablation.py'),Path('semantic/head_budget_probe.py'),Path('semantic/model.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py')]
 for c in m['train']:paths.extend([prior.prior.fp(c['video_id']),prior.oracle.ROOT/'labels'/f'{c["video_id"]}.npz',prior.prior.prior.fp('train',c['video_id'])])
 for c in m['val']:paths.append(prior.prior.prior.fp('val',c['video_id']))
 for arm in('gt32','weak'):
  for seed in range(3):paths.extend([ckpt(arm,seed),ckpt(arm,seed).with_suffix('.json')])
 p={'round':71,'arms':list(ARMS),'new_arms':list(NEW),'eligible_candidates':list(ELIGIBLE),'oracle_diagnostic_only':True,'steps':2000,'base_batch':32,'supplement_batch':32,'total_exposures':128000,'base_GT_exposures':64000,'supplement_seed_base':20261071,'head_parameters':19994,'loss':'CE_base.mean + CE_supp.mean, fixed lambda1','calibration_grid':old['protocol']['calibration_grid'],'fixed_offsets':{'gt32':old['results']['gt32']['offset'],'weak':old['results']['weak']['offset']},'old_native_teacher_validation_mean':old['results']['weak']['mean_calibrated_validation_macro'],'current_global_validation_macro':old['protocol']['current_global_validation_macro'],'validation_only':True,'input_sha256':{str(x):file_hash(x)for x in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p,old

def pools(m):
 base=prior.prior.prior.read('train',m);whole=prior.read(m);later=[]
 for a,b in zip(whole,base):
  assert int(a['video_id'])==int(b['video_id']);source=a['xy'][:,2]==0
  assert np.array_equal(a['decoded'][source],b['decoded'])and np.array_equal(a['labels'][source],b['labels'])and np.array_equal(a['xy'][source],b['xy'])
  rows=~source;later.append({k:a[k][rows]for k in ('decoded','labels','xy','truth')})
 assert sum(len(a['labels'])for a in base)==20480 and sum(len(a['labels'])for a in later)==142080
 return base,later

def train(arm,seed,base,later,val,p):
 path=ckpt(arm,seed);rp=path.with_suffix('.json')
 if rp.exists():
  r=json.loads(rp.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();bx=torch.from_numpy(np.concatenate([a['decoded']for a in base])).float().cuda();by=np.concatenate([a['labels']for a in base]);bt=np.concatenate([a['xy'][:,2]for a in base]);base_sampler=Sampler(by,bt,'baseline');base_gen=torch.Generator(device='cuda').manual_seed(seed)
 data=base if arm=='extra_gt'else later;sx=torch.from_numpy(np.concatenate([a['decoded']for a in data])).float().cuda();sy=np.concatenate([a['labels']for a in data]);st=np.concatenate([a['xy'][:,2]for a in data]);target=torch.tensor(np.concatenate([a['truth']for a in data])if arm=='aux_oracle'else sy,dtype=torch.long,device='cuda');supp_sampler=Sampler(sy,st,'baseline');supp_gen=torch.Generator(device='cuda').manual_seed(20261071+seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 basehash=hashlib.sha256();supphash=hashlib.sha256();basehist=torch.zeros(26,dtype=torch.long,device='cuda');supphist=torch.zeros(208,dtype=torch.long,device='cuda');targethist=torch.zeros(26,dtype=torch.long,device='cuda');history=[];best=-1
 for step in range(1,2001):
  fg,bg=base_sampler.draw(base_gen,count=16);bi=torch.cat([fg,bg]);fg,bg=supp_sampler.draw(supp_gen,count=16);si=torch.cat([fg,bg]);basehash.update(bi.cpu().numpy().tobytes());supphash.update(si.cpu().numpy().tobytes());basehist+=torch.bincount(base_sampler.labels[bi],minlength=26);supphist+=torch.bincount(supp_sampler.times[si]*26+supp_sampler.labels[si],minlength=208);targethist+=torch.bincount(target[si],minlength=26)
  model.train();optimizer.zero_grad(set_to_none=True);lb=F.cross_entropy(model(bx[bi]),base_sampler.labels[bi]);ls=F.cross_entropy(model(sx[si]),target[si]);loss=lb+ls;loss.backward();optimizer.step()
  if step%50==0:
   value=score(model,val);history.append({'step':step,'val_macro_f1':value,'base_loss':float(lb.detach()),'supplement_loss':float(ls.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 reference=json.loads(ckpt('gt32',seed).with_suffix('.json').read_text());assert basehash.hexdigest()==reference['sampled_indices_sha256']and basehist.cpu().tolist()==reference['sampled_class_counts']and int(basehist.sum())==64000 and int(supphist.sum())==64000
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},tmp);tmp.replace(path)
 r={'arm':arm,'seed':seed,'protocol':p,'best_step':best_step,'best_val':best,'history':history,'base_indices_sha256':basehash.hexdigest(),'supplement_indices_sha256':supphash.hexdigest(),'supplement_pool':'source_GT'if arm=='extra_gt'else'later_teacher','base_GT_class_counts':basehist.cpu().tolist(),'supplement_sampling_time_class_counts':supphist.reshape(8,26).cpu().tolist(),'supplement_actual_target_counts':targethist.cpu().tolist(),'base_exposures':64000,'supplement_exposures':64000,'total_exposures':128000,'parameter_count':19994,'checkpoint_sha256':file_hash(path)};atomic_json(rp,r);print('trained auxiliary',arm,seed,best_step,best,flush=True);del model,bx,sx,target;gc.collect();torch.cuda.empty_cache();return r

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
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed B14 auxiliary control exists')
  torch.backends.cuda.matmul.allow_tf32=False;m,p,old=freeze();base,later=pools(m);val=prior.prior.prior.read('val',m);reports={a:[train(a,seed,base,later,val,p)for seed in range(3)]for a in NEW}
  for seed in range(3):
   a=reports['aux_sam'][seed];b=reports['aux_oracle'][seed]
   for k in ('base_indices_sha256','base_GT_class_counts','supplement_indices_sha256','supplement_sampling_time_class_counts'):assert a[k]==b[k]
  offsets=dict(p['fixed_offsets']);curves={};results={}
  for arm in ARMS:
   outputs=[infer(arm,seed,val)for seed in range(3)]
   if arm in NEW:offsets[arm],curves[arm]=pick_offset(outputs,p['calibration_grid'])
   runs=[{'seed':seed,'raw_validation':evaluate(outputs[seed],0),'calibrated_validation':evaluate(outputs[seed],offsets[arm])}for seed in range(3)]
   if arm in('weak','gt32'):assert runs==old['results'][arm]['runs']
   results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))};print('auxiliary validation',arm,results[arm]['mean_calibrated_validation_macro'],flush=True)
  contrasts={}
  for left,right in [('aux_sam','extra_gt'),('aux_oracle','aux_sam'),('aux_sam','gt32')]:
   vals=[results[left]['runs'][seed]['calibrated_validation']['macro_f1_present']-results[right]['runs'][seed]['calibrated_validation']['macro_f1_present']for seed in range(3)];contrasts[left+'_minus_'+right]={'per_seed_validation_macro_difference':vals,'mean':float(np.mean(vals))}
  selected=max(ELIGIBLE,key=lambda a:results[a]['mean_calibrated_validation_macro']);teacher=max(('weak','aux_sam'),key=lambda a:results[a]['mean_calibrated_validation_macro']);selection={'eligible_candidates':list(ELIGIBLE),'oracle_excluded_from_selection':True,'validation_selected_arm':selected,'validation_selected_teacher_recipe':teacher,'teacher_matched_GT_reference':'extra_gt'if teacher=='aux_sam'else'B14_GT256_from_round58','offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{a}_{seed}':file_hash(ckpt(a,seed))for a in ARMS for seed in range(3)},'exceeds_previous_global_validation':results[selected]['mean_calibrated_validation_macro']>p['current_global_validation_macro'],'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'validation_contrasts':contrasts,'eligible_validation_choice':selected,'validation_selected_teacher_recipe':teacher,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'oracle_deployable':False,'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE B14 GT-anchor auxiliary controls',selected,teacher,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
