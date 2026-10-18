"""Factorized linear versus ReLU, same fixed DINO input and exposure budget."""
import fcntl,gc,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import feature_standardization_control as prior
from semantic import dino_register_training as candidates
from semantic.nonlinear_query_head import TwoLayerQueryHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz
from semantic.run import atomic_json
from semantic.oracle_probe import file_hash
ROOT=Path('semantic_artifacts/dino_nonlinear_head');REFERENCE='z_b14_gt32';NEW=('factor32','relu32');ARMS=(REFERENCE,)+NEW;SPLITS=prior.SPLITS

def ckpt(arm,seed):return prior.ckpt(REFERENCE,seed)if arm==REFERENCE else ROOT/arm/'runs'/f'linear_seed{seed}.pt'

def freeze():
 source=json.loads((prior.ROOT/'summary.json').read_text());assert file_hash(prior.ROOT/'summary.json')==json.loads((prior.ROOT/'verification.json').read_text())['summary_sha256'];cs=json.loads((candidates.ROOT/'summary.json').read_text());assert file_hash(candidates.ROOT/'summary.json')==json.loads((candidates.ROOT/'verification.json').read_text())['summary_sha256']
 future=candidates.ROOT/'future_confirmation_manifest.json';assert file_hash(future)==cs['protocol']['future_confirmation_manifest_sha256'];atomic_json(ROOT/'future_confirmation_manifest.json',json.loads(future.read_text()));m=json.loads((prior.ROOT/'manifest.json').read_text())
 paths=[Path(x)for x in source['protocol']['input_sha256']]+[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'statistics_manifest.json',Path(source['statistics']['b14_gt32']['path']),candidates.ROOT/'summary.json',candidates.ROOT/'verification.json',future,Path('semantic/nonlinear_query_head.py'),Path('semantic/dino_nonlinear_head.py'),Path('semantic/OVIS_DINO_NONLINEAR_HEAD_PROTOCOL.md'),Path('tests/test_nonlinear_query_head.py')]
 for seed in range(3):paths.extend([ckpt(REFERENCE,seed),ckpt(REFERENCE,seed).with_suffix('.json')]);paths.extend(prior.ROOT/REFERENCE/'logits'/f'seed{seed}_{split}.npz'for split in SPLITS)
 test=subprocess.run(['.venv/bin/python','tests/test_nonlinear_query_head.py'],capture_output=True,text=True,check=True);(ROOT/'unit_checks.log').write_text(test.stdout+test.stderr);assert 'PASS'in test.stdout
 p={'round':91,'new_heads':6,'new_head_parameters':25466,'reference_head_parameters':19994,'hidden_width':32,'steps':2000,'batch':32,'exposures':64000,'statistics':source['statistics']['b14_gt32'],'fixed_reference_offsets':source['results'][REFERENCE]['offsets'],'calibration_grid':source['protocol']['calibration_grid'],'future_confirmation_manifest_sha256':file_hash(future),'unit_checks_sha256':file_hash(ROOT/'unit_checks.log'),'source_validation_only':True,'input_sha256':{str(path):file_hash(path)for path in sorted(set(paths))}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p

def train(arm,seed,base,val,p):
 path=ckpt(arm,seed);report=path.with_suffix('.json')
 if report.exists():
  r=json.loads(report.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=TwoLayerQueryHead(arm=='relu32').cuda();initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};x=torch.from_numpy(np.concatenate([a['decoded']for a in base])).cuda();labels=np.concatenate([a['labels']for a in base]);times=np.concatenate([a['xy'][:,2]for a in base]);assert len(labels)==20480 and (times==0).all()and x.dtype==torch.float32
 sampler=Sampler(labels,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);ih=hashlib.sha256();hist=torch.zeros(26,dtype=torch.long,device='cuda');history=[];best=-1
 for step in range(1,2001):
  fg,bg=sampler.draw(gen,count=16);ix=torch.cat([fg,bg]);ih.update(ix.cpu().numpy().tobytes());hist+=torch.bincount(sampler.labels[ix],minlength=26);model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ix]),sampler.labels[ix]);assert torch.isfinite(loss);loss.backward();optimizer.step()
  if step%50==0:
   value=score(model,val);history.append({'step':step,'balanced_raw_val_macro':value,'loss':float(loss.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 ref=json.loads(ckpt(REFERENCE,seed).with_suffix('.json').read_text());assert ih.hexdigest()==ref['sampled_indices_sha256']and hist.cpu().tolist()==ref['sampled_class_counts']
 path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix('.partial.pt');torch.save({'model':state,'initial_state':initial,'nonlinear':arm=='relu32','seed':seed,'steps':2000,'best_step':best_step,'protocol':p},partial);partial.replace(path)
 r={'arm':arm,'seed':seed,'protocol':p,'head_parameters':25466,'best_step':best_step,'best_balanced_raw_val':best,'history':history,'sampled_indices_sha256':ih.hexdigest(),'sampled_class_counts':hist.cpu().tolist(),'initial_parameter_sha256':{k:hashlib.sha256(v.numpy().tobytes()).hexdigest()for k,v in initial.items()},'checkpoint_sha256':file_hash(path)};atomic_json(report,r);print('trained fixed head',arm,seed,best_step,best,flush=True);del model,x;gc.collect();torch.cuda.empty_cache();return r

def infer(arm,seed,split,data):
 path=ROOT/arm/'logits'/f'seed{seed}_{split}.npz';h=file_hash(ckpt(arm,seed))
 if path.exists():
  with np.load(path)as a:out={k:a[k].copy()for k in a.files}
  assert str(out['checkpoint_sha256'])==h;return out
 if arm==REFERENCE:
  with np.load(prior.ROOT/REFERENCE/'logits'/f'seed{seed}_{split}.npz')as a:out={k:a[k].copy()for k in a.files}
 else:
  model=TwoLayerQueryHead(arm=='relu32').cuda().eval();model.load_state_dict(torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model'])
  with torch.inference_mode():values=[model(torch.from_numpy(a['decoded']).cuda()).cpu().numpy()for a in data]
  out={'logits':np.concatenate(values),'labels':np.concatenate([a['labels']for a in data]),'times':np.concatenate([a['xy'][:,2]for a in data]),'video_ids':np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in data]),'checkpoint_sha256':np.array(h)};del model
 save_npz(path,**out);return out

def folded(seed,data,expected):
 state=torch.load(ckpt('factor32',seed),map_location='cpu',weights_only=True)['model'];a=state['input.weight'].cuda();b=state['input.bias'].cuda();w=state['output.weight'].cuda();v=state['output.bias'].cuda();weight=w@a;bias=w@b+v
 with torch.inference_mode():values=np.concatenate([F.linear(torch.from_numpy(row['decoded']).cuda(),weight,bias).cpu().numpy()for row in data])
 maximum=float(np.abs(values-expected).max());tolerance=2e-4+2e-5*float(np.abs(expected).max());assert maximum<=tolerance
 return {'maximum_folded_FP32_logit_difference':maximum,'preregistered_tolerance':tolerance,'label_changes_due_to_numerics':int(np.sum(values.argmax(-1)!=expected.argmax(-1)))}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed matched-head control exists')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  m,p=freeze();stats={'b14_gt32':p['statistics']};base=prior.read(REFERENCE,'train',m,stats);data={split:prior.read(REFERENCE,split,m,stats)for split in SPLITS};reports={};results={};curves={};offsets={};folds=[]
  for arm in ARMS:
   if arm in NEW:reports[arm]=[train(arm,seed,base,data['balanced_val'],p)for seed in range(3)]
   outputs={split:[infer(arm,seed,split,data[split])for seed in range(3)]for split in SPLITS};offsets[arm]={'raw':0.};curves[arm]={}
   for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
    if arm==REFERENCE:offsets[arm][profile]=p['fixed_reference_offsets'][profile]
    else:offsets[arm][profile],curves[arm][profile]=pick_offset(outputs[split],p['calibration_grid'])
   runs=[{'seed':seed,**{split:{profile:evaluate(outputs[split][seed],offset)for profile,offset in offsets[arm].items()}for split in SPLITS}}for seed in range(3)];results[arm]={'offsets':offsets[arm],'runs':runs,'mean_balanced_profile_macro':float(np.mean([a['balanced_val']['balanced_profile']['macro_f1_present']for a in runs]))}
   if arm=='factor32':folds=[folded(seed,data['balanced_val'],outputs['balanced_val'][seed]['logits'])for seed in range(3)]
   print('matched head source score',arm,results[arm]['mean_balanced_profile_macro'],flush=True)
  for seed in range(3):assert reports['factor32'][seed]['initial_parameter_sha256']==reports['relu32'][seed]['initial_parameter_sha256']
  pairs=[('relu32','factor32'),('factor32',REFERENCE),('relu32',REFERENCE)];contrasts={left+'_minus_'+right:prior.prior.contrasts(results,left,right)for left,right in pairs};selected=max(ARMS,key=lambda a:results[a]['mean_balanced_profile_macro']);eligible=selected in NEW and min(contrasts[selected+'_minus_'+REFERENCE]['balanced_val']['balanced_profile']['per_seed_macro_difference'])>0
  selection={'validation_selected_arm':selected,'new_arm_eligible_for_separate_confirmation':eligible,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{arm}_{seed}':file_hash(ckpt(arm,seed))for arm in ARMS for seed in range(3)},'future_confirmation_manifest_sha256':p['future_confirmation_manifest_sha256'],'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'contrasts':contrasts,'factorized_affine_equivalence':folds,'validation_selected_arm':selected,'new_arm_eligible_for_separate_confirmation':eligible,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE matched linear-versus-ReLU head control',selected,eligible,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
