"""Fixed B14 teacher indices: active/total source loss and diagnostic GT oracle."""
import fcntl,gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_b14_sam_control as prior
from semantic import dino_label_oracle as oracle
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_b14_anchor_oracle');ARMS=('weak','source_active','source_total','oracle','gt32');NEW=ARMS[1:4];ELIGIBLE=('weak','source_active','source_total','gt32')
def ckpt(arm,seed):
 if arm=='weak':return prior.ckpt('B14_sam',seed)
 if arm=='gt32':return prior.prior.ckpt('B14_gt32',seed)
 return ROOT/arm/'runs'/f'linear_seed{seed}.pt'
def freeze():
 old=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text());assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256'];m=json.loads((prior.ROOT/'manifest.json').read_text());o=json.loads((oracle.ROOT/'summary.json').read_text());ov=json.loads((oracle.ROOT/'verification.json').read_text());assert file_hash(oracle.ROOT/'summary.json')==ov['summary_sha256']
 byid={c['video_id']:c for c in o['manifest']['train']};paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'manifest.json',oracle.ROOT/'summary.json',oracle.ROOT/'verification.json',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic/dino_b14_anchor_oracle.py'),Path('semantic/OVIS_DINO_B14_ANCHOR_ORACLE_PROTOCOL.md'),Path('semantic/sampling_ablation.py'),Path('semantic/model.py'),Path('semantic/head_budget_probe.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py')]
 for c in m['train']:
  assert c['frames']==byid[c['video_id']]['frames'];path=oracle.ROOT/'labels'/f'{c["video_id"]}.npz';assert file_hash(path)==o['corrected_label_sha256'][str(path)];paths.extend([path,prior.fp(c['video_id'])])
 for c in m['val']:paths.append(prior.prior.fp('val',c['video_id']))
 for arm in ('weak','gt32'):
  for seed in range(3):paths.extend([ckpt(arm,seed),ckpt(arm,seed).with_suffix('.json')])
 p={'round':70,'arms':list(ARMS),'new_arms':list(NEW),'eligible_validation_candidates':list(ELIGIBLE),'oracle_is_diagnostic_only':True,'steps':2000,'batch':256,'head_parameters':19994,'calibration_grid':old['protocol']['calibration_grid'],'fixed_offsets':{'weak':old['results']['B14_sam']['offset'],'gt32':-2.0},'current_global_validation_macro':old['protocol']['current_global_validation_macro'],'validation_only':True,'input_sha256':{str(x):file_hash(x)for x in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p,old

def read(m):
 data=prior.read_train(m)
 for a in data:
  with np.load(oracle.ROOT/'labels'/f'{int(a["video_id"])}.npz')as o:
   assert np.array_equal(a['labels'],o['original_labels'])and np.array_equal(a['xy'],o['xy']);a['truth']=o['corrected_labels'].copy()
  source=a['xy'][:,2]==0;assert np.array_equal(a['truth'][source],a['labels'][source])and np.all((a['truth']>=0)&(a['truth']<26))
 return data

def train(arm,seed,data,val,p):
 path=ckpt(arm,seed);rp=path.with_suffix('.json')
 if rp.exists():
  r=json.loads(rp.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();features=torch.from_numpy(np.concatenate([a['decoded']for a in data])).float().cuda();labels=np.concatenate([a['labels']for a in data]);times=np.concatenate([a['xy'][:,2]for a in data]);truth=np.concatenate([a['truth']for a in data]);target=torch.tensor(truth if arm=='oracle'else labels,dtype=torch.long,device='cuda');sampler=Sampler(labels,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 hist=torch.zeros(208,dtype=torch.long,device='cuda');alltarget=torch.zeros(26,dtype=torch.long,device='cuda');nonzero=alltarget.clone();ih=hashlib.sha256();nh=np.zeros(257,np.int64);dh=np.zeros(257,np.int64);best=-1;history=[]
 for step in range(1,2001):
  fg,bg=sampler.draw(gen);ix=torch.cat([fg,bg]);ih.update(ix.cpu().numpy().tobytes());active=sampler.times[ix]==0;n=int(active.sum());assert n>0;nh[n]+=1;denominator=n if arm=='source_active'else 256;dh[denominator]+=1;hist+=torch.bincount(sampler.times[ix]*26+sampler.labels[ix],minlength=208);alltarget+=torch.bincount(target[ix],minlength=26);nonzero+=torch.bincount(target[ix][active]if arm!='oracle'else target[ix],minlength=26)
  model.train();optimizer.zero_grad(set_to_none=True);losses=F.cross_entropy(model(features[ix]),target[ix],reduction='none');loss=losses.mean()if arm=='oracle'else(losses*active.float()).sum()/denominator;loss.backward();optimizer.step()
  if step%50==0:
   value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 reference=json.loads(ckpt('weak',seed).with_suffix('.json').read_text());counts=hist.reshape(8,26).cpu().tolist();assert ih.hexdigest()==reference['sampled_indices_sha256']and counts==reference['sampled_time_class_counts']
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},tmp);tmp.replace(path)
 r={'arm':arm,'seed':seed,'protocol':p,'best_step':best_step,'best_val':best,'history':history,'sampled_indices_sha256':ih.hexdigest(),'teacher_time_class_counts':counts,'all_sampled_loss_target_counts':alltarget.cpu().tolist(),'nonzero_loss_target_counts':nonzero.cpu().tolist(),'active_anchor_count_histogram':nh.tolist(),'loss_denominator_histogram':dh.tolist(),'GT_anchor_exposures':int(np.dot(np.arange(257),nh)),'nonzero_loss_exposures':int(nonzero.sum()),'parameter_count':19994,'checkpoint_sha256':file_hash(path)};atomic_json(rp,r);print('trained anchor/oracle',arm,seed,best_step,best,flush=True);del features,model,target;gc.collect();torch.cuda.empty_cache();return r

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
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed anchor/oracle control exists')
  torch.backends.cuda.matmul.allow_tf32=False;m,p,old=freeze();data=read(m);val=prior.prior.read('val',m);reports={a:[train(a,seed,data,val,p)for seed in range(3)]for a in NEW};offsets=dict(p['fixed_offsets']);curves={};results={}
  for arm in ARMS:
   outputs=[infer(arm,seed,val)for seed in range(3)]
   if arm in NEW:offsets[arm],curves[arm]=pick_offset(outputs,p['calibration_grid'])
   runs=[{'seed':seed,'raw_validation':evaluate(outputs[seed],0),'calibrated_validation':evaluate(outputs[seed],offsets[arm])}for seed in range(3)]
   reference=old['results']['B14_sam']['runs']if arm=='weak'else json.loads((prior.prior.ROOT/'summary.json').read_text())['results']['B14_gt32']['runs']if arm=='gt32'else None
   if reference is not None:assert runs==reference
   results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))};print('anchor/oracle validation',arm,results[arm]['mean_calibrated_validation_macro'],flush=True)
  contrasts={}
  for left,right in [('source_active','weak'),('source_total','source_active'),('oracle','weak')]:
   vals=[results[left]['runs'][seed]['calibrated_validation']['macro_f1_present']-results[right]['runs'][seed]['calibrated_validation']['macro_f1_present']for seed in range(3)];contrasts[left+'_minus_'+right]={'per_seed_validation_macro_difference':vals,'mean':float(np.mean(vals))}
  selected=max(ELIGIBLE,key=lambda a:results[a]['mean_calibrated_validation_macro']);selection={'eligible_candidates':list(ELIGIBLE),'oracle_excluded_from_selection':True,'validation_selected_arm':selected,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{a}_{seed}':file_hash(ckpt(a,seed))for a in ARMS for seed in range(3)},'exceeds_previous_global_validation':results[selected]['mean_calibrated_validation_macro']>p['current_global_validation_macro'],'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'validation_contrasts':contrasts,'eligible_validation_choice':selected,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'oracle_deployable':False,'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE B14 anchor/oracle source controls',selected,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
