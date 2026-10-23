"""Matched classifier-only optimizer control after the observed background collapse."""
import fcntl,gc,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from semantic.model import SemanticTail,SemanticHead
from semantic.run import atomic_json,classification_metrics
from semantic import d4rt_finetune_readiness as ready
from semantic import d4rt_controlled_finetune as old
from tools.load_opend4rt import load_model
ROOT=Path('semantic_artifacts/d4rt_head_optimization');RATES={'low':1e-4,'high':1e-3}

def load(split):
 manifest=json.loads(ready.MANIFEST.read_text());out={}
 for c in sorted(manifest[split],key=lambda r:int(r['video_id'])):
  vid=int(c['video_id'])
  with np.load(ready.CACHE/split/f'{vid}.npz')as a:mask=a['xy'][:,2]==0 if split=='train'else np.ones(len(a['labels']),bool);r={'q':torch.from_numpy(a['prelast'].copy()).cuda().to(torch.bfloat16),'m':torch.from_numpy(a['memory'].copy()).cuda().to(torch.bfloat16)[None],'y':a['labels'].astype(np.int64),'mask':mask}
  out[vid]=r
 return out

def head_from(state):
 h=SemanticHead(num_classes=26).cuda()
 h.load_state_dict({n.removeprefix('semantic_head.'):v for n,v in state.items()if n.startswith('semantic_head.')},strict=True);return h

def train_head(x,y,initial,lr):
 head=head_from(initial);opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.01);trace=[]
 for step in range(500):
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=head(x[step][None])[0,:64].float()
  loss=F.cross_entropy(logits,y[step,:64]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();trace.append(float(loss.detach()))
 return head,trace

def describe(z,y):
 pred=z.argmax(1);fg=y>0;bg=y==0
 return {'metrics':classification_metrics(pred,y,26),'foreground_correct_fraction':float(np.mean(pred[fg]==y[fg])),'predicted_foreground_fraction':float(np.mean(pred>0)),'background_false_positive_fraction':float(np.mean(pred[bg]>0))}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists()or(ROOT/'protocol.json').exists():raise RuntimeError('Existing run; inspect before resume')
  audit=Path('semantic_artifacts/d4rt_training_fit_audit');v=json.loads((audit/'verification.json').read_text());assert v['summary_sha256']==ready.sha(audit/'summary.json');pub=json.loads((audit/'publication.log').read_text().splitlines()[-1]);assert pub['research_commit']
  prior=json.loads((old.ROOT/'protocol.json').read_text());files=[audit/'summary.json',audit/'verification.json',old.ROOT/'protocol.json',ready.CHECKPOINT,ready.CONFIG,ready.MANIFEST,Path('semantic/model.py'),Path('semantic/d4rt_head_optimization.py'),Path('semantic/OVIS_D4RT_HEAD_OPTIMIZATION_PROTOCOL.md')]+list((ready.CACHE/'train').glob('*.npz'))+list((ready.CACHE/'val').glob('*.npz'))+[old.ROOT/f'initial_seed{i}.pt'for i in range(3)]+[old.ROOT/f'A_seed{i}.pt'for i in range(3)]+[old.ROOT/f'A_seed{i}_validation.npz'for i in range(3)]
  p={'round':101,'source_training_fit_commit':pub['research_commit'],'steps':500,'rates':RATES,'input_sha256':{str(f):ready.sha(f)for f in files},'max_old_A_parameter_abs_error_allowed':1e-7,'old_validation_must_match_exactly':True,'new_confirmation':False};atomic_json(ROOT/'protocol.json',p)
  data=load('train');published=load_model(ready.CHECKPOINT,ready.CONFIG);results={key:{}for key in RATES}
  for seed in range(3):
   torch.manual_seed(seed);tail=SemanticTail(published,26).cuda().eval().requires_grad_(False);initial=torch.load(old.ROOT/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
   with torch.no_grad():
    for n,q in tail.named_parameters():
     if n in initial:q.copy_(initial[n].to(q.device))
   captured=[];hook=tail.out_norm.register_forward_hook(lambda m,inp,out:captured.append(out.detach()))
   features=[];labels=[]
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for row in prior['schedules'][str(seed)]:
     r=data[row['GT_video']];idx=row['GT_indices']+row['extra_GT_indices'];tail(r['q'][idx][None],r['m']);features.append(captured.pop()[0].float().cpu().numpy().copy());labels.append(r['y'][idx])
   hook.remove();x=np.stack(features);y=np.stack(labels);np.savez_compressed(ROOT/f'seed{seed}_schedule_features.npz',features=x,labels=y);x=torch.from_numpy(x).cuda();y=torch.from_numpy(y).cuda()
   for key,lr in RATES.items():
    head,trace=train_head(x,y,initial,lr);state={n:q.detach().cpu().clone()for n,q in head.state_dict().items()};torch.save(state,ROOT/f'{key}_seed{seed}.pt');row={'learning_rate':lr,'loss_trace':trace}
    if key=='low':
     old_a=torch.load(old.ROOT/f'A_seed{seed}.pt',map_location='cpu',weights_only=True);error=max(float((state[n]-old_a['semantic_head.'+n]).abs().max())for n in state);row['old_A_parameter_max_abs_error']=error;assert error<=1e-7,('old_A_replay_failed',seed,error)
    results[key][str(seed)]=row;print('trained head',seed,key,'final CE',trace[-1],flush=True);del head;gc.collect();torch.cuda.empty_cache()
   del tail,x,y,features;gc.collect();torch.cuda.empty_cache()
  atomic_json(ROOT/'frozen_heads.json',{'heads_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*.pt')},'rule':'all six final500 heads before train/validation scoring'})
  for split in ('train','val'):
   if split=='val':del data;gc.collect();torch.cuda.empty_cache();data=load('val')
   # Geometry tail identical across heads/seeds; initial zero-LoRA is frozen.
   torch.manual_seed(0);tail=SemanticTail(published,26).cuda().eval().requires_grad_(False);initial=torch.load(old.ROOT/'initial_seed0.pt',map_location='cpu',weights_only=True)
   with torch.no_grad():
    for n,q in tail.named_parameters():
     if n in initial:q.copy_(initial[n].to(q.device))
   captured=[];hook=tail.out_norm.register_forward_hook(lambda m,inp,out:captured.append(out.detach()));features=[];labels=[];ids=[]
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for vid,r in data.items():
     chosen=np.flatnonzero(r['mask']);q=r['q'][chosen]
     for start in range(0,len(chosen),64):tail(q[start:start+64][None],r['m']);features.append(captured.pop()[0].float().cpu().numpy().copy())
     labels.append(r['y'][chosen]);ids.extend([vid]*len(chosen))
   hook.remove();x=np.concatenate(features);y=np.concatenate(labels);np.savez_compressed(ROOT/f'{split}_evaluation_features.npz',features=x,labels=y,video_ids=np.asarray(ids));tx=torch.from_numpy(x).cuda()
   for seed in range(3):
    for key in RATES:
     head=SemanticHead(num_classes=26).cuda().eval();head.load_state_dict(torch.load(ROOT/f'{key}_seed{seed}.pt',map_location='cpu',weights_only=True));pieces=[]
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
      for start in range(0,len(y),64):pieces.append(head(tx[start:start+64]).float().cpu().numpy())
     z=np.concatenate(pieces);np.savez_compressed(ROOT/f'{key}_seed{seed}_{split}.npz',logits=z,labels=y,video_ids=np.asarray(ids));results[key][str(seed)][split]=describe(z,y);results[key][str(seed)][split]['per_video_macro']={str(int(v)):classification_metrics(z[np.asarray(ids)==v].argmax(1),y[np.asarray(ids)==v],26)['macro_f1_present']for v in sorted(set(ids))}
     if split=='val'and key=='low':
      with np.load(old.ROOT/f'A_seed{seed}_validation.npz')as a:assert np.array_equal(z,a['logits'])and np.array_equal(y,a['labels'])
     print('evaluate',split,seed,key,results[key][str(seed)][split]['metrics']['macro_f1_present'],flush=True);del head
   del tail,tx,features;gc.collect();torch.cuda.empty_cache()
  differences=[]
  for seed in range(3):
   lo=results['low'][str(seed)];hi=results['high'][str(seed)];differences.append({'seed':seed,'train_FG_macro_difference':hi['train']['metrics']['foreground_macro_f1_present']-lo['train']['metrics']['foreground_macro_f1_present'],'val_macro_difference':hi['val']['metrics']['macro_f1_present']-lo['val']['metrics']['macro_f1_present'],'val_video_equal_difference':float(np.mean([hi['val']['per_video_macro'][v]-lo['val']['per_video_macro'][v]for v in lo['val']['per_video_macro']]))})
  qualified=all(r['train_FG_macro_difference']>0 and r['val_macro_difference']>0 for r in differences)
  for path,h in p['input_sha256'].items():assert ready.sha(path)==h
  atomic_json(ROOT/'summary.json',{'round':101,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'results':results,'high_minus_low':differences,'head_lr1e3_eligible_for_next_matched_LoRA_control':qualified,'default_changed':False,'new_confirmation':False,'outputs_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*.npz')}|{str(f):ready.sha(f)for f in ROOT.glob('*.pt')}});print('DONE head optimizer control; next matched-control eligibility',qualified,flush=True)
if __name__=='__main__':main()
