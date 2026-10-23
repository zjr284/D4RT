"""Independent feature regeneration and full optimizer replay for two learning rates."""
import gc,json
from pathlib import Path
import numpy as np
import torch
from semantic.model import SemanticTail,SemanticHead
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha,metrics
from tools.load_opend4rt import load_model,DEFAULT_VARIANT
ROOT=Path('semantic_artifacts/d4rt_head_optimization');SOURCE=Path('semantic_artifacts/d4rt_controlled_finetune');CACHE=Path('semantic_artifacts/features_ovis/sam')

def check_stat(z,y,row):
 pred=z.argmax(1);assert metrics(y,pred)[0]==row['metrics'];fg=y>0;bg=~fg
 assert float(np.mean(pred[fg]==y[fg]))==row['foreground_correct_fraction']and float(np.mean(pred>0))==row['predicted_foreground_fraction']and float(np.mean(pred[bg]>0))==row['background_false_positive_fraction']

def load_head(state):
 h=SemanticHead(num_classes=26).cuda();h.load_state_dict(state);return h

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());assert p['rates']=={'low':.0001,'high':.001}and p['steps']==500 and sha(ROOT/'protocol.json')==s['protocol_sha256']
 for path,h in p['input_sha256'].items():assert sha(path)==h,path
 for path,h in s['outputs_sha256'].items():assert sha(path)==h,path
 frozen=json.loads((ROOT/'frozen_heads.json').read_text());assert len(frozen['heads_sha256'])==6
 for path,h in frozen['heads_sha256'].items():assert sha(path)==h
 manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text());prior=json.loads((SOURCE/'protocol.json').read_text());train={}
 for clip in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
  vid=int(clip['video_id'])
  with np.load(CACHE/'train'/f'{vid}.npz')as a:train[vid]={'q':torch.from_numpy(a['prelast'].copy()).cuda().to(torch.bfloat16),'memory':torch.from_numpy(a['memory'].copy()).cuda().to(torch.bfloat16)[None],'y':a['labels'].astype(np.int64)}
 published=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml');checks={}
 for seed in range(3):
  torch.manual_seed(seed);tail=SemanticTail(published,26).cuda().eval().requires_grad_(False);initial=torch.load(SOURCE/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
  with torch.no_grad():
   for n,q in tail.named_parameters():
    if n in initial:q.copy_(initial[n].to(q.device))
  captured=[];handle=tail.out_norm.register_forward_hook(lambda m,i,o:captured.append(o.detach()))
  with np.load(ROOT/f'seed{seed}_schedule_features.npz')as a:x=a['features'].copy();y=a['labels'].copy()
  assert x.shape==(500,96,1280)and y.shape==(500,96)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for step,row in enumerate(prior['schedules'][str(seed)]):
    r=train[row['GT_video']];idx=np.asarray(row['GT_indices']+row['extra_GT_indices']);tail(r['q'][idx][None],r['memory']);got=captured.pop()[0].float().cpu().numpy();assert np.array_equal(got,x[step])and np.array_equal(y[step],r['y'][idx])
  handle.remove();del tail;gc.collect();torch.cuda.empty_cache();tx=torch.from_numpy(x).cuda();ty=torch.from_numpy(y).cuda();init={n.removeprefix('semantic_head.'):v for n,v in initial.items()if n.startswith('semantic_head.')}
  for key,lr in p['rates'].items():
   h=load_head(init);opt=torch.optim.AdamW(h.parameters(),lr=lr,weight_decay=.01);logged=s['results'][key][str(seed)]
   for step in range(500):
    with torch.autocast('cuda',dtype=torch.bfloat16):scores=h(tx[step][None])[0,:64].float()
    log_prob=scores.log_softmax(-1);loss=-log_prob[torch.arange(64,device='cuda'),ty[step,:64]].mean();assert abs(float(loss.detach())-logged['loss_trace'][step])<2e-5;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
   saved=torch.load(ROOT/f'{key}_seed{seed}.pt',map_location='cpu',weights_only=True);error=0.
   for n,q in h.state_dict().items():
    actual=q.cpu();torch.testing.assert_close(actual,saved[n],rtol=5e-4,atol=2e-6);error=max(error,float((actual-saved[n]).abs().max()))
   if key=='low':
    a=torch.load(SOURCE/f'A_seed{seed}.pt',map_location='cpu',weights_only=True);original_error=max(float((saved[n]-a['semantic_head.'+n]).abs().max())for n in saved);assert original_error<=1e-7 and original_error==logged['old_A_parameter_max_abs_error']
   checks[f'{key}_{seed}']={'all500_optimizer_steps_replayed':True,'parameter_max_abs_replay_error':error};del h,opt;gc.collect();torch.cuda.empty_cache()
  print('Verified1500features subset seed',seed,'and both full500 head trajectories',flush=True);del tx,ty,x;gc.collect();torch.cuda.empty_cache()
 del train;gc.collect();torch.cuda.empty_cache()
 # Rebuild both evaluation feature populations, then score six frozen heads.
 torch.manual_seed(0);tail=SemanticTail(published,26).cuda().eval().requires_grad_(False);initial=torch.load(SOURCE/'initial_seed0.pt',map_location='cpu',weights_only=True)
 with torch.no_grad():
  for n,q in tail.named_parameters():
   if n in initial:q.copy_(initial[n].to(q.device))
 captured=[];handle=tail.out_norm.register_forward_hook(lambda m,i,o:captured.append(o.detach()))
 for split,total in [('train',15360),('val',40960)]:
  with np.load(ROOT/f'{split}_evaluation_features.npz')as a:x=a['features'].copy();y=a['labels'].copy();ids=a['video_ids'].copy()
  offset=0
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for c in sorted(manifest[split],key=lambda c:int(c['video_id'])):
    vid=int(c['video_id'])
    with np.load(CACHE/split/f'{vid}.npz')as a:indices=np.where(a['xy'][:,2]==0)[0]if split=='train'else np.arange(len(a['labels']));q=torch.from_numpy(a['prelast'][indices].copy()).cuda().to(torch.bfloat16);m=torch.from_numpy(a['memory'].copy()).cuda().to(torch.bfloat16)[None];target=a['labels'][indices]
    for start in range(0,len(indices),64):tail(q[start:start+64][None],m);got=captured.pop()[0].float().cpu().numpy();assert np.array_equal(got,x[offset+start:offset+start+len(got)])
    n=len(indices);assert np.array_equal(y[offset:offset+n],target)and np.all(ids[offset:offset+n]==vid);offset+=n
  assert offset==len(y)==total;tx=torch.from_numpy(x).cuda()
  for seed in range(3):
   for key in p['rates']:
    state=torch.load(ROOT/f'{key}_seed{seed}.pt',map_location='cpu',weights_only=True);head=load_head(state).eval();out=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     for start in range(0,len(y),64):out.append(head(tx[start:start+64]).float().cpu().numpy())
    z=np.concatenate(out)
    with np.load(ROOT/f'{key}_seed{seed}_{split}.npz')as a:assert np.array_equal(z,a['logits'])and np.array_equal(y,a['labels'])and np.array_equal(ids,a['video_ids'])
    check_stat(z,y,s['results'][key][str(seed)][split]);per={str(int(v)):metrics(y[ids==v],z[ids==v].argmax(1))[0]['macro_f1_present']for v in np.unique(ids)};assert per==s['results'][key][str(seed)][split]['per_video_macro']
    if split=='val'and key=='low':
     with np.load(SOURCE/f'A_seed{seed}_validation.npz')as a:assert np.array_equal(z,a['logits'])
    del head
  print('Verified evaluation features, six logits and metrics for',split,flush=True);del tx;gc.collect();torch.cuda.empty_cache()
 handle.remove();expected=[]
 for seed in range(3):
  lo=s['results']['low'][str(seed)];hi=s['results']['high'][str(seed)];expected.append({'seed':seed,'train_FG_macro_difference':hi['train']['metrics']['foreground_macro_f1_present']-lo['train']['metrics']['foreground_macro_f1_present'],'val_macro_difference':hi['val']['metrics']['macro_f1_present']-lo['val']['metrics']['macro_f1_present'],'val_video_equal_difference':float(np.mean([hi['val']['per_video_macro'][v]-lo['val']['per_video_macro'][v]for v in lo['val']['per_video_macro']]))})
 assert expected==s['high_minus_low'];assert all(r['train_FG_macro_difference']>0 and r['val_macro_difference']>0 for r in expected)==s['head_lr1e3_eligible_for_next_matched_LoRA_control']
 for path,h in p['input_sha256'].items():assert sha(path)==h,path
 atomic_json(ROOT/'verification.json',{'round':101,'summary_sha256':sha(ROOT/'summary.json'),'verifier_sha256':sha('semantic/verify_d4rt_head_optimization.py'),'all1500_schedule_features_and_labels_recomputed_exact':True,'all_six_full500_step_optimizers_replayed':checks,'all_three_low_lr_heads_reproduce_R99_A_within_fixed_tolerance':True,'low_lr_all_original_validation_logits_exact':True,'all12_train_validation_logits_and_metrics_recomputed_exact':True,'all_feature_inputs_and_original_outputs_unchanged':True,'contrast_and_next_control_eligibility_verified':True,'new_confirmation':False});print('DONE independent head optimizer verification',flush=True)
if __name__=='__main__':main()
