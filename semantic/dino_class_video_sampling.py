"""Validation-only B14 foreground class / conditional-video sampling control."""
import fcntl,gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_backbone_training as prior
from semantic.dino_backbone_resource import WEIGHTS,VENDOR
from semantic.model import SemanticHead
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_class_video_sampling');ARMS=('query','class','class_video')
def ckpt(arm,seed):return prior.ckpt('B14_gt32',seed)if arm=='query'else ROOT/arm/'runs'/f'linear_seed{seed}.pt'
class BalancedFG:
 def __init__(self,labels,video_ids,seed):
  self.y=torch.tensor(labels,device='cuda',dtype=torch.long);self.video=torch.tensor(video_ids,device='cuda',dtype=torch.long);self.fg=torch.where(self.y>0)[0];self.bg=torch.where(self.y==0)[0];self.classes=self.y[self.fg].unique(sorted=True)
  self.classpools=[torch.where(self.y==c)[0]for c in self.classes];self.videopools=[]
  for c,pool in zip(self.classes,self.classpools):self.videopools.append([pool[self.video[pool]==v]for v in self.video[pool].unique(sorted=True)])
  self.base=torch.Generator(device='cuda').manual_seed(seed);self.classgen=torch.Generator(device='cuda').manual_seed(20261061+seed);self.detail=torch.Generator(device='cuda').manual_seed(20262061+seed)
 def draw(self,arm):
  basefg=self.fg[torch.randint(len(self.fg),(16,),generator=self.base,device='cuda')];bg=self.bg[torch.randint(len(self.bg),(16,),generator=self.base,device='cuda')];choice=torch.randint(len(self.classes),(16,),generator=self.classgen,device='cuda');fg=torch.empty_like(choice)
  for c,pool in enumerate(self.classpools):
   rows=torch.where(choice==c)[0]
   if not len(rows):continue
   if arm=='class':fg[rows]=pool[torch.randint(len(pool),(len(rows),),generator=self.detail,device='cuda')]
   elif arm=='class_video':
    pools=self.videopools[c];selected=torch.randint(len(pools),(len(rows),),generator=self.detail,device='cuda')
    for v,pv in enumerate(pools):
     dest=rows[selected==v]
     if len(dest):fg[dest]=pv[torch.randint(len(pv),(len(dest),),generator=self.detail,device='cuda')]
   else:raise ValueError(arm)
  return fg,bg,basefg,choice

def freeze():
 previous=json.loads((prior.ROOT/'summary.json').read_text());verify=json.loads((prior.ROOT/'verification.json').read_text());assert file_hash(prior.ROOT/'summary.json')==verify['summary_sha256'];m=json.loads((prior.ROOT/'manifest.json').read_text())
 paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'manifest.json',prior.ROOT/'frozen_selection.json',WEIGHTS,Path('semantic/dino_class_video_sampling.py'),Path('semantic/OVIS_DINO_CLASS_VIDEO_SAMPLING_PROTOCOL.md'),Path('semantic/model.py'),Path('semantic/head_budget_probe.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py')]+sorted(VENDOR.rglob('*.py'))
 paths+=[prior.fp(split,c['video_id'])for split in m for c in m[split]]
 for seed in range(3):paths.extend([ckpt('query',seed),ckpt('query',seed).with_suffix('.json')])
 p={'round':61,'arms':list(ARMS),'seeds':[0,1,2],'steps':2000,'FG_batch':16,'BG_batch':16,'head_parameters':19994,'new_heads':6,'query_offset':-2.0,'calibration_grid':previous['protocol']['calibration_grid'],'shared_class_seed_base':20261061,'detail_seed_base':20262061,'validation_only':True,'no_new_confirmation':True,'input_sha256':{str(x):file_hash(x)for x in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p,previous

def train(arm,seed,clips,val,p):
 path=ckpt(arm,seed);rp=path.with_suffix('.json')
 if rp.exists():
  r=json.loads(rp.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();features=torch.from_numpy(np.concatenate([c['decoded']for c in clips])).float().cuda();labels=np.concatenate([c['labels']for c in clips]);videos=np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int64)for c in clips]);sampler=BalancedFG(labels,videos,seed)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);hist=torch.zeros(26,dtype=torch.long,device='cuda');actual=hashlib.sha256();base=hashlib.sha256();bgh=hashlib.sha256();clsh=hashlib.sha256();fgvideo=torch.zeros(80*26,dtype=torch.long,device='cuda');bgvideo=torch.zeros(80,dtype=torch.long,device='cuda');unique=np.unique(videos);vindex=torch.tensor(np.searchsorted(unique,videos),device='cuda');best=-1;history=[]
 for step in range(1,2001):
  fg,bg,bf,choice=sampler.draw(arm);ix=torch.cat([fg,bg]);actual.update(ix.cpu().numpy().tobytes());base.update(torch.cat([bf,bg]).cpu().numpy().tobytes());bgh.update(bg.cpu().numpy().tobytes());clsh.update(choice.cpu().numpy().tobytes());hist+=torch.bincount(sampler.y[ix],minlength=26);fgvideo+=torch.bincount(vindex[fg]*26+sampler.y[fg],minlength=80*26);bgvideo+=torch.bincount(vindex[bg],minlength=80)
  model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(features[ix]),sampler.y[ix]);loss.backward();optimizer.step()
  if step%50==0:
   value=score(model,val);history.append({'step':step,'val_macro_f1':value,'loss':float(loss.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 reference=json.loads(ckpt('query',seed).with_suffix('.json').read_text());assert base.hexdigest()==reference['sampled_indices_sha256']and int(hist.sum())==64000 and int(hist[0])==32000
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},tmp);tmp.replace(path)
 r={'arm':arm,'seed':seed,'protocol':p,'best_step':best_step,'best_val':best,'history':history,'sampled_class_counts':hist.cpu().tolist(),'sampled_indices_sha256':actual.hexdigest(),'base_indices_sha256':base.hexdigest(),'BG_indices_sha256':bgh.hexdigest(),'class_choices_sha256':clsh.hexdigest(),'video_ids':unique.tolist(),'FG_video_class_counts':fgvideo.reshape(80,26).cpu().tolist(),'BG_video_counts':bgvideo.cpu().tolist(),'checkpoint_sha256':file_hash(path),'parameter_count':19994};atomic_json(rp,r)
 print('class/video trained',arm,seed,best_step,best,flush=True);del features,model;gc.collect();torch.cuda.empty_cache();return r

def infer(arm,seed,clips):
 path=ROOT/arm/'logits'/f'dinov2_{seed}_val.npz';h=file_hash(ckpt(arm,seed))
 if path.exists():
  with np.load(path)as a:out={k:a[k].copy()for k in a.files}
  assert str(out['checkpoint_sha256'])==h;return out
 model=SemanticHead(hidden_dim=768,num_classes=26).cuda().eval();model.load_state_dict(torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model'])
 with torch.inference_mode():values=[model(torch.from_numpy(c['decoded']).float().cuda()).cpu().numpy()for c in clips]
 out={'logits':np.concatenate(values),'labels':np.concatenate([c['labels']for c in clips]),'times':np.concatenate([c['xy'][:,2]for c in clips]),'video_ids':np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int32)for c in clips]),'checkpoint_sha256':np.array(h)};save_npz(path,**out);del model;return out

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed class/video experiment exists')
  torch.backends.cuda.matmul.allow_tf32=False;m,p,previous=freeze();source=prior.read('train',m);val=prior.read('val',m);pool=[]
  for c in source:
   assert np.all(c['xy'][:,2]==0)and (c['labels']>0).sum()==128 and (c['labels']==0).sum()==128
   for cid in np.unique(c['labels']):
    rows=c['labels']==cid;pool.append({'video_id':int(c['video_id']),'class_id':int(cid),'queries':int(rows.sum()),'distinct_xy':len(np.unique(c['xy'][rows,:2],axis=0))})
  atomic_json(ROOT/'source_pool_audit.json',pool);reports={arm:[train(arm,s,source,val,p)for s in range(3)]for arm in ARMS[1:]}
  for seed in range(3):
   a=reports['class'][seed];b=reports['class_video'][seed]
   for key in ('sampled_class_counts','base_indices_sha256','BG_indices_sha256','class_choices_sha256','BG_video_counts'):assert a[key]==b[key]
  results={};offsets={'query':-2.0};curves={}
  for arm in ARMS:
   outputs=[infer(arm,seed,val)for seed in range(3)]
   if arm!='query':offsets[arm],curves[arm]=pick_offset(outputs,p['calibration_grid'])
   runs=[{'seed':seed,'raw_validation':evaluate(outputs[seed],0),'calibrated_validation':evaluate(outputs[seed],offsets[arm])}for seed in range(3)]
   if arm=='query':assert runs==previous['results']['B14_gt32']['runs']
   results[arm]={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([a['calibrated_validation']['macro_f1_present']for a in runs]))};print('class/video validation',arm,results[arm]['mean_calibrated_validation_macro'],flush=True)
  selected=max(ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro']);selection={'validation_selected_arm':selected,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{arm}_{seed}':file_hash(ckpt(arm,seed))for arm in ARMS for seed in range(3)},'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'training_reports':reports,'results':results,'validation_selected_arm':selected,'source_pool_audit_sha256':file_hash(ROOT/'source_pool_audit.json'),'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE class/video sampling control',selected,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
