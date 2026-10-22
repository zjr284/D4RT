"""Real-checkpoint A/B/C five-step feasibility probe; not a benchmark."""
from collections import defaultdict
import copy, fcntl, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from semantic.model import SemanticTail, geometry_vector, GEOMETRY_KEYS
from semantic.run import atomic_json
from semantic.ytvis import load_dataset
from tools.load_opend4rt import load_model, DEFAULT_VARIANT

ROOT=Path('semantic_artifacts/d4rt_finetune_readiness')
CACHE=Path('semantic_artifacts/features_ovis/sam')
MANIFEST=Path('semantic_artifacts/manifest.json')
ANN=Path('semantic_artifacts/ovis/annotations_train.json')
TEACHER=Path('semantic_artifacts/teacher_ovis')
CHECKPOINT=DEFAULT_VARIANT/'opend4rt.ckpt'
CONFIG=DEFAULT_VARIANT/'model.yaml'
ARMS=('A','B','C')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()

def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def state_sha(state):return {k:tensor_sha(v)for k,v in state.items()}
def weights(tail):return {n:p.detach().cpu().clone()for n,p in tail.named_parameters()if n.endswith(('.a','.b'))or n.startswith('semantic_head.')}
def frozen_sha(tail):return state_sha({n:p for n,p in tail.named_parameters()if not(n.endswith(('.a','.b'))or n.startswith('semantic_head.'))})

def loss_terms(out,ref,labels,gt_count=64):
 ce=F.cross_entropy(out['semantic_logits'][0,:gt_count].float(),labels[:gt_count])
 aux=F.cross_entropy(out['semantic_logits'][0,gt_count:].float(),labels[gt_count:])
 geo=F.smooth_l1_loss(geometry_vector(out).float(),geometry_vector(ref).float())
 return ce,aux,geo

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Existing result; inspect instead of rerun')
  manifest=json.loads(MANIFEST.read_text());metadata=json.loads((CACHE/'metadata.json').read_text())
  assert sha(MANIFEST)==metadata['manifest_sha256']
  splits={k:{int(c['video_id'])for c in manifest[k]}for k in ('train','val','test')}
  assert [len(splits[k])for k in splits]==[60,20,20]
  assert all(not splits[a]&splits[b]for a,b in (('train','val'),('train','test'),('val','test')))
  clip=None
  for item in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
   path=CACHE/'train'/f"{item['video_id']}.npz"
   with np.load(path)as a:xy=a['xy'].copy();y=a['labels'].copy()
   if np.any((xy[:,2]==0)&(y>0))and np.any((xy[:,2]==0)&(y==0))and np.any((xy[:,2]>0)&(y>0)):
    clip=item;break
  assert clip is not None
  vid=int(clip['video_id']);path=CACHE/'train'/f'{vid}.npz';teacher_path=TEACHER/f'{vid}.npz'
  with np.load(path)as a:data={k:a[k].copy()for k in ('xy','labels','prelast','memory','video_id')}
  with np.load(teacher_path)as a:pseudo=a['labels'].copy()
  dataset=load_dataset('ovis',ANN,Path('semantic_artifacts/ovis'))
  gt=dataset.gt_label_map(clip,0);assert np.array_equal(pseudo[0],gt)
  xy=data['xy'];labels=data['labels'];assert np.array_equal(pseudo[xy[:,2],xy[:,1],xy[:,0]],labels)
  fg=np.flatnonzero((xy[:,2]==0)&(labels>0));bg=np.flatnonzero((xy[:,2]==0)&(labels==0));aux=np.flatnonzero((xy[:,2]>0)&(labels>0))
  rng=np.random.default_rng(20261098);schedule=[]
  for step in range(5):schedule.append(np.concatenate([rng.choice(fg,32,replace=len(fg)<32),rng.choice(bg,32,replace=len(bg)<32),rng.choice(aux,32,replace=len(aux)<32)]).astype(int).tolist())
  files=[CHECKPOINT,CONFIG,MANIFEST,ANN,CACHE/'metadata.json',path,teacher_path,Path('semantic/model.py'),Path('tools/load_opend4rt.py'),Path('semantic/d4rt_finetune_readiness.py'),Path('semantic/OVIS_D4RT_FINETUNE_READINESS_PROTOCOL.md')]
  print('Hashing local source checkpoint and inputs',flush=True)
  protocol={'round':98,'seed':0,'steps':5,'query_seed':20261098,'video_id':vid,'cache_path':str(path),'teacher_path':str(teacher_path),'schedule':schedule,'GT_queries_per_step':64,'aux_queries_per_step':32,'loss_coefficients':{'A':[1.,0.,0.],'B':[1.,.1,0.],'C':[1.,.1,.25]},'input_sha256':{str(p):sha(p)for p in files},'no_validation_or_test_scoring':True,'is_performance_experiment':False}
  atomic_json(ROOT/'protocol.json',protocol)
  print('Strict loading local OpenD4RT checkpoint',flush=True)
  started=time.perf_counter();published=load_model(CHECKPOINT,CONFIG)
  total=sum(p.numel()for p in published.parameters());torch.manual_seed(0);tail=SemanticTail(published,26).eval();initial=weights(tail)
  # Independent frozen published modules provide the zero-update reference.
  reference=torch.nn.ModuleDict({'blocks':copy.deepcopy(published.decoder.blocks[-2:]),'norm':copy.deepcopy(published.decoder.out_norm),'heads':copy.deepcopy(published.heads)}).eval()
  for p in reference.parameters():p.requires_grad_(False)
  del published;gc.collect()
  reference=reference.cuda();before=torch.from_numpy(data['prelast']).cuda().to(torch.bfloat16);memory=torch.from_numpy(data['memory']).cuda().to(torch.bfloat16)[None];target=torch.from_numpy(labels.astype(np.int64)).cuda()
  def ref_forward(index):
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    x=before[index][None]
    for block in reference['blocks']:x=block(x,memory)
    return reference['heads'](reference['norm'](x))
  torch.save(initial,ROOT/'initial_adapter.pt');base_frozen=frozen_sha(tail);results={}
  for arm in ARMS:
   model=copy.deepcopy(tail).cuda().eval()
   if arm=='A':
    for n,p in model.named_parameters():
     if n.endswith(('.a','.b')):p.requires_grad_(False)
   optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=1e-4,weight_decay=.01)
   torch.cuda.reset_peak_memory_stats();rows=[];initial_geo_max=None
   for step,indices in enumerate(schedule):
    index=torch.tensor(indices,device='cuda');ref=ref_forward(index)
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(before[index][None],memory)
    if step==0:
     initial_geo_max={k:float((out[k].float()-ref[k].float()).abs().max())for k in GEOMETRY_KEYS};assert all(v==0 for v in initial_geo_max.values())
    ce,sam,geo=loss_terms(out,ref,target[index]);coeff=protocol['loss_coefficients'][arm];loss=ce+coeff[1]*geo+coeff[2]*sam
    optimizer.zero_grad(set_to_none=True);loss.backward()
    gradients={n:{'l2':float(p.grad.float().norm()),'finite':bool(torch.isfinite(p.grad).all())}for n,p in model.named_parameters()if p.grad is not None}
    assert all(r['finite']for r in gradients.values())and all(p.grad is None for n,p in model.named_parameters()if not p.requires_grad)
    if step==0:torch.save({'adapter':weights(model),'gradients':{n:p.grad.detach().cpu()for n,p in model.named_parameters()if p.grad is not None}},ROOT/f'{arm}_step0.pt')
    if step==4:torch.save({'adapter':weights(model),'optimizer':optimizer.state_dict()},ROOT/f'{arm}_before_final.pt')
    optimizer.step()
    rows.append({'step':step+1,'GT_CE':float(ce.detach()),'SAM_CE':float(sam.detach()),'geometry_smooth_l1':float(geo.detach()),'total':float(loss.detach()),'gradients':gradients})
   final=weights(model);torch.save(final,ROOT/f'{arm}_adapter.pt');assert frozen_sha(model)==base_frozen
   final_geos=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for indices in schedule:
     index=torch.tensor(indices,device='cuda');out=model(before[index][None],memory);ref=ref_forward(index)
     final_geos.append({k:{'mean_abs':float((out[k].float()-ref[k].float()).abs().mean()),'max_abs':float((out[k].float()-ref[k].float()).abs().max())}for k in GEOMETRY_KEYS})
   results[arm]={'initial_adapter_sha256':state_sha(initial),'final_adapter_sha256':state_sha(final),'initial_geometry_max_difference':initial_geo_max,'trainable_parameters':sum(p.numel()for p in model.parameters()if p.requires_grad),'frozen_parameters':sum(p.numel()for p in model.parameters()if not p.requires_grad),'rows':rows,'final_same_queries_geometry_differences':final_geos,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'frozen_tail_unchanged':True}
   print('Finished five-step arm',arm,'peak MiB',results[arm]['peak_gpu_allocated_bytes']/2**20,flush=True)
   del model,optimizer;gc.collect();torch.cuda.empty_cache()
  for p,h in protocol['input_sha256'].items():assert sha(p)==h
  atomic_json(ROOT/'summary.json',{'round':98,'protocol':protocol,'checkpoint_bytes':CHECKPOINT.stat().st_size,'base_model_parameters':total,'GPU':torch.cuda.get_device_name(),'torch':torch.__version__,'seconds':time.perf_counter()-started,'results':results,'no_validation_or_test_scoring':True,'geometry_truth_accuracy_measured':False,'source_weights_unchanged':True,'artifacts_sha256':{str(p):sha(p)for p in ROOT.glob('*.pt')}})
  print('DONE readiness; not a semantic or geometry accuracy result',flush=True)
if __name__=='__main__':main()
