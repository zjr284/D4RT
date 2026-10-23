"""Real-checkpoint two-branch source-query invariance and geometry isolation probe."""
from pathlib import Path
import copy,fcntl,gc,json,time
import numpy as np
import torch
import torch.nn.functional as F
from semantic.model import GEOMETRY_KEYS
from semantic.source_query_adapter import SourceQuerySemanticD4RT
from semantic.ytvis import load_video
from semantic.run import atomic_json
from semantic import d4rt_finetune_readiness as ready
from tools.load_opend4rt import load_model
ROOT=Path('semantic_artifacts/source_query_adapter_probe')

def tsha(t):return ready.tensor_sha(t.view(torch.uint8))
def base_hash(model):return {n:tsha(q.detach().cpu().contiguous())for n,q in model.named_parameters()}
def tail_state(model):return {n:q.detach().cpu().clone()for n,q in model.semantic_tail.named_parameters()if q.requires_grad}

def batches(clips):
 items=[]
 for c in clips:
  with np.load(ready.CACHE/'train'/f"{c['video_id']}.npz")as a:xy=a['xy'].copy();y=a['labels'].copy()
  idx=np.r_[np.where((xy[:,2]==0)&(y>0))[0][:32],np.where((xy[:,2]==0)&(y==0))[0][:32]];assert len(idx)==64;chosen=xy[idx];items.append({'video':load_video(c)[None].cuda().to(torch.bfloat16),'query':{'u':torch.tensor((chosen[:,0]+.5)/128,dtype=torch.float32,device='cuda')[None],'v':torch.tensor((chosen[:,1]+.5)/128,dtype=torch.float32,device='cuda')[None],'t_src':torch.tensor(chosen[:,2],dtype=torch.long,device='cuda')[None]},'labels':torch.tensor(y[idx].astype(np.int64),device='cuda'),'video_id':int(c['video_id']),'indices':idx.tolist()})
 rng=np.random.default_rng(104)
 for item in items:
  item['query']['t_tgt']=torch.tensor(rng.integers(8,size=(1,64)),device='cuda');item['query']['t_cam']=torch.tensor(rng.integers(8,size=(1,64)),device='cuda')
 pair={'video':torch.cat([i['video']for i in items]),'query':{k:torch.cat([i['query'][k]for i in items])for k in items[0]['query']},'labels':torch.stack([i['labels']for i in items]),'video_id':[i['video_id']for i in items],'indices':[i['indices']for i in items]}
 return items+[pair]

def variants(item):
 original={k:v for k,v in item.items()if k in ('video','query')};alter={'video':item['video'],'query':dict(item['query'])};alter['query']['t_tgt']=(alter['query']['t_tgt']+1)%8;alter['query']['t_cam']=(alter['query']['t_cam']+3)%8
 source={'video':item['video'],'query':dict(item['query'])};source['query']['t_src']=(source['query']['t_src']+1)%8
 aspect={**original,'aspect_ratio':torch.tensor([1.2]if len(item['video'])==1 else[1.2,.8],device='cuda',dtype=torch.bfloat16)}
 return {'original':original,'target_camera':alter,'source':source,'aspect':aspect}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists()or(ROOT/'protocol.json').exists():raise RuntimeError('Existing run; inspect instead of overwrite')
  parent=Path('semantic_artifacts/d4rt_auxiliary_label_oracle');assert json.loads((parent/'verification.json').read_text())['summary_sha256']==ready.sha(parent/'summary.json');pub=json.loads((parent/'publication.json').read_text());assert pub['remote_contains_research_commit']
  manifest=json.loads(ready.MANIFEST.read_text());clips=[]
  for c in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
   with np.load(ready.CACHE/'train'/f"{c['video_id']}.npz")as a:xy=a['xy'];y=a['labels']
   if np.any((xy[:,2]==0)&(y>0))and np.any((xy[:,2]==0)&(y==0)):clips.append(c)
   if len(clips)==2:break
  files=[ready.CHECKPOINT,ready.CONFIG,ready.MANIFEST,parent/'summary.json',parent/'verification.json',Path('semantic/model.py'),Path('semantic/source_query_adapter.py'),Path('semantic/source_query_adapter_probe.py'),Path('semantic/OVIS_SOURCE_QUERY_ADAPTER_PROTOCOL.md')]+[ready.CACHE/'train'/f"{c['video_id']}.npz"for c in clips]+[Path(f)for c in clips for f in c['frames']]
  data=batches(clips);p={'round':104,'source_commit':pub['research_commit'],'video_ids':[c['video_id']for c in clips],'query_indices':[r['indices']for r in data[:2]],'query_rng_seed':104,'steps':5,'input_sha256':{str(f):ready.sha(f)for f in files},'new_semantic_geometry_architecture':True,'no_accuracy_or_new_confirmation':True};atomic_json(ROOT/'protocol.json',p)
  torch.manual_seed(0);base=load_model(ready.CHECKPOINT,ready.CONFIG);model=SourceQuerySemanticD4RT(base,26);cpu_base_ptr={q.untyped_storage().data_ptr()for q in model.base.parameters()};assert not cpu_base_ptr&{q.untyped_storage().data_ptr()for q in model.semantic_tail.parameters()};model.base.to('cuda',dtype=torch.bfloat16);model.semantic_tail.cuda();model.train(True);assert not model.base.training and not model.semantic_tail.training
  initial=tail_state(model);torch.save(initial,ROOT/'initial_adapter.pt');before_base=base_hash(model.base);assert sum(q.numel()for q in model.parameters()if q.requires_grad)==217626
  records=[];torch.cuda.reset_peak_memory_stats()
  def evaluate(phase):
   for j,item in enumerate(data):
    scores={}
    for name,batch in variants(item).items():
     saved_query={k:q.clone()for k,q in batch['query'].items()};start=time.perf_counter()
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(batch);reference=model.base(batch)
     torch.cuda.synchronize();seconds=time.perf_counter()-start;diff={k:float((out[k].float()-reference[k].float()).abs().max())for k in GEOMETRY_KEYS};assert all(x==0 for x in diff.values());assert all(torch.equal(saved_query[k],batch['query'][k])for k in saved_query);path=ROOT/f'{phase}_case{j}_{name}.npz';np.savez_compressed(path,**{k:v.float().cpu().numpy()for k,v in out.items()});scores[name]=out['semantic_logits'].detach().cpu().clone();records.append({'phase':phase,'case':j,'batch_size':len(item['video']),'variant':name,'geometry_max_difference':diff,'seconds_sequential_adapter_and_base':seconds,'output_path':str(path),'sha256':ready.sha(path)})
    assert torch.equal(scores['original'],scores['target_camera'])
  evaluate('before');optimizer=torch.optim.AdamW([{'params':[q for n,q in model.semantic_tail.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},{'params':[q for n,q in model.semantic_tail.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01);trace=[];train_batch=variants(data[0])['original']
  for step in range(5):
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(train_batch);loss=F.cross_entropy(out['semantic_logits'][0].float(),data[0]['labels'])
   optimizer.zero_grad(set_to_none=True);loss.backward();assert all(q.grad is None for q in model.base.parameters())
   if step==0:torch.save({'gradients':{n:q.grad.detach().cpu().clone()for n,q in model.semantic_tail.named_parameters()if q.grad is not None}},ROOT/'first_gradients.pt')
   if step==4:torch.save({'adapter':tail_state(model),'optimizer':optimizer.state_dict()},ROOT/'before_final.pt')
   optimizer.step();trace.append(float(loss.detach()))
  final=tail_state(model);torch.save(final,ROOT/'final_adapter.pt');assert any(not torch.equal(initial[n],final[n])for n in final if n.endswith('.b'));assert before_base==base_hash(model.base);evaluate('after');assert before_base==base_hash(model.base)
  for path,h in p['input_sha256'].items():assert ready.sha(path)==h
  atomic_json(ROOT/'summary.json',{'round':104,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'records':records,'GT_CE_five_steps':trace,'initial_adapter_sha256':ready.state_sha(initial),'final_adapter_sha256':ready.state_sha(final),'base_parameter_sha256_before_and_after':before_base,'base_gradients_all_none':True,'base_tail_storage_disjoint':True,'only_semantic_LoRA_and_head_updated':True,'trainable_parameters':217626,'all_target_camera_variant_semantics_exact':True,'all24_geometry_cases_equal_same_input_frozen_base':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'artifacts_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*.pt')},'new_confirmation_or_accuracy_evaluation':False});print('DONE source-query adapter probe:24 geometry cases, B1/B2, semantic target/camera invariance',flush=True)
if __name__=='__main__':main()
