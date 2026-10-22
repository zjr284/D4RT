"""Fixed-budget four-arm OpenD4RT tail adaptation on already-observed data."""
import copy,fcntl,gc,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from semantic.model import SemanticTail,geometry_vector,GEOMETRY_KEYS
from semantic.run import atomic_json,classification_metrics
from semantic import d4rt_finetune_readiness as ready
from tools.load_opend4rt import load_model
ROOT=Path('semantic_artifacts/d4rt_controlled_finetune');ARMS=('A','B','C','D');STEPS=500
COEFF={'A':(0.,0.,0.),'B':(.1,0.,0.),'C':(.1,.25,0.),'D':(.1,0.,.25)}

def load_data(split):
 manifest=json.loads(ready.MANIFEST.read_text());out={}
 for clip in sorted(manifest[split],key=lambda r:int(r['video_id'])):
  vid=int(clip['video_id'])
  with np.load(ready.CACHE/split/f'{vid}.npz')as a:
   row={k:a[k].copy()for k in ('prelast','memory','labels','xy')}
   if split=='val':row['alternate_prelast']=a['alternate_prelast'].copy()
  for key in ('prelast','memory','alternate_prelast'):
   if key in row:row[key]=torch.from_numpy(row[key]).to('cuda',dtype=torch.bfloat16)
  row['target']=torch.from_numpy(row['labels'].astype(np.int64)).cuda();out[vid]=row
 return out

def build_schedules(train):
 pools={vid:{'fg':np.where((r['xy'][:,2]==0)&(r['labels']>0))[0],'bg':np.where((r['xy'][:,2]==0)&(r['labels']==0))[0],'sam':np.where((r['xy'][:,2]>0)&(r['labels']>0))[0]}for vid,r in train.items()}
 assert all(len(p['fg'])and len(p['bg'])for p in pools.values())
 vids=list(train);aux_vids=[vid for vid in vids if len(pools[vid]['sam'])];assert aux_vids
 schedules={}
 for seed in range(3):
  rng=np.random.default_rng(20261099+seed);rows=[]
  def sample(pool):return rng.choice(pool,32,replace=len(pool)<32).astype(int).tolist()
  for step in range(STEPS):
   vid=vids[int(rng.integers(len(vids)))];av=aux_vids[int(rng.integers(len(aux_vids)))];p=pools[vid]
   rows.append({'GT_video':vid,'aux_video':av,'GT_indices':sample(p['fg'])+sample(p['bg']),'extra_GT_indices':sample(p['fg']),'SAM_indices':sample(pools[av]['sam'])})
  schedules[str(seed)]=rows
 return schedules,{str(vid):{k:len(v)for k,v in p.items()}for vid,p in pools.items()}

def make_model(template,arm,state=None):
 model=copy.deepcopy(template).cuda().eval()
 if state is not None:
  with torch.no_grad():
   for n,p in model.named_parameters():
    if n in state:p.copy_(state[n].to(p.device))
 if arm=='A':
  for n,p in model.named_parameters():
   if n.endswith(('.a','.b')):p.requires_grad_(False)
 return model

def batches(train,row):
 gt=train[row['GT_video']];aux=train[row['aux_video']];gi=torch.tensor(row['GT_indices']+row['extra_GT_indices'],device='cuda');ai=torch.tensor(row['SAM_indices'],device='cuda')
 return gt,aux,gi,ai

def calculate(model,train,row,ref_gt,ref_aux):
 gt,aux,gi,ai=batches(train,row)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  go=model(gt['prelast'][gi][None],gt['memory'][None]);ao=model(aux['prelast'][ai][None],aux['memory'][None])
 ce=F.cross_entropy(go['semantic_logits'][0,:64].float(),gt['target'][gi[:64]])
 extra=F.cross_entropy(go['semantic_logits'][0,64:].float(),gt['target'][gi[64:]])
 sam=F.cross_entropy(ao['semantic_logits'][0].float(),aux['target'][ai])
 geo=F.smooth_l1_loss(torch.cat([geometry_vector(go),geometry_vector(ao)],dim=1).float(),torch.cat([ref_gt,ref_aux],dim=1).float())
 return ce,sam,extra,geo

def evaluate(model,reference,val):
 all_logits=[];all_alt=[];all_y=[];all_ids=[];per={};drift={k:{'absolute_sum':0.,'count':0,'max_abs':0.}for k in GEOMETRY_KEYS}
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for vid,r in val.items():
   chunks=[];alternate=[]
   for start in range(0,len(r['labels']),64):
    q=r['prelast'][start:start+64][None];m=r['memory'][None];out=model(q,m);base=reference(q,m);alt=model(r['alternate_prelast'][start:start+64][None],m)
    chunks.append(out['semantic_logits'][0].float().cpu().numpy());alternate.append(alt['semantic_logits'][0].float().cpu().numpy())
    for k in GEOMETRY_KEYS:
     d=(out[k].float()-base[k].float()).abs();drift[k]['absolute_sum']+=float(d.double().sum());drift[k]['count']+=d.numel();drift[k]['max_abs']=max(drift[k]['max_abs'],float(d.max()))
   logits=np.concatenate(chunks);alt=np.concatenate(alternate);y=r['labels'].astype(np.int64)
   per[str(vid)]=classification_metrics(logits.argmax(1),y,26);all_logits.append(logits);all_alt.append(alt);all_y.append(y);all_ids.extend([vid]*len(y))
 logits=np.concatenate(all_logits);alternate=np.concatenate(all_alt);y=np.concatenate(all_y);pred=logits.argmax(1)
 metrics={'pooled':classification_metrics(pred,y,26),'per_video':per,'target_camera_variant_agreement':float(np.mean(pred==alternate.argmax(1))),'geometry_output_drift':{k:{**r,'mean_abs':r['absolute_sum']/r['count']}for k,r in drift.items()}}
 return metrics,{'logits':logits,'alternate_logits':alternate,'labels':y,'video_ids':np.asarray(all_ids,dtype=np.int64)}

def contrasts(results):
 out={}
 for left,right in (('C','B'),('B','A'),('C','D')):
  rows=[]
  for seed in range(3):
   l=results[left][str(seed)]['evaluation'];r=results[right][str(seed)]['evaluation'];d={vid:l['per_video'][vid]['macro_f1_present']-r['per_video'][vid]['macro_f1_present']for vid in l['per_video']}
   rows.append({'seed':seed,'pooled_macro_difference':l['pooled']['macro_f1_present']-r['pooled']['macro_f1_present'],'per_video_differences':d,'video_equal_mean_difference':float(np.mean(list(d.values())))})
  out[left+'_minus_'+right]={'per_seed':rows,'mean_pooled_macro_difference':float(np.mean([r['pooled_macro_difference']for r in rows])),'mean_video_equal_difference':float(np.mean([r['video_equal_mean_difference']for r in rows]))}
 return out

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Already complete; inspect existing results')
  if(ROOT/'protocol.json').exists():raise RuntimeError('Existing partial run; inspect before explicit resume')
  verification=json.loads((ready.ROOT/'verification.json').read_text());assert verification['summary_sha256']==ready.sha(ready.ROOT/'summary.json')
  publication=json.loads((ready.ROOT/'publication.json').read_text());assert publication['remote_contains_research_commit']
  train=load_data('train');assert len(train)==60
  schedules,pools=build_schedules(train);files=[ready.CHECKPOINT,ready.CONFIG,ready.MANIFEST,ready.ANN,ready.CACHE/'metadata.json',ready.ROOT/'summary.json',ready.ROOT/'verification.json',Path('semantic/model.py'),Path('semantic/d4rt_controlled_finetune.py'),Path('semantic/OVIS_D4RT_CONTROLLED_FINETUNE_PROTOCOL.md')]
  files += list((ready.CACHE/'train').glob('*.npz'))+list((ready.CACHE/'val').glob('*.npz'))+[ready.TEACHER/f'{vid}.npz'for vid in train]
  protocol={'round':99,'steps':STEPS,'seeds':[0,1,2],'arms':list(ARMS),'coefficients':{k:list(v)for k,v in COEFF.items()},'schedules':schedules,'training_label_pools':pools,'sampling_seed_base':20261099,'readiness_research_commit':publication['research_commit'],'input_sha256':{str(p):ready.sha(p)for p in files},'only_old_training_and_validation_data':True,'new_confirmation':False}
  atomic_json(ROOT/'protocol.json',protocol);print('Frozen protocol, train videos',len(train),'SAM auxiliary eligible',sum(p['sam']>0 for p in pools.values()),flush=True)
  published=load_model(ready.CHECKPOINT,ready.CONFIG);results={arm:{}for arm in ARMS};start=time.perf_counter()
  for seed in range(3):
   torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=ready.weights(template);torch.save(initial,ROOT/f'initial_seed{seed}.pt');reference=make_model(template,'A');reference.requires_grad_(False);refs_gt=[];refs_aux=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for row in schedules[str(seed)]:
     gt,aux,gi,ai=batches(train,row);refs_gt.append(geometry_vector(reference(gt['prelast'][gi][None],gt['memory'][None]))[0].float().cpu().numpy());refs_aux.append(geometry_vector(reference(aux['prelast'][ai][None],aux['memory'][None]))[0].float().cpu().numpy())
   ref_gt=torch.from_numpy(np.stack(refs_gt)).cuda();ref_aux=torch.from_numpy(np.stack(refs_aux)).cuda();np.savez_compressed(ROOT/f'references_seed{seed}.npz',GT=np.stack(refs_gt),aux=np.stack(refs_aux));del reference;gc.collect();torch.cuda.empty_cache()
   for arm in ARMS:
    model=make_model(template,arm);frozen_before=ready.frozen_sha(model);opt=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=1e-4,weight_decay=.01);trace=[];torch.cuda.reset_peak_memory_stats();arm_start=time.perf_counter()
    for step,row in enumerate(schedules[str(seed)]):
     ce,sam,extra,geo=calculate(model,train,row,ref_gt[step][None],ref_aux[step][None]);g,s,e=COEFF[arm];loss=ce+g*geo+s*sam+e*extra
     opt.zero_grad(set_to_none=True);loss.backward()
     if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss')
     if step==0:torch.save({'adapter':ready.weights(model),'gradients':{n:p.grad.detach().cpu().clone()for n,p in model.named_parameters()if p.grad is not None}},ROOT/f'{arm}_seed{seed}_step0.pt')
     if step==STEPS-1:torch.save({'adapter':ready.weights(model),'optimizer':opt.state_dict()},ROOT/f'{arm}_seed{seed}_before_final.pt')
     assert all(p.grad is None for p in model.parameters()if not p.requires_grad)
     opt.step();trace.append({'step':step+1,'GT_CE':float(ce.detach()),'SAM_CE':float(sam.detach()),'extra_GT_CE':float(extra.detach()),'geometry_smooth_l1':float(geo.detach()),'total':float(loss.detach())})
     if(step+1)%100==0:print('train',seed,arm,step+1,'GT_CE',round(trace[-1]['GT_CE'],5),flush=True)
    assert frozen_before==ready.frozen_sha(model);final=ready.weights(model);torch.save(final,ROOT/f'{arm}_seed{seed}.pt')
    results[arm][str(seed)]={'initial_adapter_sha256':ready.state_sha(initial),'final_adapter_sha256':ready.state_sha(final),'trainable_parameters':sum(p.numel()for p in model.parameters()if p.requires_grad),'steps':STEPS,'GT_supervision_exposures':STEPS*64,'SAM_supervision_exposures':STEPS*32 if arm=='C'else 0,'extra_GT_supervision_exposures':STEPS*32 if arm=='D'else 0,'trace':trace,'seconds':time.perf_counter()-arm_start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'frozen_tail_unchanged':True}
    atomic_json(ROOT/f'{arm}_seed{seed}_training.json',results[arm][str(seed)]);del model,opt;gc.collect();torch.cuda.empty_cache()
   del template,ref_gt,ref_aux;gc.collect();torch.cuda.empty_cache()
  frozen={str(ROOT/f'{arm}_seed{seed}.pt'):ready.sha(ROOT/f'{arm}_seed{seed}.pt')for arm in ARMS for seed in range(3)};atomic_json(ROOT/'frozen_selection.json',{'rule':'fixed final step500; all12 weights saved before validation scoring','adapters_sha256':frozen,'protocol_sha256':ready.sha(ROOT/'protocol.json')});print('All12 final weights frozen; starting old validation evaluation',flush=True)
  del train;gc.collect();torch.cuda.empty_cache();val=load_data('val')
  for seed in range(3):
   torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(ROOT/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True);reference=make_model(template,'A',initial).requires_grad_(False)
   for arm in ARMS:
    state=torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);model=make_model(template,arm,state);metrics,arrays=evaluate(model,reference,val);np.savez_compressed(ROOT/f'{arm}_seed{seed}_validation.npz',**arrays);results[arm][str(seed)]['evaluation']=metrics;print('val',seed,arm,'macro',metrics['pooled']['macro_f1_present'],flush=True);del model;gc.collect();torch.cuda.empty_cache()
   del template,reference;gc.collect();torch.cuda.empty_cache()
  comparison=contrasts(results);qualified=all(all(r['pooled_macro_difference']>0 for r in comparison[k]['per_seed'])and comparison[k]['mean_video_equal_difference']>0 for k in ('C_minus_B','C_minus_D'))
  for path,h in protocol['input_sha256'].items():assert ready.sha(path)==h,path
  atomic_json(ROOT/'summary.json',{'round':99,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'results':results,'contrasts':comparison,'SAM_recipe_eligible_for_future_separate_external_protocol':qualified,'geometry_truth_accuracy_evaluated':False,'default_deployment_changed':False,'seconds':time.perf_counter()-start,'output_sha256':{str(p):ready.sha(p)for p in ROOT.glob('*.pt')}|{str(p):ready.sha(p)for p in ROOT.glob('*.npz')},'frozen_selection_sha256':ready.sha(ROOT/'frozen_selection.json')})
  print('DONE exploratory fixed-budget experiment; external eligibility',qualified,flush=True)
if __name__=='__main__':main()
