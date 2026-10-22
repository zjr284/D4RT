"""Independent loss/gradient/final-step replay for real OpenD4RT feasibility."""
from collections import defaultdict
import gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_api
from semantic.model import SemanticTail,GEOMETRY_KEYS
from semantic.run import atomic_json
from tools.load_opend4rt import load_model,DEFAULT_VARIANT

ROOT=Path('semantic_artifacts/d4rt_finetune_readiness')
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb')as stream:
  for part in iter(lambda:stream.read(4*1024*1024),b''):h.update(part)
 return h.hexdigest()
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def managed(n):return n.endswith(('.a','.b'))or n.startswith('semantic_head.')
def pack(out):return torch.cat([out[k].reshape(1,-1,int(np.prod(out[k].shape[2:])))for k in GEOMETRY_KEYS],-1).float()
def manual_losses(out,ref,labels):
 scores=out['semantic_logits'][0].float().log_softmax(-1)
 losses=-scores[torch.arange(len(labels),device='cuda'),labels]
 delta=(pack(out)-ref).abs()
 geo=torch.where(delta<1,.5*delta.square(),delta-.5).mean()
 return losses[:64].mean(),losses[64:].mean(),geo

def main():
 summary=json.loads((ROOT/'summary.json').read_text());p=summary['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
 for path,h in p['input_sha256'].items():assert digest(path)==h,path
 for path,h in summary['artifacts_sha256'].items():assert digest(path)==h,path
 with np.load(p['cache_path'])as a:xy=a['xy'].copy();y=a['labels'].copy();pre=a['prelast'].copy();mem=a['memory'].copy()
 with np.load(p['teacher_path'])as a:teacher=a['labels'].copy()
 manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text());clip=next(c for c in manifest['train']if int(c['video_id'])==p['video_id'])
 # Rebuild GT by direct hits in individual resized instance masks; no dataset GT helper.
 annotation=json.loads(Path('semantic_artifacts/ovis/annotations_train.json').read_text());anns=[a for a in annotation['annotations']if a['video_id']==p['video_id']];hits=defaultdict(list)
 for a in anns:
  rle=a['segmentations'][clip['start']]
  if rle is None:continue
  decoded=mask_api.decode(rle)
  if decoded.ndim==3:decoded=decoded.any(2)
  reduced=np.asarray(Image.fromarray(decoded.astype('uint8')).resize((128,128),Image.Resampling.NEAREST))
  for j in np.where(xy[:,2]==0)[0]:
   u,v,t=xy[j]
   if reduced[v,u]:hits[int(j)].append(int(a['category_id']))
 for j in np.where(xy[:,2]==0)[0]:assert y[j]==(0 if not hits[int(j)]else hits[int(j)][0]if len(hits[int(j)])==1 else-1)
 assert np.array_equal(y,teacher[xy[:,2],xy[:,1],xy[:,0]])
 fg=np.where((xy[:,2]==0)&(y>0))[0];bg=np.where((xy[:,2]==0)&(y==0))[0];aux=np.where((xy[:,2]>0)&(y>0))[0]
 rng=np.random.default_rng(20261098)
 for row in p['schedule']:
  expected=np.r_[rng.choice(fg,32,replace=len(fg)<32),rng.choice(bg,32,replace=len(bg)<32),rng.choice(aux,32,replace=len(aux)<32)]
  assert np.array_equal(expected,row)
 assert p['steps']==len(p['schedule'])==5 and p['loss_coefficients']=={'A':[1.,0.,0.],'B':[1.,.1,0.],'C':[1.,.1,.25]}
 initial=torch.load(ROOT/'initial_adapter.pt',weights_only=True,map_location='cpu');initial_hash={n:tsha(v)for n,v in initial.items()}
 published=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml');assert sum(q.numel()for q in published.parameters())==summary['base_model_parameters']
 template=SemanticTail(published,26).eval();del published;gc.collect()
 before=torch.from_numpy(pre).cuda().to(torch.bfloat16);memory=torch.from_numpy(mem).cuda().to(torch.bfloat16)[None];labels=torch.from_numpy(y.astype(np.int64)).cuda()
 def create(arm,state):
  import copy
  model=copy.deepcopy(template).cuda().eval()
  for n,q in model.named_parameters():
   if n in state:q.data.copy_(state[n].to(q.device))
   if arm=='A'and n.endswith(('.a','.b')):q.requires_grad_(False)
  return model
 ref_model=create('A',initial)
 def reference(idx):
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return pack(ref_model(before[idx][None],memory)).detach()
 verification={};first_grad={}
 for arm in ('A','B','C'):
  result=summary['results'][arm];assert result['initial_adapter_sha256']==initial_hash
  final=torch.load(ROOT/f'{arm}_adapter.pt',weights_only=True,map_location='cpu')
  assert {n:tsha(v)for n,v in final.items()}==result['final_adapter_sha256']
  delta={n:float((final[n]-initial[n]).abs().max())for n in final}
  assert any(v>0 for n,v in delta.items()if n.startswith('semantic_head.'))
  assert all(v==0 for n,v in delta.items()if n.endswith(('.a','.b')))if arm=='A'else any(v>0 for n,v in delta.items()if n.endswith('.b'))
  model=create(arm,initial);assert sum(q.numel()for q in model.parameters()if q.requires_grad)==result['trainable_parameters']==(33306 if arm=='A'else 217626)
  frozen_before={n:tsha(q)for n,q in model.named_parameters()if not managed(n)}
  idx=torch.tensor(p['schedule'][0],device='cuda');ref=reference(idx)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(before[idx][None],memory)
  assert torch.equal(pack(out),ref)
  ce,aux_loss,geo=manual_losses(out,ref,labels[idx]);weight=p['loss_coefficients'][arm];total=ce+weight[1]*geo+weight[2]*aux_loss;total.backward()
  logged=result['rows'][0]
  for key,value in [('GT_CE',ce),('SAM_CE',aux_loss),('geometry_smooth_l1',geo),('total',total)]:assert abs(float(value.detach())-logged[key])<2e-6
  stored=torch.load(ROOT/f'{arm}_step0.pt',weights_only=True,map_location='cpu');grads={n:q.grad.detach().cpu()for n,q in model.named_parameters()if q.grad is not None}
  assert set(grads)==set(stored['gradients'])==set(logged['gradients'])
  max_gradient_error=0.
  for n,g in grads.items():
   assert managed(n)and bool(torch.isfinite(g).all());torch.testing.assert_close(g,stored['gradients'][n],rtol=5e-4,atol=5e-6)
   max_gradient_error=max(max_gradient_error,float((g-stored['gradients'][n]).abs().max()))
  assert all(q.grad is None for n,q in model.named_parameters()if not q.requires_grad)
  first_grad[arm]={n:g for n,g in grads.items()}
  del model,out;gc.collect();torch.cuda.empty_cache()
  # Independent final optimizer step from the recorded pre-step state and moments.
  checkpoint=torch.load(ROOT/f'{arm}_before_final.pt',weights_only=True,map_location='cpu');model=create(arm,checkpoint['adapter'])
  opt=torch.optim.AdamW([q for q in model.parameters()if q.requires_grad],lr=1e-4,weight_decay=.01);opt.load_state_dict(checkpoint['optimizer']);idx=torch.tensor(p['schedule'][-1],device='cuda');ref=reference(idx)
  with torch.autocast('cuda',dtype=torch.bfloat16):out=model(before[idx][None],memory)
  ce,aux_loss,geo=manual_losses(out,ref,labels[idx]);loss=ce+weight[1]*geo+weight[2]*aux_loss;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  maximum=0.
  for n,q in model.named_parameters():
   if n in final:
    actual=q.detach().cpu();torch.testing.assert_close(actual,final[n],rtol=3e-4,atol=2e-6);maximum=max(maximum,float((actual-final[n]).abs().max()))
  assert frozen_before=={n:tsha(q)for n,q in model.named_parameters()if not managed(n)}
  del model,opt,out;gc.collect();torch.cuda.empty_cache()
  model=create(arm,final)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for row,logged_geo in zip(p['schedule'],result['final_same_queries_geometry_differences']):
    idx=torch.tensor(row,device='cuda');out=model(before[idx][None],memory);ref=ref_model(before[idx][None],memory)
    for k in GEOMETRY_KEYS:
     d=(out[k].float()-ref[k].float()).abs();assert float(d.mean())==logged_geo[k]['mean_abs']and float(d.max())==logged_geo[k]['max_abs']
  for r in result['rows']:
   assert abs(r['total']-(r['GT_CE']+weight[1]*r['geometry_smooth_l1']+weight[2]*r['SAM_CE']))<2e-6
  verification[arm]={'trainable_parameters':result['trainable_parameters'],'first_gradient_max_abs_replay_error':max_gradient_error,'final_step_parameter_max_abs_replay_error':maximum,'all_six_geometry_outputs_recomputed_on_five_batches':True,'only_whitelisted_parameters_changed':True}
  print('Verified readiness arm',arm,flush=True);del model,out;gc.collect();torch.cuda.empty_cache()
 assert any(not torch.equal(first_grad['B'][n],first_grad['C'][n])for n in first_grad['B'])
 for path,h in p['input_sha256'].items():assert digest(path)==h,path
 atomic_json(ROOT/'verification.json',{'round':98,'summary_sha256':digest(ROOT/'summary.json'),'verifier_sha256':digest('semantic/verify_d4rt_finetune_readiness.py'),'source_first_frame_GT_independent_mask_hits_checked':int((xy[:,2]==0).sum()),'all_SAM_training_cache_point_labels_match_saved_teacher':True,'all_three_initializations_and_query_schedules_identical':True,'B_C_gradient_difference_confirms_SAM_term_enters_training':True,'arms':verification,'original_checkpoint_and_all_source_inputs_unchanged':True,'no_performance_or_geometry_truth_accuracy_claim':True})
 print('DONE independent feasibility verification',flush=True)
if __name__=='__main__':main()
