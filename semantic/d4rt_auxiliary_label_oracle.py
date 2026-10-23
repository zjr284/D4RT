"""Same-query SAM/GT auxiliary-label diagnosis; oracle is not a deployable method."""
from pathlib import Path
import fcntl,gc,json
import numpy as np
import torch
import torch.nn.functional as F
from semantic.model import SemanticTail,geometry_vector
from semantic.run import atomic_json
from semantic.ytvis import load_dataset
from semantic import d4rt_finetune_readiness as ready
from semantic import d4rt_matched_head_finetune as source
from tools.load_opend4rt import load_model
ROOT=Path('semantic_artifacts/d4rt_auxiliary_label_oracle');ARMS=('sam_common','oracle_common')

def objective(model,train,truth,row,rg,ra,arm):
 gt,aux,gi,ai=source.batches(train,row)
 with torch.autocast('cuda',dtype=torch.bfloat16):go=model(gt['prelast'][gi][None],gt['memory'][None]);ao=model(aux['prelast'][ai][None],aux['memory'][None])
 ce=F.cross_entropy(go['semantic_logits'][0,:64].float(),gt['target'][gi[:64]])
 oracle=truth[row['aux_video']][ai];valid=oracle>=0;target=aux['target'][ai]if arm=='sam_common'else oracle
 aux_loss=F.cross_entropy(ao['semantic_logits'][0,valid].float(),target[valid])if bool(valid.any())else ao['semantic_logits'].float().sum()*0
 geo=F.smooth_l1_loss(torch.cat([geometry_vector(go),geometry_vector(ao)],1).float(),torch.cat([rg,ra],1).float())
 return ce,aux_loss,geo,int(valid.sum())

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'protocol.json').exists()or(ROOT/'summary.json').exists():raise RuntimeError('Inspect previous run; never overwrite')
  previous=json.loads((source.ROOT/'summary.json').read_text());assert ready.sha(source.ROOT/'summary.json')==json.loads((source.ROOT/'verification.json').read_text())['summary_sha256'];pub=json.loads((source.ROOT/'publication.json').read_text());assert pub['remote_contains_research_commit'];schedules=json.loads((source.ROOT/'protocol.json').read_text())['schedules'];train=source.load_data('train');manifest=json.loads(ready.MANIFEST.read_text());clips={int(c['video_id']):c for c in manifest['train']};dataset=load_dataset('ovis',ready.ANN,Path('semantic_artifacts/ovis'));truth={};confusion=np.zeros((26,26),np.int64);ignored=total=0;bindings={}
  for vid,r in train.items():
   maps=np.stack([dataset.gt_label_map(clips[vid],t)for t in range(8)]);xy=r['xy'];y=maps[xy[:,2],xy[:,1],xy[:,0]].astype(np.int64);assert np.array_equal(y[xy[:,2]==0],r['labels'][xy[:,2]==0]);later=(xy[:,2]>0)&(r['labels']>0);valid=later&(y>=0);np.add.at(confusion,(y[valid],r['labels'][valid]),1);ignored+=int((later&(y<0)).sum());total+=int(later.sum());truth[vid]=torch.from_numpy(y).cuda();path=ROOT/'associations'/f'{vid}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,GT_labels=y,SAM_labels=r['labels'],xy=xy);bindings[str(path)]=ready.sha(path)
  counts={seed:[int((truth[row['aux_video']][torch.tensor(row['SAM_indices'],device='cuda')]>=0).sum())for row in rows]for seed,rows in schedules.items()}
  files=[ready.CHECKPOINT,ready.CONFIG,ready.MANIFEST,ready.ANN,source.ROOT/'summary.json',source.ROOT/'verification.json',source.ROOT/'protocol.json',Path('semantic/model.py'),Path('semantic/d4rt_matched_head_finetune.py'),Path('semantic/d4rt_auxiliary_label_oracle.py'),Path('semantic/OVIS_D4RT_AUXILIARY_LABEL_ORACLE_PROTOCOL.md')]+list((ready.CACHE/'train').glob('*.npz'))+list((ready.CACHE/'val').glob('*.npz'))+[ready.TEACHER/f'{vid}.npz'for vid in train]+[source.ROOT/f'initial_seed{i}.pt'for i in range(3)]+[source.ROOT/f'references_seed{i}.npz'for i in range(3)]
  p={'round':103,'source_commit':pub['research_commit'],'steps':500,'arms':list(ARMS),'head_lr':.001,'LoRA_lr':.0001,'aux_coefficient':.25,'geometry_coefficient':.1,'input_sha256':{str(f):ready.sha(f)for f in files},'GT_binding_sha256':bindings,'schedule_common_valid_aux_counts':counts,'query_conditional_teacher_confusion_GT_rows_SAM_columns':confusion.tolist(),'query_conditional_later_SAM_FG_count':total,'query_conditional_later_ignored_GT_count':ignored,'oracle_is_extra_full_GT_supervision_not_SAM_method':True,'new_confirmation':False};atomic_json(ROOT/'protocol.json',p);print('Frozen query-conditional label audit',total,'ignored',ignored,'correct',int(np.trace(confusion)),flush=True)
  published=load_model(ready.CHECKPOINT,ready.CONFIG);results={a:{}for a in ARMS}
  for seed in range(3):
   torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(source.ROOT/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
   with np.load(source.ROOT/f'references_seed{seed}.npz')as a:rg=torch.from_numpy(a['GT'].copy()).cuda();ra=torch.from_numpy(a['aux'].copy()).cuda()
   for arm in ARMS:
    model=source.make_model(template,'C',initial);frozen=ready.frozen_sha(model);opt=torch.optim.AdamW([{'params':[q for n,q in model.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},{'params':[q for n,q in model.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01);trace=[]
    for step,row in enumerate(schedules[str(seed)]):
     ce,aux,geo,k=objective(model,train,truth,row,rg[step][None],ra[step][None],arm);assert k==counts[str(seed)][step];loss=ce+.1*geo+.25*aux;opt.zero_grad(set_to_none=True);loss.backward();assert bool(torch.isfinite(loss))and all(q.grad is None for q in model.parameters()if not q.requires_grad)
     if step==0:torch.save({'adapter':ready.weights(model),'gradients':{n:q.grad.detach().cpu().clone()for n,q in model.named_parameters()if q.grad is not None}},ROOT/f'{arm}_seed{seed}_step0.pt')
     if step==499:torch.save({'adapter':ready.weights(model),'optimizer':opt.state_dict()},ROOT/f'{arm}_seed{seed}_before_final.pt')
     opt.step();trace.append({'step':step+1,'GT_CE':float(ce.detach()),'auxiliary_CE':float(aux.detach()),'geometry_smooth_l1':float(geo.detach()),'valid_aux_points':k,'total':float(loss.detach())})
     if(step+1)%100==0:print('train',seed,arm,step+1,flush=True)
    assert frozen==ready.frozen_sha(model);final=ready.weights(model);torch.save(final,ROOT/f'{arm}_seed{seed}.pt');results[arm][str(seed)]={'initial_adapter_sha256':ready.state_sha(initial),'final_adapter_sha256':ready.state_sha(final),'GT_exposures':32000,'common_aux_exposures':sum(counts[str(seed)]),'empty_aux_batches':counts[str(seed)].count(0),'trace':trace,'frozen_tail_unchanged':True};del model,opt;gc.collect();torch.cuda.empty_cache()
   del template,rg,ra;gc.collect();torch.cuda.empty_cache()
  atomic_json(ROOT/'frozen_selection.json',{'rule':'all six final500 adapters frozen before old validation scoring','adapters_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*_seed[012].pt')},'protocol_sha256':ready.sha(ROOT/'protocol.json')});del train,truth;gc.collect();torch.cuda.empty_cache();val=source.load_data('val')
  for seed in range(3):
   torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(source.ROOT/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True);reference=source.make_model(template,'A',initial).requires_grad_(False)
   for arm in ARMS:
    model=source.make_model(template,'C',torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True));metrics,arrays=source.evaluate(model,reference,val);np.savez_compressed(ROOT/f'{arm}_seed{seed}_validation.npz',**arrays);results[arm][str(seed)]['evaluation']=metrics;print('val',seed,arm,metrics['pooled']['macro_f1_present'],flush=True);del model;gc.collect();torch.cuda.empty_cache()
   del reference,template;gc.collect();torch.cuda.empty_cache()
  contrasts=[]
  for seed in range(3):
   o=results['oracle_common'][str(seed)]['evaluation'];a=results['sam_common'][str(seed)]['evaluation'];d={vid:o['per_video'][vid]['macro_f1_present']-a['per_video'][vid]['macro_f1_present']for vid in a['per_video']};contrasts.append({'seed':seed,'pooled_macro_difference':o['pooled']['macro_f1_present']-a['pooled']['macro_f1_present'],'per_video_differences':d,'video_equal_mean_difference':float(np.mean(list(d.values())))})
  supports=all(c['pooled_macro_difference']>0 for c in contrasts)and np.mean([c['video_equal_mean_difference']for c in contrasts])>0
  for path,h in p['input_sha256'].items():assert ready.sha(path)==h
  for path,h in bindings.items():assert ready.sha(path)==h
  atomic_json(ROOT/'summary.json',{'round':103,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'results':results,'oracle_minus_SAM_common':contrasts,'label_noise_contributes_under_fixed_queries_rule_satisfied':bool(supports),'oracle_eligible_for_external_or_deployment':False,'new_confirmation':False,'geometry_truth_accuracy_evaluated':False,'output_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*.pt')}|{str(f):ready.sha(f)for f in ROOT.glob('*.npz')}});print('DONE label-oracle diagnostic; support rule',supports,flush=True)
if __name__=='__main__':main()
