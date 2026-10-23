"""Same-coordinate raw/source semantic-factor training beside frozen geometry."""
from pathlib import Path
import copy,fcntl,gc,json,time
import numpy as np
import torch
import torch.nn.functional as F
from semantic.model import SemanticTail,GEOMETRY_KEYS
from semantic.query_mode_adapter import QueryModeSemanticD4RT
from semantic.ytvis import load_dataset,load_video,sample_queries
from semantic.run import atomic_json,classification_metrics
from semantic import d4rt_finetune_readiness as ready
from tools.load_opend4rt import load_model
ROOT=Path('semantic_artifacts/source_query_factorial')
ARMS={'raw_GT':('raw',False,0.,0.),'source_GT':('source',False,0.,0.),'raw_SAM':('raw',False,.25,0.),'source_SAM':('source',False,.25,0.),'source_head':('source',True,0.,0.),'source_extraGT':('source',False,0.,.25)}
CONTRASTS=(('source_SAM','source_GT'),('source_GT','source_head'),('source_GT','raw_GT'),('source_SAM','raw_SAM'),('source_SAM','source_extraGT'))

def query_for(clip,split,dataset):
 if split=='train':
  with np.load(ready.TEACHER/f"{clip['video_id']}.npz")as a:labels=a['labels'].copy()
 else:labels=np.stack([dataset.gt_label_map(clip,t)for t in range(8)])
 return sample_queries(labels,256,42+int(clip['video_id']))

def prepare(manifest):
 model=load_model(ready.CHECKPOINT,ready.CONFIG).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False);dataset=load_dataset('ovis',ready.ANN,Path('semantic_artifacts/ovis'));hashes={};drifts=[]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for split in ('train','val'):
   for clip in sorted(manifest[split],key=lambda c:int(c['video_id'])):
    vid=int(clip['video_id']);query,labels,xy=query_for(clip,split,dataset)
    with np.load(ready.CACHE/split/f'{vid}.npz')as a:
     assert np.array_equal(xy.numpy(),a['xy'])and np.array_equal(labels.numpy(),a['labels']);old_pre=a['prelast'].copy();old_memory=a['memory'].copy()
    video=load_video(clip)[None].cuda().to(torch.bfloat16);memory=model.encode_video(video);out={'raw':[],'source':[],'raw_alt':[]}
    for start in range(0,len(labels),64):
     q={k:v[:,start:start+64].cuda()for k,v in query.items()}
     for mode in ('raw','source','raw_alt'):
      if mode=='raw_alt'and split=='train':continue
      actual=dict(q)
      if mode=='source':actual['t_tgt']=actual['t_src'];actual['t_cam']=actual['t_src']
      if mode=='raw_alt':actual['t_tgt']=(q['t_tgt']+1)%8;actual['t_cam']=(q['t_cam']+1)%8
      z=model.query_embedder(video=video,u=actual['u'].to(video.dtype),v=actual['v'].to(video.dtype),t_src=actual['t_src'],t_tgt=actual['t_tgt'],t_cam=actual['t_cam'])
      for block in model.decoder.blocks[:-2]:z=block(z,memory)
      out[mode].append(z[0].float().cpu().numpy())
    raw=np.concatenate(out['raw']);source=np.concatenate(out['source']);mem=memory[0].float().cpu().numpy();path=ROOT/'features'/split/f'{vid}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    package={'raw':raw,'source':source,'memory':mem,'labels':labels.numpy().astype(np.int16),'xy':xy.numpy().astype(np.int16),**{'query_'+k:v[0].numpy()for k,v in query.items()}}
    if split=='val':package['raw_alt']=np.concatenate(out['raw_alt'])
    np.savez_compressed(path,**package);hashes[str(path)]=ready.sha(path);drifts.append({'split':split,'video_id':vid,'queries':len(labels),'raw_pre_vs_old_max_abs':float(np.abs(raw-old_pre.astype(np.float32)).max()),'memory_vs_old_max_abs':float(np.abs(mem-old_memory.astype(np.float32)).max()),'raw_vs_source_prefix_max_abs':float(np.abs(raw-source).max())});print('features',split,vid,flush=True)
 del model;gc.collect();torch.cuda.empty_cache();atomic_json(ROOT/'feature_manifest.json',{'sha256':hashes,'old_cache_differences':drifts});return hashes

def load_features(split):
 data={}
 for path in sorted((ROOT/'features'/split).glob('*.npz'),key=lambda p:int(p.stem)):
  with np.load(path)as a:r={k:a[k].copy()for k in a.files}
  for k in ('raw','source','raw_alt','memory'):
   if k in r:r[k]=torch.from_numpy(r[k]).cuda().to(torch.bfloat16)
  r['y']=torch.from_numpy(r['labels'].astype(np.int64)).cuda();data[int(path.stem)]=r
 return data

def make_tail(template,head_only,state=None):
 tail=copy.deepcopy(template).cuda().eval()
 if state is not None:
  with torch.no_grad():
   for n,q in tail.named_parameters():
    if n in state:q.copy_(state[n].to(q.device))
 if head_only:
  for n,q in tail.named_parameters():
   if n.endswith(('.a','.b')):q.requires_grad_(False)
 return tail

def evaluate(tail,data,mode):
 zs=[];alts=[];ys=[];ids=[];per={}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for vid,r in data.items():
   z=[];alt=[]
   for first in range(0,len(r['labels']),64):
    q=r[mode][first:first+64][None];z.append(tail(q,r['memory'][None])['semantic_logits'][0].float().cpu().numpy());aq=q if mode=='source'else r['raw_alt'][first:first+64][None];alt.append(tail(aq,r['memory'][None])['semantic_logits'][0].float().cpu().numpy())
   z=np.concatenate(z);alt=np.concatenate(alt);zs.append(z);alts.append(alt);ys.append(r['labels']);ids.extend([vid]*len(z));per[str(vid)]=classification_metrics(z.argmax(1),r['labels'],26)
 z=np.concatenate(zs);alt=np.concatenate(alts);y=np.concatenate(ys);return {'pooled':classification_metrics(z.argmax(1),y,26),'per_video':per,'target_camera_variant_agreement':float(np.mean(z.argmax(1)==alt.argmax(1)))},{'logits':z,'alternate_logits':alt,'labels':y,'video_ids':np.asarray(ids)}

def full_interface(published,template,data,results):
 base=load_model(ready.CHECKPOINT,ready.CONFIG).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False);manifest=json.loads(ready.MANIFEST.read_text());clips=sorted(manifest['val'],key=lambda c:int(c['video_id']))[:2];rows=[];batches=[]
 for clip in clips:
  vid=int(clip['video_id']);r=data[vid];q={k:torch.from_numpy(r['query_'+k][:64].copy()).cuda()[None]for k in ('u','v','t_src','t_tgt','t_cam')};batches.append({'video':load_video(clip)[None].cuda().to(torch.bfloat16),'query':q,'ids':[vid]})
 batches.append({'video':torch.cat([b['video']for b in batches]),'query':{k:torch.cat([b['query'][k]for b in batches])for k in batches[0]['query']},'ids':[int(c['video_id'])for c in clips]})
 for seed in range(3):
  for arm,(mode,head_only,_,_)in ARMS.items():
   interface=QueryModeSemanticD4RT(base,26,mode);interface.semantic_tail=make_tail(template,head_only,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True));interface.eval();original_scores={}
   for j,b in enumerate(batches):
    for variant in ('original','target_camera'):
     query=dict(b['query'])
     if variant=='target_camera':query['t_tgt']=(query['t_tgt']+1)%8;query['t_cam']=(query['t_cam']+1)%8
     batch={'video':b['video'],'query':query}
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=interface(batch);ref=base(batch)
     assert all(torch.equal(out[k],ref[k])for k in GEOMETRY_KEYS);scores=out['semantic_logits'].float().cpu().numpy()
     if variant=='original':original_scores[j]=scores
     if variant=='target_camera'and mode=='source':assert np.array_equal(original_scores[j],scores)
     if j<2:
      with np.load(ROOT/f'{arm}_seed{seed}_validation.npz')as cached:
       sel=np.where(cached['video_ids']==b['ids'][0])[0][:64];assert np.array_equal(scores,cached['logits'if variant=='original'else'alternate_logits'][sel])
     path=ROOT/'interface'/f'{arm}_seed{seed}_case{j}_{variant}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**{k:v.float().cpu().numpy()for k,v in out.items()});rows.append({'arm':arm,'seed':seed,'case':j,'batch_size':len(b['video']),'video_ids':b['ids'],'variant':variant,'path':str(path),'sha256':ready.sha(path),'all_geometry_exact_base':True})
   del interface;gc.collect();torch.cuda.empty_cache()
 del base;gc.collect();torch.cuda.empty_cache();atomic_json(ROOT/'interface_checks.json',{'cases':rows,'all108_geometry_exact_frozen_base':True,'all_B1_semantics_exact_cached_outputs':True,'source_target_camera_semantics_exact':True});return rows

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists()or(ROOT/'protocol.json').exists():raise RuntimeError('Existing run; inspect before explicit resume')
  parent=Path('semantic_artifacts/source_query_adapter_probe');assert json.loads((parent/'verification.json').read_text())['summary_sha256']==ready.sha(parent/'summary.json');pub=json.loads((parent/'publication.json').read_text());assert pub['remote_contains_research_commit'];manifest=json.loads(ready.MANIFEST.read_text());schedule_path=Path('semantic_artifacts/d4rt_controlled_finetune/protocol.json');schedules=json.loads(schedule_path.read_text())['schedules']
  files=[ready.CHECKPOINT,ready.CONFIG,ready.MANIFEST,ready.ANN,parent/'summary.json',parent/'verification.json',schedule_path,Path('semantic/model.py'),Path('semantic/ytvis.py'),Path('semantic/source_query_adapter.py'),Path('semantic/query_mode_adapter.py'),Path('semantic/source_query_factorial.py'),Path('semantic/OVIS_SOURCE_QUERY_FACTORIAL_PROTOCOL.md')]+list((ready.CACHE/'train').glob('*.npz'))+list((ready.CACHE/'val').glob('*.npz'))+[ready.TEACHER/f"{c['video_id']}.npz"for c in manifest['train']]+[Path(f)for split in ('train','val')for c in manifest[split]for f in c['frames']]
  p={'round':105,'source_commit':pub['research_commit'],'arms':{k:list(v)for k,v in ARMS.items()},'steps':500,'seeds':[0,1,2],'head_lr':.001,'LoRA_lr':.0001,'geometry_loss_in_semantic_branch':False,'input_sha256':{str(f):ready.sha(f)for f in sorted(set(files))},'new_confirmation':False};atomic_json(ROOT/'protocol.json',p);feature_hashes=prepare(manifest);train=load_features('train');published=load_model(ready.CHECKPOINT,ready.CONFIG);results={a:{}for a in ARMS}
  for seed in range(3):
   torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=ready.weights(template);torch.save(initial,ROOT/f'initial_seed{seed}.pt')
   for arm,(mode,head_only,sam_weight,extra_weight)in ARMS.items():
    tail=make_tail(template,head_only);frozen=ready.frozen_sha(tail);opt=torch.optim.AdamW([{'params':[q for n,q in tail.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},{'params':[q for n,q in tail.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01);trace=[]
    for step,row in enumerate(schedules[str(seed)]):
     g=train[row['GT_video']];a=train[row['aux_video']];gi=torch.tensor(row['GT_indices']+row['extra_GT_indices'],device='cuda');ai=torch.tensor(row['SAM_indices'],device='cuda')
     with torch.autocast('cuda',dtype=torch.bfloat16):go=tail(g[mode][gi][None],g['memory'][None])['semantic_logits'][0].float();ao=tail(a[mode][ai][None],a['memory'][None])['semantic_logits'][0].float()
     ce=F.cross_entropy(go[:64],g['y'][gi[:64]]);sam=F.cross_entropy(ao,a['y'][ai]);extra=F.cross_entropy(go[64:],g['y'][gi[64:]]);loss=ce+sam_weight*sam+extra_weight*extra;opt.zero_grad(set_to_none=True);loss.backward();assert bool(torch.isfinite(loss))and all(q.grad is None for q in tail.parameters()if not q.requires_grad)
     if step==0:torch.save({'gradients':{n:q.grad.detach().cpu().clone()for n,q in tail.named_parameters()if q.grad is not None}},ROOT/f'{arm}_seed{seed}_step0.pt')
     if step==499:torch.save({'adapter':ready.weights(tail),'optimizer':opt.state_dict()},ROOT/f'{arm}_seed{seed}_before_final.pt')
     opt.step();trace.append({'step':step+1,'GT_CE':float(ce.detach()),'SAM_CE':float(sam.detach()),'extra_GT_CE':float(extra.detach()),'total':float(loss.detach())})
     if(step+1)%100==0:print('train',seed,arm,step+1,flush=True)
    assert frozen==ready.frozen_sha(tail);final=ready.weights(tail);torch.save(final,ROOT/f'{arm}_seed{seed}.pt');results[arm][str(seed)]={'initial_adapter_sha256':ready.state_sha(initial),'final_adapter_sha256':ready.state_sha(final),'trainable_parameters':sum(q.numel()for q in tail.parameters()if q.requires_grad),'GT_exposures':32000,'SAM_exposures':16000 if sam_weight else 0,'extra_GT_exposures':16000 if extra_weight else 0,'trace':trace,'frozen_tail_unchanged':True};del tail,opt;gc.collect();torch.cuda.empty_cache()
   del template;gc.collect();torch.cuda.empty_cache()
  atomic_json(ROOT/'frozen_selection.json',{'rule':'all18 final500 weights before old validation scoring','adapters_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*_seed[012].pt')},'protocol_sha256':ready.sha(ROOT/'protocol.json')});del train;gc.collect();torch.cuda.empty_cache();val=load_features('val')
  for seed in range(3):
   torch.manual_seed(seed);template=SemanticTail(published,26).eval()
   for arm,(mode,head_only,_,_)in ARMS.items():
    tail=make_tail(template,head_only,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True));metrics,arrays=evaluate(tail,val,mode);np.savez_compressed(ROOT/f'{arm}_seed{seed}_validation.npz',**arrays);results[arm][str(seed)]['evaluation']=metrics;print('val',seed,arm,metrics['pooled']['macro_f1_present'],flush=True);del tail;gc.collect();torch.cuda.empty_cache()
   del template;gc.collect();torch.cuda.empty_cache()
  torch.manual_seed(0);template=SemanticTail(published,26).eval();interface_rows=full_interface(published,template,val,results);comparisons={}
  for left,right in CONTRASTS:
   rows=[]
   for seed in range(3):
    l=results[left][str(seed)]['evaluation'];r=results[right][str(seed)]['evaluation'];d={v:l['per_video'][v]['macro_f1_present']-r['per_video'][v]['macro_f1_present']for v in r['per_video']};rows.append({'seed':seed,'pooled_macro_difference':l['pooled']['macro_f1_present']-r['pooled']['macro_f1_present'],'per_video_differences':d,'video_equal_mean_difference':float(np.mean(list(d.values())))})
   comparisons[left+'_minus_'+right]={'per_seed':rows,'mean_pooled_macro_difference':float(np.mean([r['pooled_macro_difference']for r in rows])),'mean_video_equal_difference':float(np.mean([r['video_equal_mean_difference']for r in rows]))}
  interactions=[]
  for seed in range(3):
   def macro(a):return results[a][str(seed)]['evaluation']['pooled']['macro_f1_present']
   interactions.append((macro('source_SAM')-macro('source_GT'))-(macro('raw_SAM')-macro('raw_GT')))
  eligible=all(all(r['pooled_macro_difference']>0 for r in comparisons[key]['per_seed'])and comparisons[key]['mean_video_equal_difference']>0 for key in ('source_SAM_minus_source_GT','source_SAM_minus_source_extraGT'))
  for path,h in p['input_sha256'].items():assert ready.sha(path)==h,path
  for path,h in feature_hashes.items():assert ready.sha(path)==h,path
  atomic_json(ROOT/'summary.json',{'round':105,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'results':results,'contrasts':comparisons,'canonical_SAM_pooled_interaction_by_seed':interactions,'source_SAM_eligible_for_future_external_protocol':eligible,'all108_interface_checks_verified_in_run':True,'geometry_main_weights_frozen':True,'geometry_truth_accuracy_evaluated':False,'default_changed':False,'new_confirmation':False,'feature_manifest_sha256':ready.sha(ROOT/'feature_manifest.json'),'interface_checks_sha256':ready.sha(ROOT/'interface_checks.json'),'outputs_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*.pt')}|{str(f):ready.sha(f)for f in ROOT.glob('*_validation.npz')}});print('DONE source-query factorial; external eligibility',eligible,flush=True)
if __name__=='__main__':main()
