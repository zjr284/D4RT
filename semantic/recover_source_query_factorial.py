"""Resume a post-validation reporting failure without training or changing models."""
from pathlib import Path
import fcntl,gc,json
import numpy as np
import torch
from semantic import source_query_factorial as work
from semantic.model import SemanticTail,GEOMETRY_KEYS
from semantic.query_mode_adapter import QueryModeSemanticD4RT
from semantic.ytvis import load_video
from semantic.run import atomic_json,classification_metrics
from semantic import d4rt_finetune_readiness as ready
from tools.load_opend4rt import load_model
ROOT=work.ROOT;ARMS=work.ARMS;make_tail=work.make_tail
def corrected_full_interface(published,template,data,results):
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
       sel=np.where(cached['video_ids']==b['ids'][0])[0][:64];expected=cached['logits'if variant=='original'else'alternate_logits'][sel];assert scores.shape[0]==1 and scores[0].shape==expected.shape;assert np.array_equal(scores[0],expected)
     path=ROOT/'interface'/f'{arm}_seed{seed}_case{j}_{variant}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**{k:v.float().cpu().numpy()for k,v in out.items()});rows.append({'arm':arm,'seed':seed,'case':j,'batch_size':len(b['video']),'video_ids':b['ids'],'variant':variant,'path':str(path),'sha256':ready.sha(path),'all_geometry_exact_base':True})
   del interface;gc.collect();torch.cuda.empty_cache()
 del base;gc.collect();torch.cuda.empty_cache();atomic_json(ROOT/'interface_checks.json',{'cases':rows,'all108_geometry_exact_frozen_base':True,'all_B1_semantics_exact_cached_outputs':True,'source_target_camera_semantics_exact':True});return rows

def main():
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Already recovered; inspect existing summary')
  p=json.loads((ROOT/'protocol.json').read_text());am=json.loads((ROOT/'recovery_amendment.json').read_text());assert am['original_protocol_sha256']==ready.sha(ROOT/'protocol.json')
  for path,h in p['input_sha256'].items():assert ready.sha(path)==h,path
  for path,h in am['preserved_output_sha256'].items():assert ready.sha(path)==h,path
  for path,h in am['recovery_code_sha256'].items():assert ready.sha(path)==h,path
  results={arm:{}for arm in ARMS}
  for seed in range(3):
   initial=torch.load(ROOT/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
   for arm,(mode,head_only,sw,ew)in ARMS.items():
    final=torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);before=torch.load(ROOT/f'{arm}_seed{seed}_before_final.pt',map_location='cpu',weights_only=True);assert all(float(v['step'])==499 for v in before['optimizer']['state'].values());assert len(before['optimizer']['state'])==(2 if head_only else 10)
    with np.load(ROOT/f'{arm}_seed{seed}_validation.npz')as a:z=a['logits'].copy();alt=a['alternate_logits'].copy();y=a['labels'].copy();ids=a['video_ids'].copy()
    evaluation={'pooled':classification_metrics(z.argmax(1),y,26),'per_video':{str(int(v)):classification_metrics(z[ids==v].argmax(1),y[ids==v],26)for v in sorted(set(ids))},'target_camera_variant_agreement':float(np.mean(z.argmax(1)==alt.argmax(1)))}
    results[arm][str(seed)]={'initial_adapter_sha256':ready.state_sha(initial),'final_adapter_sha256':ready.state_sha(final),'trainable_parameters':33306 if head_only else 217626,'GT_exposures':32000,'SAM_exposures':16000 if sw else 0,'extra_GT_exposures':16000 if ew else 0,'trace':None,'original_stepwise_loss_trace_missing':True,'fixed_steps_from_preserved_pre_final_AdamW_state':500,'evaluation':evaluation}
  val=work.load_features('val');published=load_model(ready.CHECKPOINT,ready.CONFIG);torch.manual_seed(0);template=SemanticTail(published,26).eval();corrected_full_interface(published,template,val,results);comparisons={}
  for left,right in work.CONTRASTS:
   rows=[]
   for seed in range(3):
    l=results[left][str(seed)]['evaluation'];r=results[right][str(seed)]['evaluation'];d={v:l['per_video'][v]['macro_f1_present']-r['per_video'][v]['macro_f1_present']for v in r['per_video']};rows.append({'seed':seed,'pooled_macro_difference':l['pooled']['macro_f1_present']-r['pooled']['macro_f1_present'],'per_video_differences':d,'video_equal_mean_difference':float(np.mean(list(d.values())))})
   comparisons[left+'_minus_'+right]={'per_seed':rows,'mean_pooled_macro_difference':float(np.mean([r['pooled_macro_difference']for r in rows])),'mean_video_equal_difference':float(np.mean([r['video_equal_mean_difference']for r in rows]))}
  interactions=[]
  for seed in range(3):
   def v(arm):return results[arm][str(seed)]['evaluation']['pooled']['macro_f1_present']
   interactions.append((v('source_SAM')-v('source_GT'))-(v('raw_SAM')-v('raw_GT')))
  eligible=all(all(r['pooled_macro_difference']>0 for r in comparisons[k]['per_seed'])and comparisons[k]['mean_video_equal_difference']>0 for k in ('source_SAM_minus_source_GT','source_SAM_minus_source_extraGT'))
  for path,h in p['input_sha256'].items():assert ready.sha(path)==h,path
  for path,h in am['preserved_output_sha256'].items():assert ready.sha(path)==h,path
  atomic_json(ROOT/'summary.json',{'round':105,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'recovery_amendment_sha256':ready.sha(ROOT/'recovery_amendment.json'),'original_stepwise_loss_traces_missing_not_recreated':True,'results':results,'contrasts':comparisons,'canonical_SAM_pooled_interaction_by_seed':interactions,'source_SAM_eligible_for_future_external_protocol':eligible,'all108_interface_checks_verified_in_run':True,'geometry_main_weights_frozen':True,'geometry_truth_accuracy_evaluated':False,'default_changed':False,'new_confirmation':False,'feature_manifest_sha256':ready.sha(ROOT/'feature_manifest.json'),'interface_checks_sha256':ready.sha(ROOT/'interface_checks.json'),'outputs_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*.pt')}|{str(f):ready.sha(f)for f in ROOT.glob('*_validation.npz')}});print('DONE reporting recovery; no models retrained; eligibility',eligible,flush=True)
if __name__=='__main__':main()
