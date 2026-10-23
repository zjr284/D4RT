"""Independent fresh-prefix, optimizer, validation and geometry-interface verification."""
from pathlib import Path
import copy,gc,json
import numpy as np
import torch
from semantic.model import SemanticTail,GEOMETRY_KEYS
from semantic.ytvis import load_dataset,load_video
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha,metrics,close_dict
from tools.load_opend4rt import load_model,DEFAULT_VARIANT
ROOT=Path('semantic_artifacts/source_query_factorial');CACHE=Path('semantic_artifacts/features_ovis/sam')
ARMS={'raw_GT':('raw',False,0.,0.),'source_GT':('source',False,0.,0.),'raw_SAM':('raw',False,.25,0.),'source_SAM':('source',False,.25,0.),'source_head':('source',True,0.,0.),'source_extraGT':('source',False,0.,.25)}
CONTRASTS=(('source_SAM','source_GT'),('source_GT','source_head'),('source_GT','raw_GT'),('source_SAM','raw_SAM'),('source_SAM','source_extraGT'))
def hash_tensor(q):
 import hashlib
 return hashlib.sha256(q.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def managed(n):return n.startswith('semantic_head.')or n.endswith(('.a','.b'))

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());assert p['round']==105 and p['steps']==500 and p['arms']=={k:list(v)for k,v in ARMS.items()}and s['protocol_sha256']==sha(ROOT/'protocol.json')
 for path,h in p['input_sha256'].items():assert sha(path)==h,path
 am=json.loads((ROOT/'recovery_amendment_v2.json').read_text());assert s['recovery_amendment_sha256']==sha(ROOT/'recovery_amendment_v2.json')and am['models_retrained']==0 and am['original_training_code_unchanged']and am['evaluation_and_eligibility_rules_unchanged'];assert s['original_stepwise_loss_traces_missing_not_recreated'] and s['semantic_prefix_runtime_cast_to_training_BF16'] and am['semantic_prefix_runtime_cast_to_training_BF16'];assert am['parent_amendment_sha256']==sha(ROOT/'recovery_amendment.json');parent=json.loads((ROOT/'recovery_amendment.json').read_text())
 for path,h in parent['recovery_code_sha256'].items():assert sha(path)==h,path
 for path,h in am['preserved_output_sha256'].items():assert sha(path)==h,path
 for path,h in am['recovery_code_sha256'].items():assert sha(path)==h,path
 for path,h in s['outputs_sha256'].items():assert sha(path)==h,path
 f=json.loads((ROOT/'feature_manifest.json').read_text());assert sha(ROOT/'feature_manifest.json')==s['feature_manifest_sha256']and len(f['sha256'])==80
 for path,h in f['sha256'].items():assert sha(path)==h
 frozen=json.loads((ROOT/'frozen_selection.json').read_text());required={str(ROOT/f'{arm}_seed{seed}.pt')for arm in ARMS for seed in range(3)};assert required<=set(frozen['adapters_sha256']);assert set(frozen['adapters_sha256'])-required=={str(ROOT/f'initial_seed{seed}.pt')for seed in range(3)}
 for path,h in frozen['adapters_sha256'].items():assert sha(path)==h
 manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text());dataset=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'));base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
 def prefix(video,q,memory,mode):
  src=q['t_src'];tgt=src if mode=='source'else q['t_tgt'];cam=src if mode=='source'else q['t_cam'];x=base.query_embedder(video=video,u=q['u'].to(video.dtype),v=q['v'].to(video.dtype),t_src=src,t_tgt=tgt,t_cam=cam)
  for block in base.decoder.blocks[:-2]:x=block(x,memory)
  return x
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for split in ('train','val'):
   for c in sorted(manifest[split],key=lambda c:int(c['video_id'])):
    vid=int(c['video_id'])
    if split=='train':
     with np.load(Path('semantic_artifacts/teacher_ovis')/f'{vid}.npz')as a:maps=a['labels'].copy()
    else:maps=np.stack([dataset.gt_label_map(c,t)for t in range(8)])
    rng=np.random.default_rng(42+vid);coords=[]
    for t in range(8):
     for pool in (np.argwhere(maps[t]>0),np.argwhere(maps[t]==0)):
      if len(pool):coords.extend((int(u),int(v),t)for v,u in pool[rng.choice(len(pool),128,replace=len(pool)<128)])
    xy=np.asarray(coords,dtype=np.int64);n=len(xy);tgt=rng.integers(0,8,n);cam=rng.integers(0,8,n);y=maps[xy[:,2],xy[:,1],xy[:,0]]
    with np.load(ROOT/'features'/split/f'{vid}.npz')as a:stored={k:a[k].copy()for k in a.files}
    with np.load(CACHE/split/f'{vid}.npz')as a:assert np.array_equal(xy,a['xy'])and np.array_equal(y,a['labels']);oldpre=a['prelast'].copy();oldmem=a['memory'].copy()
    assert np.array_equal(xy,stored['xy'])and np.array_equal(y,stored['labels'])and np.array_equal(tgt,stored['query_t_tgt'])and np.array_equal(cam,stored['query_t_cam'])
    u=(xy[:,0]/127).astype(np.float32);v=(xy[:,1]/127).astype(np.float32);assert np.array_equal(u,stored['query_u'])and np.array_equal(v,stored['query_v'])and np.array_equal(xy[:,2],stored['query_t_src'])
    video=load_video(c)[None].cuda().to(torch.bfloat16);memory=base.encode_video(video);assert np.array_equal(memory[0].float().cpu().numpy(),stored['memory'])
    for start in range(0,n,64):
     q={'u':torch.from_numpy(u[start:start+64]).cuda()[None],'v':torch.from_numpy(v[start:start+64]).cuda()[None],'t_src':torch.from_numpy(xy[start:start+64,2]).cuda()[None],'t_tgt':torch.from_numpy(tgt[start:start+64]).cuda()[None],'t_cam':torch.from_numpy(cam[start:start+64]).cuda()[None]}
     for mode in ('raw','source'):
      got=prefix(video,q,memory,mode)[0].float().cpu().numpy();assert np.array_equal(got,stored[mode][start:start+64])
     if split=='val':
      alt=dict(q);alt['t_tgt']=(alt['t_tgt']+1)%8;alt['t_cam']=(alt['t_cam']+1)%8;assert np.array_equal(prefix(video,alt,memory,'raw')[0].float().cpu().numpy(),stored['raw_alt'][start:start+64])
    drift=next(r for r in f['old_cache_differences']if r['split']==split and r['video_id']==vid);assert drift['queries']==n and drift['raw_pre_vs_old_max_abs']==float(np.abs(stored['raw']-oldpre.astype(np.float32)).max())and drift['memory_vs_old_max_abs']==float(np.abs(stored['memory']-oldmem.astype(np.float32)).max())and drift['raw_vs_source_prefix_max_abs']==float(np.abs(stored['raw']-stored['source']).max())
   print('Verified all fresh paired prefixes and original queries',split,flush=True)
 del base;gc.collect();torch.cuda.empty_cache();published=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml');schedules=json.loads(Path('semantic_artifacts/d4rt_controlled_finetune/protocol.json').read_text())['schedules']
 def data(split):
  out={}
  for path in sorted((ROOT/'features'/split).glob('*.npz'),key=lambda a:int(a.stem)):
   with np.load(path)as a:r={k:a[k].copy()for k in a.files}
   for k in ('raw','source','raw_alt','memory'):
    if k in r:r[k]=torch.from_numpy(r[k]).cuda().to(torch.bfloat16)
   r['y']=torch.from_numpy(r['labels'].astype(np.int64)).cuda();out[int(path.stem)]=r
  return out
 def model_for(template,head_only,state):
  m=copy.deepcopy(template).cuda().eval()
  with torch.no_grad():
   for n,q in m.named_parameters():
    if n in state:q.copy_(state[n].to(q.device))
    if head_only and n.endswith(('.a','.b')):q.requires_grad_(False)
  return m
 def optimizer(m):return torch.optim.AdamW([{'params':[q for n,q in m.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},{'params':[q for n,q in m.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01)
 train=data('train');replays={}
 def loss(m,row,mode,sw,ew):
  g=train[row['GT_video']];a=train[row['aux_video']];gi=torch.tensor(row['GT_indices']+row['extra_GT_indices'],device='cuda');ai=torch.tensor(row['SAM_indices'],device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):go=m(g[mode][gi][None],g['memory'][None])['semantic_logits'][0].float();ao=m(a[mode][ai][None],a['memory'][None])['semantic_logits'][0].float()
  gl=go.log_softmax(-1);al=ao.log_softmax(-1);gt=-gl[torch.arange(64,device='cuda'),g['y'][gi[:64]]].mean();extra=-gl[torch.arange(64,96,device='cuda'),g['y'][gi[64:]]].mean();aux=-al[torch.arange(32,device='cuda'),a['y'][ai]].mean();return gt+sw*aux+ew*extra
 for seed in range(3):
  torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(ROOT/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True);ih={n:hash_tensor(q)for n,q in initial.items()}
  for arm,(mode,ho,sw,ew)in ARMS.items():
   logged=s['results'][arm][str(seed)];assert logged['initial_adapter_sha256']==ih and logged['GT_exposures']==32000 and logged['SAM_exposures']==(16000 if sw else 0)and logged['extra_GT_exposures']==(16000 if ew else 0);final=torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);assert {n:hash_tensor(q)for n,q in final.items()}==logged['final_adapter_sha256']
   if ho:assert all(torch.equal(final[n],initial[n])for n in final if n.endswith(('.a','.b')))
   m=model_for(template,ho,initial);assert sum(q.numel()for q in m.parameters()if q.requires_grad)==logged['trainable_parameters']==(33306 if ho else 217626);frozen_hash={n:hash_tensor(q)for n,q in m.named_parameters()if not managed(n)};l=loss(m,schedules[str(seed)][0],mode,sw,ew);l.backward();assert logged['trace']is None and logged['original_stepwise_loss_trace_missing'];stored=torch.load(ROOT/f'{arm}_seed{seed}_step0.pt',map_location='cpu',weights_only=True)['gradients'];grads={n:q.grad for n,q in m.named_parameters()if q.grad is not None};assert set(grads)==set(stored);ge=0.
   for n,g in grads.items():assert managed(n);torch.testing.assert_close(g.cpu(),stored[n],rtol=5e-4,atol=5e-6);ge=max(ge,float((g.cpu()-stored[n]).abs().max()))
   del m;gc.collect();torch.cuda.empty_cache();before=torch.load(ROOT/f'{arm}_seed{seed}_before_final.pt',map_location='cpu',weights_only=True);m=model_for(template,ho,before['adapter']);assert all(float(v['step'])==499 for v in before['optimizer']['state'].values());opt=optimizer(m);opt.load_state_dict(before['optimizer']);l=loss(m,schedules[str(seed)][-1],mode,sw,ew);opt.zero_grad(set_to_none=True);l.backward();opt.step();pe=0.
   for n,q in m.named_parameters():
    if n in final:torch.testing.assert_close(q.detach().cpu(),final[n],rtol=3e-4,atol=2e-6);pe=max(pe,float((q.detach().cpu()-final[n]).abs().max()))
   assert frozen_hash=={n:hash_tensor(q)for n,q in m.named_parameters()if not managed(n)}
   assert logged['fixed_steps_from_preserved_pre_final_AdamW_state']==500 and len(schedules[str(seed)])==500
   replays[f'{arm}_{seed}']={'first_gradient_max_abs_error':ge,'last_optimizer_parameter_max_abs_error':pe};del m,opt;gc.collect();torch.cuda.empty_cache()
  print('Verified six gradient/optimizer pairs seed',seed,flush=True);del template;gc.collect();torch.cuda.empty_cache()
 del train;gc.collect();torch.cuda.empty_cache();val=data('val');rebuilt={a:{}for a in ARMS};confusions={}
 for seed in range(3):
  torch.manual_seed(seed);template=SemanticTail(published,26).eval()
  for arm,(mode,ho,_,_)in ARMS.items():
   m=model_for(template,ho,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True));saved=np.load(ROOT/f'{arm}_seed{seed}_validation.npz');offset=0;per={}
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for vid,r in val.items():
     z=[];alt=[]
     for start in range(0,len(r['labels']),64):q=r[mode][start:start+64][None];z.append(m(q,r['memory'][None])['semantic_logits'][0].float().cpu().numpy());aq=q if mode=='source'else r['raw_alt'][start:start+64][None];alt.append(m(aq,r['memory'][None])['semantic_logits'][0].float().cpu().numpy())
     z=np.concatenate(z);alt=np.concatenate(alt);n=len(z);assert np.array_equal(z,saved['logits'][offset:offset+n])and np.array_equal(alt,saved['alternate_logits'][offset:offset+n])and np.array_equal(r['labels'],saved['labels'][offset:offset+n])and np.all(saved['video_ids'][offset:offset+n]==vid);per[str(vid)]=metrics(r['labels'],z.argmax(1))[0];offset+=n
   assert offset==40960;pooled,cm=metrics(saved['labels'],saved['logits'].argmax(1));logged=s['results'][arm][str(seed)]['evaluation'];close_dict(pooled,logged['pooled']);close_dict(per,logged['per_video']);assert float(np.mean(saved['logits'].argmax(1)==saved['alternate_logits'].argmax(1)))==logged['target_camera_variant_agreement'];rebuilt[arm][str(seed)]={'pooled':pooled,'per_video':per};confusions[f'{arm}_{seed}']=cm.tolist();saved.close();del m;gc.collect();torch.cuda.empty_cache()
  print('Verified six full validation models seed',seed,flush=True);del template;gc.collect();torch.cuda.empty_cache()
 # Manual end-to-end geometry/canonical checks using the original base, not the wrapper.
 base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False);clips=sorted(manifest['val'],key=lambda c:int(c['video_id']))[:2];bs=[]
 for c in clips:
  r=val[int(c['video_id'])];q={k:torch.from_numpy(r['query_'+k][:64].copy()).cuda()[None]for k in ('u','v','t_src','t_tgt','t_cam')};bs.append({'video':load_video(c)[None].cuda().to(torch.bfloat16),'query':q,'ids':[int(c['video_id'])]})
 bs.append({'video':torch.cat([b['video']for b in bs]),'query':{k:torch.cat([b['query'][k]for b in bs])for k in bs[0]['query']},'ids':[int(c['video_id'])for c in clips]});manual={}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for j,b in enumerate(bs):
   for variant in ('original','target_camera'):
    q=dict(b['query'])
    if variant=='target_camera':q['t_tgt']=(q['t_tgt']+1)%8;q['t_cam']=(q['t_cam']+1)%8
    batch={'video':b['video'],'query':q};memory=base.encode_video(b['video']);manual[(j,variant)]={'geometry':base(batch),'memory':memory,'prefix':{mode:prefix(b['video'],q,memory,mode)for mode in ('raw','source')}}
 checks=json.loads((ROOT/'interface_checks.json').read_text());assert sha(ROOT/'interface_checks.json')==s['interface_checks_sha256']and len(checks['cases'])==108
 torch.manual_seed(0);template=SemanticTail(published,26).eval()
 for seed in range(3):
  for arm,(mode,ho,_,_)in ARMS.items():
   m=model_for(template,ho,torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True));original={}
   for record in [r for r in checks['cases']if r['seed']==seed and r['arm']==arm]:
    j=record['case'];variant=record['variant'];assert record['video_ids']==bs[j]['ids']and record['batch_size']==len(bs[j]['video'])and sha(record['path'])==record['sha256'];expected=manual[(j,variant)]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=m(expected['prefix'][mode].to(torch.bfloat16),expected['memory'])['semantic_logits'].float().cpu().numpy()
    with np.load(record['path'])as saved:
     assert np.array_equal(z,saved['semantic_logits'])
     for k in GEOMETRY_KEYS:assert np.array_equal(expected['geometry'][k].float().cpu().numpy(),saved[k])
    if variant=='original':original[j]=z
    if variant=='target_camera'and mode=='source':assert np.array_equal(original[j],z)
    if j<2:
     with np.load(ROOT/f'{arm}_seed{seed}_validation.npz')as saved:idx=np.flatnonzero(saved['video_ids']==bs[j]['ids'][0])[:64];assert np.array_equal(z[0],saved['logits'if variant=='original'else'alternate_logits'][idx])
   del m;gc.collect();torch.cuda.empty_cache()
  print('Verified36 full interface cases seed',seed,flush=True)
 for left,right in CONTRASTS:
  rows=[]
  for seed in range(3):
   l=rebuilt[left][str(seed)];r=rebuilt[right][str(seed)];d={v:l['per_video'][v]['macro_f1_present']-r['per_video'][v]['macro_f1_present']for v in r['per_video']};rows.append({'seed':seed,'pooled_macro_difference':l['pooled']['macro_f1_present']-r['pooled']['macro_f1_present'],'per_video_differences':d,'video_equal_mean_difference':float(np.mean(list(d.values())))})
  got=s['contrasts'][left+'_minus_'+right];assert got['per_seed']==rows and abs(got['mean_pooled_macro_difference']-np.mean([r['pooled_macro_difference']for r in rows]))<1e-12 and abs(got['mean_video_equal_difference']-np.mean([r['video_equal_mean_difference']for r in rows]))<1e-12
 interaction=[]
 for seed in range(3):
  def value(a):return rebuilt[a][str(seed)]['pooled']['macro_f1_present']
  interaction.append((value('source_SAM')-value('source_GT'))-(value('raw_SAM')-value('raw_GT')))
 assert interaction==s['canonical_SAM_pooled_interaction_by_seed'];eligible=all(all(r['pooled_macro_difference']>0 for r in s['contrasts'][key]['per_seed'])and s['contrasts'][key]['mean_video_equal_difference']>0 for key in ('source_SAM_minus_source_GT','source_SAM_minus_source_extraGT'));assert eligible==s['source_SAM_eligible_for_future_external_protocol']
 for path,h in p['input_sha256'].items():assert sha(path)==h,path
 for path,h in f['sha256'].items():assert sha(path)==h,path
 atomic_json(ROOT/'verification.json',{'round':105,'semantic_prefix_runtime_BF16_matches_training_verified':True,'summary_sha256':sha(ROOT/'summary.json'),'verifier_sha256':sha('semantic/verify_source_query_factorial.py'),'all80_original_queries_RGB_memory_and_paired_prefixes_rebuilt_exact':True,'all18_gradient_and_final_optimizer_replays':replays,'all18_complete_original_and_alternate_validation_logits_exact':True,'all_confusions':confusions,'all108_manual_interface_geometry_semantics_and_B1_cache_checks_exact':True,'all_five_contrasts_interaction_and_eligibility_verified':True,'all_source_fingerprints_unchanged':True,'frozen_ledger_extra_three_initial_states_not_evaluated_as_models':True,'original_loss_traces_missing_not_fabricated_and_saved_optimizer_budget_verified':True,'recovery_amendment_sha256':sha(ROOT/'recovery_amendment_v2.json'),'new_confirmation':False});print('DONE independent source-query factorial verification',flush=True)
if __name__=='__main__':main()
