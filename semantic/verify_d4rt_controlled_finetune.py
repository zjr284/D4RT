"""Independent schedules, labels, gradients, optimizer and validation replay."""
from collections import defaultdict
import copy,gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_api
from semantic.model import SemanticTail,GEOMETRY_KEYS
from semantic.run import atomic_json
from tools.load_opend4rt import load_model,DEFAULT_VARIANT
ROOT=Path('semantic_artifacts/d4rt_controlled_finetune');CACHE=Path('semantic_artifacts/features_ovis/sam');ARMS=('A','B','C','D');COEF={'A':(0.,0.,0.),'B':(.1,0.,0.),'C':(.1,.25,0.),'D':(.1,0.,.25)}
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def managed(n):return n.startswith('semantic_head.')or n.endswith(('.a','.b'))
def flat(out):return torch.cat([out[k].reshape(1,-1,int(np.prod(out[k].shape[2:])))for k in GEOMETRY_KEYS],dim=-1).float()
def metrics(y,pred):
 m=np.zeros((26,26),np.int64);np.add.at(m,(y,pred),1);n=m.sum(1);p=m.sum(0);present=np.where(n>0)[0];f={str(int(c)):float(2*m[c,c]/(n[c]+p[c]))for c in present};fg=[f[str(int(c))]for c in present if c!=0]
 return {'accuracy':float(np.trace(m)/len(y)),'macro_f1_present':float(np.mean(list(f.values()))),'foreground_macro_f1_present':float(np.mean(fg))if fg else None,'support':{str(int(c)):int(n[c])for c in present},'per_class_f1':f},m

def close_dict(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b),(set(a),set(b))
  for k in a:close_dict(a[k],b[k])
 elif isinstance(a,(float,np.floating)):assert abs(a-b)<1e-12,(a,b)
 else:assert a==b,(a,b)

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());assert sha(ROOT/'protocol.json')==s['protocol_sha256'];assert p['steps']==500 and p['seeds']==[0,1,2]and p['arms']==list(ARMS)and p['coefficients']=={k:list(v)for k,v in COEF.items()}
 for path,h in p['input_sha256'].items():assert sha(path)==h,path
 for path,h in s['output_sha256'].items():assert sha(path)==h,path
 selection=json.loads((ROOT/'frozen_selection.json').read_text());assert sha(ROOT/'frozen_selection.json')==s['frozen_selection_sha256']and selection['protocol_sha256']==s['protocol_sha256']
 assert len(selection['adapters_sha256'])==12
 for path,h in selection['adapters_sha256'].items():assert sha(path)==h
 manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text());ann=json.loads(Path('semantic_artifacts/ovis/annotations_train.json').read_text());by=defaultdict(list)
 for a in ann['annotations']:by[int(a['video_id'])].append(a)
 train={};pools={};source_GT_points=0
 for clip in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
  vid=int(clip['video_id'])
  with np.load(CACHE/'train'/f'{vid}.npz')as a:r={k:a[k].copy()for k in ('prelast','memory','labels','xy')}
  with np.load(Path('semantic_artifacts/teacher_ovis')/f'{vid}.npz')as a:pseudo=a['labels'].copy()
  xy=r['xy'];y=r['labels'];assert np.array_equal(y,pseudo[xy[:,2],xy[:,1],xy[:,0]])
  positions=np.where(xy[:,2]==0)[0];counts=np.zeros(len(positions),int);cats=np.zeros(len(positions),int)
  for a in by[vid]:
   raw=a['segmentations'][clip['start']]
   if raw is None:continue
   native=mask_api.decode(raw)
   if native.ndim==3:native=native.any(2)
   reduced=np.asarray(Image.fromarray(native.astype('uint8')).resize((128,128),Image.Resampling.NEAREST));hit=reduced[xy[positions,1],xy[positions,0]]>0;counts+=hit;cats[hit]=a['category_id']
  expected=np.where(counts==0,0,np.where(counts==1,cats,-1));assert np.array_equal(expected,y[positions]);source_GT_points+=len(positions)
  pools[vid]={'fg':np.where((xy[:,2]==0)&(y>0))[0],'bg':np.where((xy[:,2]==0)&(y==0))[0],'sam':np.where((xy[:,2]>0)&(y>0))[0]}
  r['prelast']=torch.from_numpy(r['prelast']).cuda().to(torch.bfloat16);r['memory']=torch.from_numpy(r['memory']).cuda().to(torch.bfloat16)[None];r['y']=torch.from_numpy(y.astype(np.int64)).cuda();train[vid]=r
 assert {str(v):{k:len(x)for k,x in pool.items()}for v,pool in pools.items()}==p['training_label_pools']
 vids=list(train);eligible=[v for v in vids if len(pools[v]['sam'])]
 for seed in range(3):
  rng=np.random.Generator(np.random.PCG64(20261099+seed));assert len(p['schedules'][str(seed)])==500
  for row in p['schedules'][str(seed)]:
   gv=vids[int(rng.integers(len(vids)))];av=eligible[int(rng.integers(len(eligible)))];pool=pools[gv]
   def choose(a):return rng.choice(a,32,replace=len(a)<32).tolist()
   expected={'GT_video':gv,'aux_video':av,'GT_indices':choose(pool['fg'])+choose(pool['bg']),'extra_GT_indices':choose(pool['fg']),'SAM_indices':choose(pools[av]['sam'])};assert row==expected
 print('Verified all60 GT/teacher labels and all1500 schedules',flush=True)
 published=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml');training_checks={};validated={a:{}for a in ARMS};confusions={}
 def model_for(template,arm,state):
  model=copy.deepcopy(template).cuda().eval()
  with torch.no_grad():
   for n,q in model.named_parameters():
    if n in state:q.copy_(state[n].to(q.device))
    if arm=='A'and n.endswith(('.a','.b')):q.requires_grad_(False)
  return model
 def batch_outputs(model,row):
  gt=train[row['GT_video']];aux=train[row['aux_video']];gi=torch.tensor(row['GT_indices']+row['extra_GT_indices'],device='cuda');ai=torch.tensor(row['SAM_indices'],device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):go=model(gt['prelast'][gi][None],gt['memory']);ao=model(aux['prelast'][ai][None],aux['memory'])
  return go,ao,gt['y'][gi],aux['y'][ai]
 def manual_loss(model,row,rg,ra,coef):
  go,ao,gy,ay=batch_outputs(model,row);lp=go['semantic_logits'][0].float().log_softmax(-1);al=ao['semantic_logits'][0].float().log_softmax(-1);ce=-lp[torch.arange(64,device='cuda'),gy[:64]].mean();extra=-lp[torch.arange(64,96,device='cuda'),gy[64:]].mean();aux=-al[torch.arange(32,device='cuda'),ay].mean();d=(torch.cat([flat(go),flat(ao)],1)-torch.cat([rg,ra],1)).abs();geo=torch.where(d<1,d.square()/2,d-.5).mean();return ce+coef[0]*geo+coef[1]*aux+coef[2]*extra
 for seed in range(3):
  torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(ROOT/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True);ih={n:tensor_hash(v)for n,v in initial.items()};reference=model_for(template,'A',initial).requires_grad_(False)
  with np.load(ROOT/f'references_seed{seed}.npz')as a:rg=torch.from_numpy(a['GT']).cuda();ra=torch.from_numpy(a['aux']).cuda()
  with torch.no_grad():
   for j,row in enumerate(p['schedules'][str(seed)]):
    go,ao,_,_=batch_outputs(reference,row);assert torch.equal(flat(go)[0],rg[j])and torch.equal(flat(ao)[0],ra[j])
  first_grad={}
  for arm in ARMS:
   logged=s['results'][arm][str(seed)];assert logged['initial_adapter_sha256']==ih and logged['steps']==500 and logged['GT_supervision_exposures']==32000 and logged['SAM_supervision_exposures']==(16000 if arm=='C'else 0)and logged['extra_GT_supervision_exposures']==(16000 if arm=='D'else 0)
   final=torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);assert {n:tensor_hash(v)for n,v in final.items()}==logged['final_adapter_sha256'];assert any(not torch.equal(final[n],initial[n])for n in final if n.startswith('semantic_head.'))
   if arm=='A':assert all(torch.equal(final[n],initial[n])for n in final if n.endswith(('.a','.b')))
   else:assert any(not torch.equal(final[n],initial[n])for n in final if n.endswith('.b'))
   model=model_for(template,arm,initial);assert sum(q.numel()for q in model.parameters()if q.requires_grad)==logged['trainable_parameters']==(33306 if arm=='A'else 217626);frozen={n:tensor_hash(q)for n,q in model.named_parameters()if not managed(n)}
   loss=manual_loss(model,p['schedules'][str(seed)][0],rg[0][None],ra[0][None],COEF[arm]);loss.backward();assert abs(float(loss.detach())-logged['trace'][0]['total'])<2e-6
   stored=torch.load(ROOT/f'{arm}_seed{seed}_step0.pt',map_location='cpu',weights_only=True);grads={n:q.grad.detach().cpu()for n,q in model.named_parameters()if q.grad is not None};assert set(grads)==set(stored['gradients']);maximum=0.
   for n,g in grads.items():
    assert managed(n)and bool(torch.isfinite(g).all());torch.testing.assert_close(g,stored['gradients'][n],rtol=5e-4,atol=5e-6);maximum=max(maximum,float((g-stored['gradients'][n]).abs().max()))
   first_grad[arm]=grads;del model;gc.collect();torch.cuda.empty_cache()
   before=torch.load(ROOT/f'{arm}_seed{seed}_before_final.pt',map_location='cpu',weights_only=True);model=model_for(template,arm,before['adapter']);opt=torch.optim.AdamW([q for q in model.parameters()if q.requires_grad],lr=1e-4,weight_decay=.01);opt.load_state_dict(before['optimizer']);loss=manual_loss(model,p['schedules'][str(seed)][-1],rg[-1][None],ra[-1][None],COEF[arm]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();end_error=0.
   for n,q in model.named_parameters():
    if n in final:torch.testing.assert_close(q.detach().cpu(),final[n],rtol=3e-4,atol=2e-6);end_error=max(end_error,float((q.detach().cpu()-final[n]).abs().max()))
   assert frozen=={n:tensor_hash(q)for n,q in model.named_parameters()if not managed(n)}
   for j,r in enumerate(logged['trace']):assert r['step']==j+1 and abs(r['total']-(r['GT_CE']+COEF[arm][0]*r['geometry_smooth_l1']+COEF[arm][1]*r['SAM_CE']+COEF[arm][2]*r['extra_GT_CE']))<3e-6
   training_checks[f'{arm}_{seed}']={'first_gradient_max_abs_error':maximum,'final_optimizer_step_parameter_max_abs_error':end_error,'shared_initialization_and_only_whitelist_updated':True};del model,opt;gc.collect();torch.cuda.empty_cache()
  assert any(not torch.equal(first_grad['B'][n],first_grad['C'][n])for n in first_grad['B']);assert any(not torch.equal(first_grad['B'][n],first_grad['D'][n])for n in first_grad['B'])
  print('Verified full reference schedule and four gradient/optimizer replays seed',seed,flush=True);del template,reference,rg,ra;gc.collect();torch.cuda.empty_cache()
 del train;gc.collect();torch.cuda.empty_cache()
 val=[]
 for clip in sorted(manifest['val'],key=lambda c:int(c['video_id'])):
  vid=int(clip['video_id'])
  with np.load(CACHE/'val'/f'{vid}.npz')as a:r={k:a[k].copy()for k in ('prelast','alternate_prelast','memory','labels')}
  for k in ('prelast','alternate_prelast','memory'):r[k]=torch.from_numpy(r[k]).cuda().to(torch.bfloat16)
  val.append((vid,r))
 for seed in range(3):
  torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(ROOT/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True);reference=model_for(template,'A',initial).requires_grad_(False)
  for arm in ARMS:
   final=torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);model=model_for(template,arm,final);saved=np.load(ROOT/f'{arm}_seed{seed}_validation.npz');offset=0;per={};drift={k:{'sum':0.,'count':0,'max':0.}for k in GEOMETRY_KEYS}
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for vid,r in val:
     logits=[];alt=[]
     for start in range(0,len(r['labels']),64):
      out=model(r['prelast'][start:start+64][None],r['memory'][None]);base=reference(r['prelast'][start:start+64][None],r['memory'][None]);other=model(r['alternate_prelast'][start:start+64][None],r['memory'][None]);logits.append(out['semantic_logits'][0].float().cpu().numpy());alt.append(other['semantic_logits'][0].float().cpu().numpy())
      for k in GEOMETRY_KEYS:
       d=(out[k].float()-base[k].float()).abs().cpu().numpy();drift[k]['sum']+=d.astype(np.float64).sum();drift[k]['count']+=d.size;drift[k]['max']=max(drift[k]['max'],float(d.max()))
     z=np.concatenate(logits);az=np.concatenate(alt);n=len(z);assert np.array_equal(z,saved['logits'][offset:offset+n])and np.array_equal(az,saved['alternate_logits'][offset:offset+n]);assert np.array_equal(r['labels'],saved['labels'][offset:offset+n])and np.all(saved['video_ids'][offset:offset+n]==vid);per[str(vid)]=metrics(r['labels'],z.argmax(1))[0];offset+=n
   assert offset==len(saved['labels'])==40960
   pooled,confusion=metrics(saved['labels'],saved['logits'].argmax(1));logged=s['results'][arm][str(seed)]['evaluation'];close_dict(pooled,logged['pooled']);close_dict(per,logged['per_video']);agreement=float(np.mean(saved['logits'].argmax(1)==saved['alternate_logits'].argmax(1)));assert agreement==logged['target_camera_variant_agreement']
   for k,d in drift.items():
    l=logged['geometry_output_drift'][k];assert abs(d['sum']-l['absolute_sum'])<1e-8 and d['count']==l['count']and d['max']==l['max_abs']and abs(d['sum']/d['count']-l['mean_abs'])<1e-12
   validated[arm][str(seed)]={'pooled':pooled,'per_video':per};confusions[f'{arm}_{seed}']=confusion.tolist();print('Verified full validation',seed,arm,flush=True);saved.close();del model;gc.collect();torch.cuda.empty_cache()
  del reference,template;gc.collect();torch.cuda.empty_cache()
 for left,right in (('C','B'),('B','A'),('C','D')):
  expected=[]
  for seed in range(3):
   l=validated[left][str(seed)];r=validated[right][str(seed)];diffs={v:l['per_video'][v]['macro_f1_present']-r['per_video'][v]['macro_f1_present']for v in l['per_video']};expected.append({'seed':seed,'pooled_macro_difference':l['pooled']['macro_f1_present']-r['pooled']['macro_f1_present'],'per_video_differences':diffs,'video_equal_mean_difference':float(np.mean(list(diffs.values())))})
  c=s['contrasts'][left+'_minus_'+right]
  assert c['per_seed']==expected and abs(c['mean_pooled_macro_difference']-np.mean([r['pooled_macro_difference']for r in expected]))<1e-12 and abs(c['mean_video_equal_difference']-np.mean([r['video_equal_mean_difference']for r in expected]))<1e-12
 qualified=all(all(r['pooled_macro_difference']>0 for r in s['contrasts'][k]['per_seed'])and s['contrasts'][k]['mean_video_equal_difference']>0 for k in ('C_minus_B','C_minus_D'));assert qualified==s['SAM_recipe_eligible_for_future_separate_external_protocol']
 for path,h in p['input_sha256'].items():assert sha(path)==h,path
 atomic_json(ROOT/'verification.json',{'round':99,'summary_sha256':sha(ROOT/'summary.json'),'protocol_sha256':sha(ROOT/'protocol.json'),'verifier_sha256':sha('semantic/verify_d4rt_controlled_finetune.py'),'all60_training_first_frame_GT_points_independently_rebuilt':source_GT_points,'all60_teacher_cache_labels_verified':True,'all1500_schedules_and_supervision_budgets_recomputed':True,'all3000_training_reference_batches_recomputed':True,'all12_initial_gradients_and_final_optimizer_steps_replayed':training_checks,'all12_full40960_query_original_and_alternate_validation_outputs_exact':True,'all12_confusions':confusions,'all72_geometry_output_aggregates_independently_rebuilt':True,'all_three_contrasts_and_external_eligibility_verified':True,'all_source_fingerprints_unchanged':True,'no_new_confirmation_or_geometry_truth_accuracy_claim':True})
 print('DONE independent controlled fine-tuning verification',flush=True)
if __name__=='__main__':main()
