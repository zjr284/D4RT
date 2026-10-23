"""Independent raw-RLE association and dual auxiliary-target training replay."""
from pathlib import Path
from collections import defaultdict
import copy,gc,json
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_api
from semantic.model import SemanticTail,GEOMETRY_KEYS
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha,metrics,close_dict
from tools.load_opend4rt import load_model,DEFAULT_VARIANT
ROOT=Path('semantic_artifacts/d4rt_auxiliary_label_oracle');SOURCE=Path('semantic_artifacts/d4rt_matched_head_finetune');CACHE=Path('semantic_artifacts/features_ovis/sam');ARMS=('sam_common','oracle_common')
def flatten(out):return torch.cat([out[k].reshape(1,-1,int(np.prod(out[k].shape[2:])))for k in GEOMETRY_KEYS],-1).float()

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());assert sha(ROOT/'protocol.json')==s['protocol_sha256']and p['round']==103 and p['arms']==list(ARMS)and p['steps']==500
 for key in ('input_sha256','GT_binding_sha256'):
  for path,h in p[key].items():assert sha(path)==h,path
 for path,h in s['output_sha256'].items():assert sha(path)==h,path
 frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert len(frozen['adapters_sha256'])==6 and frozen['protocol_sha256']==s['protocol_sha256']
 for path,h in frozen['adapters_sha256'].items():assert sha(path)==h
 manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text());annotation=json.loads(Path('semantic_artifacts/ovis/annotations_train.json').read_text());by=defaultdict(list)
 for a in annotation['annotations']:by[int(a['video_id'])].append(a)
 train={};truth={};conf=np.zeros((26,26),np.int64);total=ignored=0;schedules=json.loads((SOURCE/'protocol.json').read_text())['schedules']
 for c in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
  vid=int(c['video_id'])
  with np.load(CACHE/'train'/f'{vid}.npz')as a:r={k:a[k].copy()for k in ('prelast','memory','labels','xy')}
  xy=r['xy'];gt=np.zeros(len(xy),np.int64)
  for t in range(8):
   positions=np.flatnonzero(xy[:,2]==t);counts=np.zeros(len(positions),np.int64);classes=np.zeros(len(positions),np.int64)
   for a in by[vid]:
    raw=a['segmentations'][c['start']+t]
    if raw is None:continue
    m=mask_api.decode(raw)
    if m.ndim==3:m=m.any(2)
    m=np.asarray(Image.fromarray(m.astype('uint8')).resize((128,128),Image.Resampling.NEAREST));hit=m[xy[positions,1],xy[positions,0]]>0;counts+=hit;classes[hit]=a['category_id']
   gt[positions]=np.where(counts==0,0,np.where(counts==1,classes,-1))
  with np.load(ROOT/'associations'/f'{vid}.npz')as a:assert np.array_equal(gt,a['GT_labels'])and np.array_equal(xy,a['xy'])and np.array_equal(r['labels'],a['SAM_labels'])
  with np.load(Path('semantic_artifacts/teacher_ovis')/f'{vid}.npz')as a:assert np.array_equal(r['labels'],a['labels'][xy[:,2],xy[:,1],xy[:,0]])
  later=(xy[:,2]>0)&(r['labels']>0);valid=later&(gt>=0);np.add.at(conf,(gt[valid],r['labels'][valid]),1);total+=int(later.sum());ignored+=int((later&(gt<0)).sum())
  r['q']=torch.from_numpy(r['prelast']).cuda().to(torch.bfloat16);r['m']=torch.from_numpy(r['memory']).cuda().to(torch.bfloat16)[None];r['y']=torch.from_numpy(r['labels'].astype(np.int64)).cuda();train[vid]=r;truth[vid]=torch.from_numpy(gt).cuda()
 assert conf.tolist()==p['query_conditional_teacher_confusion_GT_rows_SAM_columns']and total==p['query_conditional_later_SAM_FG_count']and ignored==p['query_conditional_later_ignored_GT_count']
 for seed,rows in schedules.items():assert [int((truth[row['aux_video']][torch.tensor(row['SAM_indices'],device='cuda')]>=0).sum())for row in rows]==p['schedule_common_valid_aux_counts'][seed]
 print('Verified all480 source GT frame associations and identical common masks',flush=True)
 published=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml');replays={}
 def create(template,state):
  model=copy.deepcopy(template).cuda().eval()
  with torch.no_grad():
   for n,q in model.named_parameters():
    if n in state:q.copy_(state[n].to(q.device))
  return model
 def optimizer(model):return torch.optim.AdamW([{'params':[q for n,q in model.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},{'params':[q for n,q in model.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01)
 def loss(model,row,rg,ra,arm):
  g=train[row['GT_video']];a=train[row['aux_video']];gi=torch.tensor(row['GT_indices']+row['extra_GT_indices'],device='cuda');ai=torch.tensor(row['SAM_indices'],device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):go=model(g['q'][gi][None],g['m']);ao=model(a['q'][ai][None],a['m'])
  l=go['semantic_logits'][0,:64].float().log_softmax(-1);ce=-l[torch.arange(64,device='cuda'),g['y'][gi[:64]]].mean();gt=truth[row['aux_video']][ai];valid=torch.where(gt>=0)[0];target=a['y'][ai]if arm=='sam_common'else gt
  if len(valid):
   z=ao['semantic_logits'][0,valid].float().log_softmax(-1);aux=-z[torch.arange(len(valid),device='cuda'),target[valid]].mean()
  else:aux=ao['semantic_logits'].float().sum()*0
  d=(torch.cat([flatten(go),flatten(ao)],1)-torch.cat([rg,ra],1)).abs();geo=torch.where(d<1,d.square()/2,d-.5).mean();return ce+.1*geo+.25*aux
 for seed in range(3):
  torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(SOURCE/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True)
  with np.load(SOURCE/f'references_seed{seed}.npz')as a:rg=torch.from_numpy(a['GT'].copy()).cuda();ra=torch.from_numpy(a['aux'].copy()).cuda()
  for arm in ARMS:
   logged=s['results'][arm][str(seed)];assert logged['GT_exposures']==32000 and logged['common_aux_exposures']==sum(p['schedule_common_valid_aux_counts'][str(seed)])and logged['empty_aux_batches']==p['schedule_common_valid_aux_counts'][str(seed)].count(0)
   model=create(template,initial);value=loss(model,schedules[str(seed)][0],rg[0][None],ra[0][None],arm);value.backward();assert abs(float(value.detach())-logged['trace'][0]['total'])<2e-6;stored=torch.load(ROOT/f'{arm}_seed{seed}_step0.pt',map_location='cpu',weights_only=True);grads={n:q.grad.detach().cpu()for n,q in model.named_parameters()if q.grad is not None};assert set(grads)==set(stored['gradients']);grad_error=0.
   for n,g in grads.items():assert n.startswith('semantic_head.')or n.endswith(('.a','.b'));torch.testing.assert_close(g,stored['gradients'][n],rtol=5e-4,atol=5e-6);grad_error=max(grad_error,float((g-stored['gradients'][n]).abs().max()))
   del model;gc.collect();torch.cuda.empty_cache();saved=torch.load(ROOT/f'{arm}_seed{seed}_before_final.pt',map_location='cpu',weights_only=True);model=create(template,saved['adapter']);opt=optimizer(model);opt.load_state_dict(saved['optimizer']);value=loss(model,schedules[str(seed)][-1],rg[-1][None],ra[-1][None],arm);opt.zero_grad(set_to_none=True);value.backward();opt.step();final=torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);error=0.
   for n,q in model.named_parameters():
    if n in final:torch.testing.assert_close(q.detach().cpu(),final[n],rtol=3e-4,atol=2e-6);error=max(error,float((q.detach().cpu()-final[n]).abs().max()))
   for j,row in enumerate(logged['trace']):assert row['step']==j+1 and row['valid_aux_points']==p['schedule_common_valid_aux_counts'][str(seed)][j]and abs(row['total']-(row['GT_CE']+.25*row['auxiliary_CE']+.1*row['geometry_smooth_l1']))<3e-6
   replays[f'{arm}_{seed}']={'gradient_max_abs_error':grad_error,'final_optimizer_step_parameter_max_abs_error':error};del model,opt;gc.collect();torch.cuda.empty_cache()
  print('Verified dual label gradients and optimizer seed',seed,flush=True);del template,rg,ra;gc.collect();torch.cuda.empty_cache()
 del train,truth;gc.collect();torch.cuda.empty_cache();val=[]
 for c in sorted(manifest['val'],key=lambda c:int(c['video_id'])):
  vid=int(c['video_id'])
  with np.load(CACHE/'val'/f'{vid}.npz')as a:r={k:a[k].copy()for k in ('prelast','alternate_prelast','memory','labels')}
  for k in ('prelast','alternate_prelast','memory'):r[k]=torch.from_numpy(r[k]).cuda().to(torch.bfloat16)
  val.append((vid,r))
 rebuilt={arm:{}for arm in ARMS};confusions={}
 for seed in range(3):
  torch.manual_seed(seed);template=SemanticTail(published,26).eval();initial=torch.load(SOURCE/f'initial_seed{seed}.pt',map_location='cpu',weights_only=True);reference=create(template,initial).requires_grad_(False)
  for arm in ARMS:
   final=torch.load(ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);model=create(template,final);saved=np.load(ROOT/f'{arm}_seed{seed}_validation.npz');offset=0;per={};drift={k:[0.,0,0.]for k in GEOMETRY_KEYS}
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for vid,r in val:
     z=[];alt=[]
     for start in range(0,len(r['labels']),64):
      out=model(r['prelast'][start:start+64][None],r['memory'][None]);base=reference(r['prelast'][start:start+64][None],r['memory'][None]);other=model(r['alternate_prelast'][start:start+64][None],r['memory'][None]);z.append(out['semantic_logits'][0].float().cpu().numpy());alt.append(other['semantic_logits'][0].float().cpu().numpy())
      for k in GEOMETRY_KEYS:
       d=(out[k].float()-base[k].float()).abs().cpu().numpy();drift[k][0]+=d.astype(np.float64).sum();drift[k][1]+=d.size;drift[k][2]=max(drift[k][2],float(d.max()))
     z=np.concatenate(z);alt=np.concatenate(alt);n=len(z);assert np.array_equal(z,saved['logits'][offset:offset+n])and np.array_equal(alt,saved['alternate_logits'][offset:offset+n])and np.array_equal(r['labels'],saved['labels'][offset:offset+n])and np.all(saved['video_ids'][offset:offset+n]==vid);per[str(vid)]=metrics(r['labels'],z.argmax(1))[0];offset+=n
   assert offset==40960;logged=s['results'][arm][str(seed)]['evaluation'];pooled,cm=metrics(saved['labels'],saved['logits'].argmax(1));close_dict(pooled,logged['pooled']);close_dict(per,logged['per_video']);assert float(np.mean(saved['logits'].argmax(1)==saved['alternate_logits'].argmax(1)))==logged['target_camera_variant_agreement']
   for k,(total,count,max_abs)in drift.items():l=logged['geometry_output_drift'][k];assert abs(total-l['absolute_sum'])<1e-8 and count==l['count']and max_abs==l['max_abs']and abs(total/count-l['mean_abs'])<1e-12
   rebuilt[arm][str(seed)]={'pooled':pooled,'per_video':per};confusions[f'{arm}_{seed}']=cm.tolist();print('Verified label-oracle validation',seed,arm,flush=True);saved.close();del model;gc.collect();torch.cuda.empty_cache()
  del reference,template;gc.collect();torch.cuda.empty_cache()
 differences=[]
 for seed in range(3):
  o=rebuilt['oracle_common'][str(seed)];a=rebuilt['sam_common'][str(seed)];d={v:o['per_video'][v]['macro_f1_present']-a['per_video'][v]['macro_f1_present']for v in a['per_video']};differences.append({'seed':seed,'pooled_macro_difference':o['pooled']['macro_f1_present']-a['pooled']['macro_f1_present'],'per_video_differences':d,'video_equal_mean_difference':float(np.mean(list(d.values())))})
 assert differences==s['oracle_minus_SAM_common'];assert bool(all(d['pooled_macro_difference']>0 for d in differences)and np.mean([d['video_equal_mean_difference']for d in differences])>0)==s['label_noise_contributes_under_fixed_queries_rule_satisfied'];assert s['oracle_eligible_for_external_or_deployment']is False
 for key in ('input_sha256','GT_binding_sha256'):
  for path,h in p[key].items():assert sha(path)==h
 atomic_json(ROOT/'verification.json',{'round':103,'summary_sha256':sha(ROOT/'summary.json'),'verifier_sha256':sha('semantic/verify_d4rt_auxiliary_label_oracle.py'),'all60_eight_frame_RLE_query_associations_independently_rebuilt':True,'all_teacher_labels_common_masks_and1500_budgets_verified':True,'all6_gradient_and_final_optimizer_replays':replays,'all6_full_original_and_alternate_validation_logits_exact':True,'all_confusions':confusions,'all36_geometry_output_aggregates_and_contrast_verified':True,'all_source_fingerprints_unchanged':True,'oracle_not_a_SAM_method_or_external_candidate':True});print('DONE independent label-oracle verification',flush=True)
if __name__=='__main__':main()
