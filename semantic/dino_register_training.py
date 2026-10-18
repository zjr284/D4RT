"""Frozen official register backbone, same source queries and head exposure budget."""
from collections import defaultdict
import fcntl,gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_register_resource as resource
from semantic import feature_standardization_control as prior
from semantic import coco_fusion_cohort as coco_resource
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.run import atomic_json
from semantic.oracle_probe import file_hash
ROOT=Path('semantic_artifacts/dino_register_training');OLD=('b14_gt32','z_b14_gt32');NEW=('reg_raw','reg_z');ARMS=OLD+NEW;SPLITS=prior.SPLITS

def fp(split,vid):return ROOT/'features'/split/f'{vid}.npz'
def ckpt(arm,seed):return prior.ckpt(arm,seed)if arm in OLD else ROOT/arm/'runs'/f'linear_seed{seed}.pt'

def future_candidates():
 previous=json.loads((coco_resource.prior.old.ROOT/'manifest.json').read_text());recent=json.loads((coco_resource.prior.ROOT/'manifest.json').read_text());excluded={im['id']for role in ('candidate','reserve')for im in previous[role]}|{im['id']for role in ('candidate','reserve')for im in recent[role]};assert len(excluded)==240
 payload=json.loads((coco_resource.META/'instances_val2017.json').read_bytes());mapping={int(k):int(v)for k,v in json.loads((coco_resource.META/'summary.json').read_text())['shared_name_mapping'].items()};images=sorted(payload['images'],key=lambda a:a['id']);rank=[images[int(i)]['id']for i in np.random.default_rng(20261090).permutation(len(images))];byid={im['id']:im for im in images};classes=sorted(set(mapping.values())-{1});pools={c:set()for c in classes}
 for ann in payload['annotations']:
  c=mapping.get(ann['category_id'])
  if c in pools and not ann['iscrowd']and ann['area']>0:pools[c].add(ann['image_id'])
 union=set().union(*pools.values());seen=set(excluded);result={}
 for role in ('candidate','reserve'):
  selected=[]
  for _ in range(2):
   for c in classes:
    value=next((i for i in rank if i in pools[c]and i not in seen),None)
    if value is not None and len(selected)<80:selected.append(value);seen.add(value)
  for i in rank:
   if len(selected)<80 and i in union and i not in seen:selected.append(i);seen.add(i)
  assert len(selected)==80;result[role]=[byid[i]for i in selected]
 return {'seed':20261090,'excluded_previously_evaluated_COCO_ids':sorted(excluded),'metadata_previously_seen':True,'no_new_RGB_GT_mask_or_model_evaluation':True,**result}

def freeze():
 source=json.loads((prior.ROOT/'summary.json').read_text());assert file_hash(prior.ROOT/'summary.json')==json.loads((prior.ROOT/'verification.json').read_text())['summary_sha256'];rs=json.loads((resource.ROOT/'summary.json').read_text());assert file_hash(resource.ROOT/'summary.json')==json.loads((resource.ROOT/'verification.json').read_text())['summary_sha256']
 assert file_hash(resource.WEIGHTS)==rs['acquisition']['weight_sha256'];m=json.loads((prior.ROOT/'manifest.json').read_text());future=future_candidates();atomic_json(ROOT/'future_confirmation_manifest.json',future)
 paths=[Path(x)for x in {**source['protocol']['input_sha256'],**rs['protocol']['input_sha256']}]+[resource.ROOT/n for n in ('summary.json','verification.json','acquisition.json')]+[resource.WEIGHTS,Path('semantic/dino_register_resource.py'),Path('semantic/dino_register_training.py'),Path('semantic/OVIS_DINO_REGISTER_TRAINING_PROTOCOL.md'),prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'statistics_manifest.json',Path(source['statistics']['b14_gt32']['path']),ROOT/'future_confirmation_manifest.json',coco_resource.META/'instances_val2017.json',coco_resource.META/'summary.json',coco_resource.prior.ROOT/'manifest.json',coco_resource.prior.old.ROOT/'manifest.json']
 for split in ('train','val'):
  for clip in m[split]:paths.extend(Path(x)for x in (clip['frames'][:1]if split=='train'else clip['frames']))
 for arm in OLD:
  for seed in range(3):paths.extend([ckpt(arm,seed),ckpt(arm,seed).with_suffix('.json')]);paths.extend(prior.ROOT/arm/'logits'/f'seed{seed}_{split}.npz'for split in SPLITS)
 p={'round':90,'new_heads':6,'head_parameters':19994,'feature_dimension':768,'steps':2000,'batch':32,'exposures':64000,'fixed_plain_offsets':{arm:source['results'][arm]['offsets']for arm in OLD},'plain_statistics':source['statistics']['b14_gt32'],'calibration_grid':source['protocol']['calibration_grid'],'source_validation_only':True,'future_confirmation_manifest_sha256':file_hash(ROOT/'future_confirmation_manifest.json'),'input_sha256':{str(path):file_hash(path)for path in sorted(set(paths))}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p

def metadata(split,vid):
 path=prior.raw_path('b14_gt32',split,vid)
 with np.load(path)as a:out={k:a[k].copy()for k in a.files if k!='decoded'}
 if split=='spatial_val':
  with np.load(prior.prior.spatial.qp(vid))as a:out['labels']=a['labels'].copy()
 return out

def cache(m):
 model=None
 for split in ('train','val'):
  for clip in m[split]:
   keys=('train',)if split=='train'else SPLITS
   if all(fp(key,clip['video_id']).exists()for key in keys):continue
   if model is None:model=resource.load_register()
   metas={key:metadata(key,clip['video_id'])for key in keys};xy=np.concatenate([metas[key]['xy']for key in keys]);video=resource.prior.rgb.load_native_video({'frames':clip['frames'][:1]if split=='train'else clip['frames']}).cuda();tokens=resource.prior.tokens_b14(model,video)
   with torch.inference_mode():features=patch_at_queries(tokens,xy).half().cpu().numpy()
   start=0
   for key in keys:
    n=len(metas[key]['labels']);save_npz(fp(key,clip['video_id']),**metas[key],decoded=features[start:start+n]);start+=n
   print('official register feature',split,clip['video_id'],flush=True)
 if model is not None:del model;gc.collect();torch.cuda.empty_cache()

def register_statistics(m):
 values=[]
 for clip in sorted(m['train'],key=lambda a:str(a['video_id'])):
  with np.load(fp('train',clip['video_id']))as a:values.append(a['decoded'].copy())
 x=np.concatenate(values);assert len(x)==20480;mu,scale,std=prior.fit_statistics(x);path=ROOT/'register_statistics.npz'
 if path.exists():
  with np.load(path)as a:assert np.array_equal(mu,a['mean'])and np.array_equal(scale,a['scale'])and np.array_equal(std,a['std'])
 else:save_npz(path,mean=mu,scale=scale,std=std)
 r={'path':str(path),'sha256':file_hash(path),'samples':len(x),'dimension':768,'fallback_scale_channels':int((std<=1e-6).sum()),'std_min':float(std.min()),'std_median':float(np.median(std)),'std_max':float(std.max())};atomic_json(ROOT/'statistics_manifest.json',r);return r

def read(arm,split,m,stats,p):
 if arm in OLD:return prior.read(arm,split,m,{'b14_gt32':p['plain_statistics']})
 with np.load(stats['path'])as a:mu=a['mean'].copy();scale=a['scale'].copy()
 rows=[]
 for clip in sorted(m['train']if split=='train'else m['val'],key=lambda a:str(a['video_id'])):
  with np.load(fp(split,clip['video_id']))as a:row={k:a[k].copy()for k in a.files}
  if arm=='reg_z':row['decoded']=(row['decoded'].astype(np.float32)-mu)/scale
  rows.append(row)
 return rows

def train(arm,seed,base,val,p,stats):
 path=ckpt(arm,seed);report=path.with_suffix('.json')
 if report.exists():
  r=json.loads(report.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();x=torch.from_numpy(np.concatenate([a['decoded']for a in base])).float().cuda();y=np.concatenate([a['labels']for a in base]);times=np.concatenate([a['xy'][:,2]for a in base]);assert len(y)==20480 and(times==0).all()
 sampler=Sampler(y,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);ih=hashlib.sha256();hist=torch.zeros(26,dtype=torch.long,device='cuda');history=[];best=-1
 for step in range(1,2001):
  fg,bg=sampler.draw(gen,count=16);ix=torch.cat([fg,bg]);ih.update(ix.cpu().numpy().tobytes());hist+=torch.bincount(sampler.labels[ix],minlength=26);model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ix]),sampler.labels[ix]);assert torch.isfinite(loss);loss.backward();optimizer.step()
  if step%50==0:
   value=score(model,val);history.append({'step':step,'balanced_raw_val_macro':value,'loss':float(loss.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 reference=json.loads(ckpt('b14_gt32',seed).with_suffix('.json').read_text());assert ih.hexdigest()==reference['sampled_indices_sha256']and hist.cpu().tolist()==reference['sampled_class_counts']
 path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p,'register_statistics_sha256':stats['sha256']},partial);partial.replace(path)
 r={'arm':arm,'seed':seed,'protocol':p,'register_statistics_sha256':stats['sha256'],'head_parameters':19994,'best_step':best_step,'best_balanced_raw_val':best,'history':history,'sampled_indices_sha256':ih.hexdigest(),'sampled_class_counts':hist.cpu().tolist(),'checkpoint_sha256':file_hash(path)};atomic_json(report,r);print('trained official register head',arm,seed,best_step,best,flush=True);del model,x;gc.collect();torch.cuda.empty_cache();return r

def infer(arm,seed,split,data):
 path=ROOT/arm/'logits'/f'seed{seed}_{split}.npz';h=file_hash(ckpt(arm,seed))
 if path.exists():
  with np.load(path)as a:out={k:a[k].copy()for k in a.files}
  assert str(out['checkpoint_sha256'])==h;return out
 if arm in OLD:
  with np.load(prior.ROOT/arm/'logits'/f'seed{seed}_{split}.npz')as a:out={k:a[k].copy()for k in a.files}
 else:
  state=torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model']
  with torch.inference_mode():values=[F.linear(torch.from_numpy(a['decoded']).float().cuda(),state['classifier.weight'].cuda(),state['classifier.bias'].cuda()).cpu().numpy()for a in data]
  out={'logits':np.concatenate(values),'labels':np.concatenate([a['labels']for a in data]),'times':np.concatenate([a['xy'][:,2]for a in data]),'video_ids':np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in data]),'checkpoint_sha256':np.array(h)}
 save_npz(path,**out);return out

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed register2x2 exists')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  m,p=freeze();cache(m);stats=register_statistics(m);reports={};results={};offsets={};curves={}
  for arm in ARMS:
   data={split:read(arm,split,m,stats,p)for split in ('train',)+SPLITS}
   if arm in NEW:reports[arm]=[train(arm,seed,data['train'],data['balanced_val'],p,stats)for seed in range(3)]
   outputs={split:[infer(arm,seed,split,data[split])for seed in range(3)]for split in SPLITS};offsets[arm]={'raw':0.};curves[arm]={}
   for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
    if arm in OLD:offsets[arm][profile]=p['fixed_plain_offsets'][arm][profile]
    else:offsets[arm][profile],curves[arm][profile]=pick_offset(outputs[split],p['calibration_grid'])
   runs=[{'seed':seed,**{split:{profile:evaluate(outputs[split][seed],offset)for profile,offset in offsets[arm].items()}for split in SPLITS}}for seed in range(3)];results[arm]={'offsets':offsets[arm],'runs':runs,'mean_balanced_profile_macro':float(np.mean([a['balanced_val']['balanced_profile']['macro_f1_present']for a in runs]))};print('register2x2 source score',arm,results[arm]['mean_balanced_profile_macro'],flush=True);del data,outputs
  pairs=[('reg_raw','b14_gt32'),('reg_z','z_b14_gt32'),('reg_z','reg_raw'),('z_b14_gt32','b14_gt32')];contrasts={a+'_minus_'+b:prior.prior.contrasts(results,a,b)for a,b in pairs};interaction={}
  for split in SPLITS:
   interaction[split]={}
   for profile in ('raw','balanced_profile','spatial_profile'):
    diff=np.array(contrasts['reg_z_minus_reg_raw'][split][profile]['per_seed_macro_difference'])-np.array(contrasts['z_b14_gt32_minus_b14_gt32'][split][profile]['per_seed_macro_difference']);interaction[split][profile]={'per_seed_difference_in_differences':diff.tolist(),'mean':float(diff.mean())}
  selected=max(ARMS,key=lambda a:results[a]['mean_balanced_profile_macro']);matched='b14_gt32'if selected=='reg_raw'else'z_b14_gt32';eligible=selected in NEW and results[selected]['mean_balanced_profile_macro']>max(results[a]['mean_balanced_profile_macro']for a in OLD)and min(contrasts[selected+'_minus_'+matched]['balanced_val']['balanced_profile']['per_seed_macro_difference'])>0
  selection={'validation_selected_arm':selected,'matched_plain_reference':matched if selected in NEW else None,'new_arm_eligible_for_separate_confirmation':eligible,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{arm}_{seed}':file_hash(ckpt(arm,seed))for arm in ARMS for seed in range(3)},'register_statistics_sha256':stats['sha256'],'future_confirmation_manifest_sha256':p['future_confirmation_manifest_sha256'],'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'register_statistics':stats,'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'features').rglob('*.npz'))},'training_reports':reports,'results':results,'contrasts':contrasts,'interaction':interaction,'validation_selected_arm':selected,'new_arm_eligible_for_separate_confirmation':eligible,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE official register2x2 source control',selected,eligible,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
