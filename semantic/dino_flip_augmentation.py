"""Frozen RGB flip views replace half the source query exposures, not add updates."""
import fcntl,gc,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import feature_standardization_control as prior
from semantic import coco_sam_auxiliary as dino
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import evaluate,pick_offset
from semantic.semantic_feature_control import save_npz,patch_at_queries
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_flip_augmentation');REFERENCE='z_b14_gt32';NEW='flip_mix';ARMS=(REFERENCE,NEW);SPLITS=prior.SPLITS

def fp(vid):return ROOT/'features'/f'{vid}.npz'
def ckpt(arm,seed):return prior.ckpt(REFERENCE,seed)if arm==REFERENCE else ROOT/arm/'runs'/f'linear_seed{seed}.pt'
def mirror_queries(xy):
 if xy.ndim!=2 or xy.shape[1]!=3 or not np.isfinite(xy).all() or (xy[:,:2]<0).any()or(xy[:,:2]>127).any()or(xy[:,2]!=0).any():raise ValueError('Finite source0 query coordinates in0..127 required')
 result=xy.copy();result[:,0]=127-result[:,0];return result

def freeze():
 source=json.loads((prior.ROOT/'summary.json').read_text());assert file_hash(prior.ROOT/'summary.json')==json.loads((prior.ROOT/'verification.json').read_text())['summary_sha256']
 stats=source['statistics']['b14_gt32'];assert file_hash(stats['path'])==stats['sha256'];m=json.loads((prior.ROOT/'manifest.json').read_text())
 paths=[Path(x)for x in source['protocol']['input_sha256']]+[prior.ROOT/n for n in ('summary.json','verification.json','manifest.json','statistics_manifest.json')]+[Path(stats['path']),dino.WEIGHTS,Path('semantic/dino_flip_augmentation.py'),Path('semantic/OVIS_DINO_FLIP_AUGMENTATION_PROTOCOL.md'),Path('tests/test_dino_flip_coordinates.py'),Path('semantic/feature_standardization_control.py')]+sorted(dino.VENDOR.rglob('*.py'))
 for clip in m['train']:paths.append(Path(clip['frames'][0]))
 for seed in range(3):
  paths.extend([ckpt(REFERENCE,seed),ckpt(REFERENCE,seed).with_suffix('.json')]);paths.extend(prior.ROOT/REFERENCE/'logits'/f'seed{seed}_{split}.npz'for split in SPLITS)
 assert file_hash(dino.WEIGHTS)=='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
 test=subprocess.run([str(Path('.venv/bin/python')),'tests/test_dino_flip_coordinates.py'],capture_output=True,text=True,check=True);(ROOT/'unit_checks.log').write_text(test.stdout+test.stderr);assert 'PASS'in test.stdout
 p={'round':88,'new_heads':3,'steps':2000,'batch':32,'exposures':64000,'feature_dimension':768,'head_parameters':19994,'view_seed_base':20261088,'flip_probability':.5,'statistics':stats,'fixed_reference_offsets':source['results'][REFERENCE]['offsets'],'calibration_grid':source['protocol']['calibration_grid'],'source_validation_only':True,'unit_checks_sha256':file_hash(ROOT/'unit_checks.log'),'input_sha256':{str(path):file_hash(path)for path in sorted(set(paths))}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',m);return m,p

def cache(m):
 tasks=[clip for clip in m['train']if not fp(clip['video_id']).exists()]
 if not tasks:return
 model=dino.load_b14()
 for clip in tasks:
  with np.load(prior.raw_path('b14_gt32','train',clip['video_id']))as a:meta={k:a[k].copy()for k in a.files if k!='decoded'}
  xy=mirror_queries(meta['xy']);video=dino.rgb.load_native_video({'frames':clip['frames'][:1]}).cuda();video=torch.flip(video,dims=[-1]);tokens=dino.tokens_b14(model,video)
  with torch.inference_mode():decoded=patch_at_queries(tokens,xy).half().cpu().numpy()
  assert decoded.shape==(256,768)and np.isfinite(decoded).all();save_npz(fp(clip['video_id']),**meta,decoded=decoded,sampling_xy=xy,augmentation=np.array('horizontal_flip'))
  print('source flip cache',clip['video_id'],flush=True)
 del model;gc.collect();torch.cuda.empty_cache()

def flipped_data(m,stats):
 with np.load(stats['path'])as a:mean=a['mean'].copy();scale=a['scale'].copy()
 rows=[]
 for clip in sorted(m['train'],key=lambda c:str(c['video_id'])):
  with np.load(fp(clip['video_id']))as a:row={k:a[k].copy()for k in a.files}
  row['decoded']=(row['decoded'].astype(np.float32)-mean)/scale;rows.append(row)
 return rows

def train(seed,base,flipped,val,p):
 path=ckpt(NEW,seed);report=path.with_suffix('.json')
 if report.exists():
  r=json.loads(report.read_text());assert r['protocol']==p and file_hash(path)==r['checkpoint_sha256'];return r
 torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();x=torch.from_numpy(np.concatenate([a['decoded']for a in base])).cuda();z=torch.from_numpy(np.concatenate([a['decoded']for a in flipped])).cuda();y=np.concatenate([a['labels']for a in base]);times=np.concatenate([a['xy'][:,2]for a in base])
 assert len(y)==20480 and (times==0).all()and x.dtype==z.dtype==torch.float32
 for a,b in zip(base,flipped):assert np.array_equal(a['labels'],b['labels'])and np.array_equal(a['xy'],b['xy'])
 sampler=Sampler(y,times,'baseline');gen=torch.Generator(device='cuda').manual_seed(seed);views=torch.Generator(device='cuda').manual_seed(20261088+seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);ih=hashlib.sha256();vh=hashlib.sha256();hist=torch.zeros(26,dtype=torch.long,device='cuda');flip_hist=hist.clone();best=-1;history=[]
 for step in range(1,2001):
  fg,bg=sampler.draw(gen,count=16);ix=torch.cat([fg,bg]);chosen=torch.rand(32,generator=views,device='cuda')<.5;ih.update(ix.cpu().numpy().tobytes());vh.update(chosen.cpu().numpy().tobytes());hist+=torch.bincount(sampler.labels[ix],minlength=26);flip_hist+=torch.bincount(sampler.labels[ix[chosen]],minlength=26)
  values=torch.where(chosen[:,None],z[ix],x[ix]);model.train();optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(values),sampler.labels[ix]);assert torch.isfinite(loss);loss.backward();optimizer.step()
  if step%50==0:
   value=score(model,val);history.append({'step':step,'balanced_raw_val_macro':value,'loss':float(loss.detach())})
   if value>best:best=value;best_step=step;state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
 reference=json.loads(ckpt(REFERENCE,seed).with_suffix('.json').read_text());assert ih.hexdigest()==reference['sampled_indices_sha256']and hist.cpu().tolist()==reference['sampled_class_counts']
 path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix('.partial.pt');torch.save({'model':state,'seed':seed,'steps':2000,'best_step':best_step,'protocol':p},partial);partial.replace(path)
 r={'seed':seed,'protocol':p,'best_step':best_step,'best_balanced_raw_val':best,'history':history,'sampled_indices_sha256':ih.hexdigest(),'sampled_class_counts':hist.cpu().tolist(),'sampled_views_sha256':vh.hexdigest(),'flipped_class_exposure_counts':flip_hist.cpu().tolist(),'flipped_exposures':int(flip_hist.sum()),'original_exposures':64000-int(flip_hist.sum()),'head_parameters':19994,'checkpoint_sha256':file_hash(path)};atomic_json(report,r);print('trained flip replacement',seed,best_step,best,r['flipped_exposures'],flush=True);del model,x,z;gc.collect();torch.cuda.empty_cache();return r

def infer(arm,seed,split,data):
 path=ROOT/arm/'logits'/f'seed{seed}_{split}.npz';h=file_hash(ckpt(arm,seed))
 if path.exists():
  with np.load(path)as a:out={k:a[k].copy()for k in a.files}
  assert str(out['checkpoint_sha256'])==h;return out
 if arm==REFERENCE:
  with np.load(prior.ROOT/REFERENCE/'logits'/f'seed{seed}_{split}.npz')as a:out={k:a[k].copy()for k in a.files}
 else:
  state=torch.load(ckpt(arm,seed),map_location='cpu',weights_only=True)['model']
  with torch.inference_mode():values=[F.linear(torch.from_numpy(a['decoded']).cuda(),state['classifier.weight'].cuda(),state['classifier.bias'].cuda()).cpu().numpy()for a in data]
  out={'logits':np.concatenate(values),'labels':np.concatenate([a['labels']for a in data]),'times':np.concatenate([a['xy'][:,2]for a in data]),'video_ids':np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in data]),'checkpoint_sha256':np.array(h)}
 save_npz(path,**out);return out

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed frozen flip control exists')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  m,p=freeze();cache(m);allstats=json.loads((prior.ROOT/'statistics_manifest.json').read_text());base=prior.read(REFERENCE,'train',m,allstats);flipped=flipped_data(m,p['statistics']);datasets={split:prior.read(REFERENCE,split,m,allstats)for split in SPLITS};reports=[train(seed,base,flipped,datasets['balanced_val'],p)for seed in range(3)];results={};offsets={};curves={}
  for arm in ARMS:
   outputs={split:[infer(arm,seed,split,datasets[split])for seed in range(3)]for split in SPLITS};offsets[arm]={'raw':0.};curves[arm]={}
   for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
    if arm==REFERENCE:offsets[arm][profile]=p['fixed_reference_offsets'][profile]
    else:offsets[arm][profile],curves[arm][profile]=pick_offset(outputs[split],p['calibration_grid'])
   runs=[{'seed':seed,**{split:{profile:evaluate(outputs[split][seed],offset)for profile,offset in offsets[arm].items()}for split in SPLITS}}for seed in range(3)]
   results[arm]={'offsets':offsets[arm],'runs':runs,'mean_balanced_profile_macro':float(np.mean([a['balanced_val']['balanced_profile']['macro_f1_present']for a in runs]))}
  selected=max(ARMS,key=lambda a:results[a]['mean_balanced_profile_macro']);contrast=prior.prior.contrasts(results,NEW,REFERENCE);eligible=min(contrast['balanced_val']['balanced_profile']['per_seed_macro_difference'])>0
  selection={'validation_selected_arm':selected,'offsets':offsets,'curves':curves,'checkpoint_sha256':{f'{arm}_{seed}':file_hash(ckpt(arm,seed))for arm in ARMS for seed in range(3)},'new_arm_eligible_for_separate_confirmation':eligible,'deployment_unchanged':'native_gt'};atomic_json(ROOT/'frozen_selection.json',selection)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'feature_hashes':{str(fp(clip['video_id'])):file_hash(fp(clip['video_id']))for clip in m['train']},'training_reports':reports,'results':results,'flip_minus_reference_contrast':contrast,'validation_selected_arm':selected,'new_arm_eligible_for_separate_confirmation':eligible,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE matched-exposure source flip control',selected,eligible,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
