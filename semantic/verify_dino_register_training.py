"""Independent metadata selection, register feature replay, moments and24 logits."""
import hashlib,json
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from semantic import dino_register_training as work
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def manual_features(model,frames,xy):
 arrays=[]
 for path in frames:
  with Image.open(path)as im:arrays.append(np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32))
 video=torch.stack([torch.from_numpy(a).permute(2,0,1)/255 for a in arrays]).cuda();mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  image=(F.interpolate(video,size=(448,448),mode='bicubic',align_corners=False)-mean)/std;hidden=model.prepare_tokens_with_masks(image)
  for block in model.blocks:hidden=block(hidden)
  patches=model.norm(hidden)[:,5:].reshape(len(video),32,32,768).permute(0,3,1,2)
 values=torch.empty((len(xy),768),dtype=torch.float32,device='cuda')
 with torch.inference_mode():
  for t in sorted(set(map(int,xy[:,2]))):
   rows=np.flatnonzero(xy[:,2]==t);grid=(((torch.from_numpy(xy[rows,:2]).float().cuda()+.5)/128)*2-1)[None,None];values[torch.tensor(rows,device='cuda')]=F.grid_sample(patches[t:t+1].float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
 return values.half().cpu().numpy()

def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==90 and p['new_heads']==6
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 future=json.loads((ROOT/'future_confirmation_manifest.json').read_text());assert file_hash(ROOT/'future_confirmation_manifest.json')==p['future_confirmation_manifest_sha256']and future['metadata_previously_seen']and future['no_new_RGB_GT_mask_or_model_evaluation']
 meta=work.coco_resource.META;payload=json.loads((meta/'instances_val2017.json').read_bytes());mapping={int(k):int(v)for k,v in json.loads((meta/'summary.json').read_text())['shared_name_mapping'].items()};classes=sorted(set(mapping.values())-{1});pools={c:set()for c in classes}
 for ann in payload['annotations']:
  cid=mapping.get(ann['category_id'])
  if cid in pools and not ann['iscrowd']and ann['area']>0:pools[cid].add(ann['image_id'])
 early=json.loads((work.coco_resource.prior.old.ROOT/'manifest.json').read_text());recent=json.loads((work.coco_resource.prior.ROOT/'manifest.json').read_text());excluded={im['id']for role in ('candidate','reserve')for im in early[role]}|{im['id']for role in ('candidate','reserve')for im in recent[role]};assert len(excluded)==240 and future['excluded_previously_evaluated_COCO_ids']==sorted(excluded)
 images=sorted(payload['images'],key=lambda a:a['id']);order=np.random.default_rng(20261090).permutation(len(images));ranking=[images[int(i)]for i in order];seen=set(excluded);union=set().union(*pools.values())
 for role in ('candidate','reserve'):
  selected=[]
  for _ in range(2):
   for c in classes:
    eligible=[im for im in ranking if im['id']not in seen and im['id']in pools[c]]
    if eligible and len(selected)<80:selected.append(eligible[0]);seen.add(eligible[0]['id'])
  for im in ranking:
   if len(selected)<80 and im['id']not in seen and im['id']in union:selected.append(im);seen.add(im['id'])
  assert len(selected)==80 and future[role]==selected
 m=json.loads((ROOT/'manifest.json').read_text());data={split:[]for split in ('train',)+work.SPLITS}
 for split in data:
  for clip in sorted(m['train']if split=='train'else m['val'],key=lambda a:str(a['video_id'])):
   path=work.fp(split,clip['video_id']);assert file_hash(path)==s['feature_hashes'][str(path)]
   with np.load(path)as a:row={k:a[k].copy()for k in a.files}
   with np.load(work.prior.raw_path('b14_gt32',split,clip['video_id']))as a:reference={k:a[k].copy()for k in a.files if k!='decoded'}
   if split=='spatial_val':
    with np.load(work.prior.prior.spatial.qp(clip['video_id']))as a:reference['labels']=a['labels'].copy()
   assert set(row)==set(reference)|{'decoded'}
   for key,value in reference.items():assert np.array_equal(row[key],value)
   assert row['decoded'].dtype==np.float16 and row['decoded'].shape==(256 if split=='train'else 2048,768)and np.isfinite(row['decoded']).all()
   if split=='train':assert(row['xy'][:,2]==0).all()
   data[split].append(row)
 assert len(s['feature_hashes'])==120
 model=work.resource.load_register();replays=[]
 for split in ('train','val'):
  clip=m[split][0]
  for key in (('train',)if split=='train'else work.SPLITS):
   with np.load(work.fp(key,clip['video_id']))as a:xy=a['xy'].copy();expected=a['decoded'].copy()
   value=manual_features(model,clip['frames'][:1]if split=='train'else clip['frames'],xy);assert np.array_equal(value,expected);replays.append({'split':key,'video_id':clip['video_id']})
 del model;torch.cuda.empty_cache()
 x=np.concatenate([a['decoded']for a in data['train']]).astype(np.float64);assert len(x)==20480;mu64=np.sum(x,axis=0)/len(x);std64=np.sqrt(np.sum((x-mu64)**2,axis=0)/len(x));mu=mu64.astype(np.float32);scale=np.where(std64<=1e-6,1.,std64).astype(np.float32);stats=s['register_statistics']
 with np.load(stats['path'])as a:assert np.array_equal(a['mean'],mu)and np.array_equal(a['std'],std64.astype(np.float32))and np.array_equal(a['scale'],scale)
 assert file_hash(stats['path'])==stats['sha256']and stats['samples']==20480 and stats['dimension']==768 and stats['fallback_scale_channels']==int(np.sum(std64<=1e-6))
 equal(stats,json.loads((ROOT/'statistics_manifest.json').read_text()));labels=np.concatenate([a['labels']for a in data['train']]);fg=torch.tensor(np.flatnonzero(labels>0),device='cuda');bg=torch.tensor(np.flatnonzero(labels==0),device='cuda');hashes={}
 for seed in range(3):
  gen=torch.Generator(device='cuda').manual_seed(seed);digest=hashlib.sha256();hist=np.zeros(26,np.int64)
  for _ in range(2000):
   f=fg[torch.randint(len(fg),(16,),generator=gen,device='cuda')];b=bg[torch.randint(len(bg),(16,),generator=gen,device='cuda')];ix=torch.cat([f,b]).cpu().numpy();digest.update(ix.tobytes());hist+=np.bincount(labels[ix],minlength=26)
  hashes[str(seed)]=digest.hexdigest();assert hist.sum()==64000
  reference=json.loads(work.ckpt('b14_gt32',seed).with_suffix('.json').read_text());assert reference['sampled_indices_sha256']==digest.hexdigest()and reference['sampled_class_counts']==hist.tolist()
  for arm in work.NEW:
   path=work.ckpt(arm,seed);r=json.loads(path.with_suffix('.json').read_text());assert r==s['training_reports'][arm][seed]and r['protocol']==p and r['register_statistics_sha256']==stats['sha256']and r['sampled_indices_sha256']==digest.hexdigest()and r['sampled_class_counts']==hist.tolist()
   assert[r['step']for r in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda a:a['balanced_raw_val_macro']);assert best['step']==r['best_step']and best['balanced_raw_val_macro']==r['best_balanced_raw_val']
   cp=torch.load(path,map_location='cpu',weights_only=True);assert cp['protocol']==p and cp['seed']==seed and cp['steps']==2000 and cp['best_step']==r['best_step']and cp['register_statistics_sha256']==stats['sha256']and sum(v.numel()for v in cp['model'].values())==r['head_parameters']==19994 and file_hash(path)==r['checkpoint_sha256']
 print('verified future cohorts,120 metadata,3 manual replays,training statistics and6 budgets',flush=True)
 frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];results={};offsets={}
 for arm in work.ARMS:
  outputs={}
  for split in work.SPLITS:
   if arm in work.OLD:rows=work.prior.read(arm,split,m,{'b14_gt32':p['plain_statistics']})
   elif arm=='reg_raw':rows=data[split]
   else:rows=[{**a,'decoded':(a['decoded'].astype(np.float32)-mu)/scale}for a in data[split]]
   y=np.concatenate([a['labels']for a in rows]);times=np.concatenate([a['xy'][:,2]for a in rows]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in rows]);outputs[split]=[]
   for seed in range(3):
    path=work.ckpt(arm,seed);assert file_hash(path)==frozen['checkpoint_sha256'][f'{arm}_{seed}'];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in rows])
    with np.load(ROOT/arm/'logits'/f'seed{seed}_{split}.npz')as a:
     for key,v in [('logits',values),('labels',y),('times',times),('video_ids',ids)]:assert np.array_equal(a[key],v)
     assert str(a['checkpoint_sha256'])==file_hash(path)
    outputs[split].append({'logits':values,'labels':y,'times':times})
    if arm in work.NEW and split=='balanced_val':assert abs(independent_eval(values,y,times,0)['macro_f1_present']-s['training_reports'][arm][seed]['best_balanced_raw_val'])<1e-12
  offsets[arm]={'raw':0.}
  for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
   if arm in work.OLD:offsets[arm][profile]=p['fixed_plain_offsets'][arm][profile];continue
   curve=[];best=-1;choice=None
   for offset in sorted(p['calibration_grid'],key=lambda a:(abs(a),a)):
    values=[independent_eval(a['logits'],a['labels'],a['times'],offset)['macro_f1_present']for a in outputs[split]];mean=float(np.mean(values));curve.append({'offset':float(offset),'macro_f1_by_seed':values,'mean_macro_f1':mean})
    if mean>best:best=mean;choice=float(offset)
   equal(curve,frozen['curves'][arm][profile]);offsets[arm][profile]=choice
  runs=[{'seed':seed,**{split:{profile:independent_eval(outputs[split][seed]['logits'],outputs[split][seed]['labels'],outputs[split][seed]['times'],offset)for profile,offset in offsets[arm].items()}for split in work.SPLITS}}for seed in range(3)];results[arm]={'offsets':offsets[arm],'runs':runs,'mean_balanced_profile_macro':float(np.mean([a['balanced_val']['balanced_profile']['macro_f1_present']for a in runs]))};equal(results[arm],s['results'][arm]);print('verified24 raw outputs and fixed source profiles',arm,flush=True)
 pairs=[('reg_raw','b14_gt32'),('reg_z','z_b14_gt32'),('reg_z','reg_raw'),('z_b14_gt32','b14_gt32')];contrasts={}
 for left,right in pairs:
  contrasts[left+'_minus_'+right]={}
  for split in work.SPLITS:
   contrasts[left+'_minus_'+right][split]={}
   for profile in ('raw','balanced_profile','spatial_profile'):
    values=[results[left]['runs'][seed][split][profile]['macro_f1_present']-results[right]['runs'][seed][split][profile]['macro_f1_present']for seed in range(3)];contrasts[left+'_minus_'+right][split][profile]={'per_seed_macro_difference':values,'mean':float(np.mean(values))}
 equal(contrasts,s['contrasts'])
 for split in work.SPLITS:
  for profile in ('raw','balanced_profile','spatial_profile'):
   values=np.asarray(contrasts['reg_z_minus_reg_raw'][split][profile]['per_seed_macro_difference'])-np.asarray(contrasts['z_b14_gt32_minus_b14_gt32'][split][profile]['per_seed_macro_difference']);equal({'per_seed_difference_in_differences':values.tolist(),'mean':float(values.mean())},s['interaction'][split][profile])
 selected=max(work.ARMS,key=lambda a:results[a]['mean_balanced_profile_macro']);matched='b14_gt32'if selected=='reg_raw'else'z_b14_gt32';eligible=selected in work.NEW and results[selected]['mean_balanced_profile_macro']>max(results[a]['mean_balanced_profile_macro']for a in work.OLD)and min(contrasts[selected+'_minus_'+matched]['balanced_val']['balanced_profile']['per_seed_macro_difference'])>0
 assert selected==s['validation_selected_arm']==frozen['validation_selected_arm']and eligible==s['new_arm_eligible_for_separate_confirmation']==frozen['new_arm_eligible_for_separate_confirmation']and frozen['offsets']==offsets and frozen['future_confirmation_manifest_sha256']==p['future_confirmation_manifest_sha256']and frozen['register_statistics_sha256']==stats['sha256']
 assert frozen['matched_plain_reference']==(matched if selected in work.NEW else None)and s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt'
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'future_metadata_candidate80_reserve80_reconstructed_excluding240':True,'all120_feature_metadata_and_source0_labels_verified':True,'independent_register_feature_replays':replays,'training_only_population_statistics_recomputed':True,'three_complete_sampling_replays':hashes,'all6_head_budgets_and_checkpoint_selections_verified':True,'all24_logits_four_shared_calibration_curves_metrics_contrasts_interactions_recomputed':True,'source_selection_confirmation_eligibility_and_all_frozen_inputs_verified':True,'no_new_confirmation_or_deployment_change':True});print('DONE independent register2x2 source verification',selected,eligible,flush=True)
if __name__=='__main__':main()
