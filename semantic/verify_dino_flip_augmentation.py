"""Independent native flip extraction and full independent index/view generator replay."""
import hashlib,json
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from semantic import dino_flip_augmentation as work
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==88 and p['new_heads']==3
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert file_hash(ROOT/'unit_checks.log')==p['unit_checks_sha256']and 'PASS'in(ROOT/'unit_checks.log').read_text()
 m=json.loads((ROOT/'manifest.json').read_text());assert len(s['feature_hashes'])==80
 for clip in m['train']:
  path=work.fp(clip['video_id']);assert file_hash(path)==s['feature_hashes'][str(path)]
  with np.load(work.prior.raw_path('b14_gt32','train',clip['video_id']))as old:reference={k:old[k].copy()for k in old.files if k!='decoded'}
  with np.load(path)as a:
   assert set(a.files)==set(reference)|{'decoded','sampling_xy','augmentation'}
   for key,value in reference.items():assert np.array_equal(a[key],value)
   assert (a['xy'][:,2]==0).all()and (a['sampling_xy'][:,2]==0).all()and np.array_equal(a['sampling_xy'][:,0],127-reference['xy'][:,0])and np.array_equal(a['sampling_xy'][:,1:],reference['xy'][:,1:])
   assert str(a['augmentation'])=='horizontal_flip'and a['decoded'].shape==(256,768)and a['decoded'].dtype==np.float16 and np.isfinite(a['decoded']).all()
 first=m['train'][0];model=work.dino.load_b14()
 with Image.open(first['frames'][0])as im:rgb=np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32)
 video=torch.from_numpy(rgb[:,::-1].copy()).permute(2,0,1)[None].cuda()/255
 with torch.inference_mode():
  mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
  native=(F.interpolate(video,size=(448,448),mode='bicubic',align_corners=False)-mean)/std
  with torch.autocast('cuda',dtype=torch.bfloat16):patch=model.forward_features(native)['x_norm_patchtokens'].reshape(1,32,32,768).permute(0,3,1,2)
  with np.load(work.fp(first['video_id']))as a:xy=a['sampling_xy'].copy();expected=a['decoded'].copy()
  grid=torch.from_numpy(((xy[:,:2].astype(np.float32)+.5)/128)*2-1).cuda()[None,None]
  decoded=F.grid_sample(patch.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T.half().cpu().numpy()
 assert np.array_equal(decoded,expected);del model;torch.cuda.empty_cache()
 stats=json.loads((work.prior.ROOT/'statistics_manifest.json').read_text());datasets={split:work.prior.read(work.REFERENCE,split,m,stats)for split in ('train',)+work.SPLITS};labels=np.concatenate([a['labels']for a in datasets['train']]);assert len(labels)==20480
 fg=torch.tensor(np.flatnonzero(labels>0),device='cuda');bg=torch.tensor(np.flatnonzero(labels==0),device='cuda');replays={}
 for seed in range(3):
  index_generator=torch.Generator(device='cuda').manual_seed(seed);view_generator=torch.Generator(device='cuda').manual_seed(20261088+seed);ih=hashlib.sha256();vh=hashlib.sha256();hist=np.zeros(26,np.int64);flip_hist=np.zeros(26,np.int64)
  for _ in range(2000):
   f=fg[torch.randint(len(fg),(16,),generator=index_generator,device='cuda')];b=bg[torch.randint(len(bg),(16,),generator=index_generator,device='cuda')];ix=torch.cat([f,b]).cpu().numpy();chosen=(torch.rand(32,generator=view_generator,device='cuda')<.5).cpu().numpy();ih.update(ix.tobytes());vh.update(chosen.tobytes());hist+=np.bincount(labels[ix],minlength=26);flip_hist+=np.bincount(labels[ix[chosen]],minlength=26)
  r=s['training_reports'][seed];assert r==json.loads(work.ckpt(work.NEW,seed).with_suffix('.json').read_text())and r['protocol']==p and r['sampled_indices_sha256']==ih.hexdigest()and r['sampled_views_sha256']==vh.hexdigest()and r['sampled_class_counts']==hist.tolist()and r['flipped_class_exposure_counts']==flip_hist.tolist()and r['flipped_exposures']==int(flip_hist.sum())and r['original_exposures']==64000-int(flip_hist.sum())and hist.sum()==64000
  ref=json.loads(work.ckpt(work.REFERENCE,seed).with_suffix('.json').read_text());assert ih.hexdigest()==ref['sampled_indices_sha256']and hist.tolist()==ref['sampled_class_counts']
  assert [row['step']for row in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda a:a['balanced_raw_val_macro']);assert best['step']==r['best_step']and best['balanced_raw_val_macro']==r['best_balanced_raw_val']
  cp=torch.load(work.ckpt(work.NEW,seed),map_location='cpu',weights_only=True);assert cp['protocol']==p and cp['steps']==2000 and cp['seed']==seed and cp['best_step']==r['best_step']and sum(v.numel()for v in cp['model'].values())==r['head_parameters']==19994 and file_hash(work.ckpt(work.NEW,seed))==r['checkpoint_sha256']
  replays[str(seed)]={'index_hash':ih.hexdigest(),'view_hash':vh.hexdigest(),'flipped_exposures':int(flip_hist.sum())}
 frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];results={}
 for arm in work.ARMS:
  outputs={}
  for split in work.SPLITS:
   rows=datasets[split];y=np.concatenate([a['labels']for a in rows]);times=np.concatenate([a['xy'][:,2]for a in rows]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in rows]);outputs[split]=[]
   for seed in range(3):
    path=work.ckpt(arm,seed);state=torch.load(path,map_location='cpu',weights_only=True)['model'];assert file_hash(path)==frozen['checkpoint_sha256'][f'{arm}_{seed}'];values=np.concatenate([linear(a['decoded'],state)for a in rows])
    with np.load(ROOT/arm/'logits'/f'seed{seed}_{split}.npz')as a:
     for key,value in [('logits',values),('labels',y),('times',times),('video_ids',ids)]:assert np.array_equal(a[key],value)
     assert str(a['checkpoint_sha256'])==file_hash(path)
    outputs[split].append({'logits':values,'labels':y,'times':times})
    if arm==work.NEW and split=='balanced_val':assert abs(independent_eval(values,y,times,0)['macro_f1_present']-s['training_reports'][seed]['best_balanced_raw_val'])<1e-12
  offsets={'raw':0.}
  for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
   if arm==work.REFERENCE:offsets[profile]=p['fixed_reference_offsets'][profile];continue
   best=-1;choice=None;curve=[]
   for offset in sorted(p['calibration_grid'],key=lambda v:(abs(v),v)):
    values=[independent_eval(a['logits'],a['labels'],a['times'],offset)['macro_f1_present']for a in outputs[split]];mean=float(np.mean(values));curve.append({'offset':float(offset),'macro_f1_by_seed':values,'mean_macro_f1':mean})
    if mean>best:best=mean;choice=float(offset)
   equal(curve,frozen['curves'][arm][profile]);offsets[profile]=choice
  runs=[{'seed':seed,**{split:{profile:independent_eval(outputs[split][seed]['logits'],outputs[split][seed]['labels'],outputs[split][seed]['times'],offset)for profile,offset in offsets.items()}for split in work.SPLITS}}for seed in range(3)];results[arm]={'offsets':offsets,'runs':runs,'mean_balanced_profile_macro':float(np.mean([a['balanced_val']['balanced_profile']['macro_f1_present']for a in runs]))};equal(results[arm],s['results'][arm])
 comparison={}
 for split in work.SPLITS:
  comparison[split]={}
  for profile in ('raw','balanced_profile','spatial_profile'):
   values=[results[work.NEW]['runs'][seed][split][profile]['macro_f1_present']-results[work.REFERENCE]['runs'][seed][split][profile]['macro_f1_present']for seed in range(3)];comparison[split][profile]={'per_seed_macro_difference':values,'mean':float(np.mean(values))}
 equal(comparison,s['flip_minus_reference_contrast']);selected=max(work.ARMS,key=lambda a:results[a]['mean_balanced_profile_macro']);eligible=min(comparison['balanced_val']['balanced_profile']['per_seed_macro_difference'])>0
 assert selected==s['validation_selected_arm']==frozen['validation_selected_arm']and eligible==s['new_arm_eligible_for_separate_confirmation']==frozen['new_arm_eligible_for_separate_confirmation']
 assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt'
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all80_flip_metadata_sampling_coordinates_and_label_identity_verified':True,'first562_native_flip_and_manual_B14_patch_feature_recomputed_bitwise':True,'three_complete_separate_index_view_generator_replays':replays,'three_head_budgets_and_checkpoint_selection_verified':True,'all12_validation_logits_two_calibration_curves_and_profile_metrics_recomputed':True,'all_frozen_inputs_unchanged':True,'no_new_confirmation_or_deployment_change':True});print('DONE independent matched-exposure horizontal flip control',selected,eligible,flush=True)
if __name__=='__main__':main()
