"""Independent block-hook features, metadata/sampling and six-output audit."""
import hashlib,json
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from semantic import dino_b14_multilayer as work
from semantic.dino_backbone_resource import load_b14,VENDOR
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def hook_features(model,paths,xy):
 frames=[]
 for path in paths:
  with Image.open(path)as im:a=np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32)
  frames.append(torch.from_numpy(a.copy()).permute(2,0,1)/255)
 video=torch.stack(frames).cuda();mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];collected=[];handles=[block.register_forward_hook(lambda module,args,result:collected.append(result.detach().clone()))for block in list(model.blocks)[-4:]]
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   final=model.forward_features((F.interpolate(video,size=(448,448),mode='bicubic',align_corners=False)-mean)/std);normalized=[model.norm(out)[:,1+model.num_register_tokens:]for out in collected]
 finally:
  for handle in handles:handle.remove()
 assert len(normalized)==4 and torch.equal(normalized[-1],final['x_norm_patchtokens']);pooled=torch.stack([a.float()for a in normalized]).mean(0);result=[]
 with torch.inference_mode():
  for tokens in (pooled,normalized[-1].float()):
   grids=tokens.reshape(len(paths),32,32,768).permute(0,3,1,2);out=torch.empty(len(xy),768,device='cuda')
   for t in np.unique(xy[:,2]):
    rows=np.flatnonzero(xy[:,2]==t);positions=torch.from_numpy(xy[rows,:2]).float().cuda();grid=(((positions+.5)/128)*2-1)[None,None];out[torch.from_numpy(rows).cuda()]=F.grid_sample(grids[int(t):int(t)+1],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
   result.append(out.half().cpu().numpy())
 return result

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==76 and p['feature_dimension']==768 and p['head_parameters']==19994
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 old=json.loads((work.prior.ROOT/'summary.json').read_text());m=json.loads((ROOT/'manifest.json').read_text());assert m==json.loads((work.prior.ROOT/'manifest.json').read_text());source_vendor={str(x)for x in VENDOR.rglob('*.py')}
 for path in source_vendor|{str(work.WEIGHTS)}:assert p['input_sha256'][path]==old['protocol']['input_sha256'][path]
 cost=json.loads((ROOT/'extraction.json').read_text());assert file_hash(ROOT/'extraction.json')==s['extraction_sha256']and len(cost)==100;costs={(x['split'],x['video_id']):x for x in cost};assert len(costs)==100;data={};olddata={}
 for split in m:
  data[split]=[];olddata[split]=[]
  for c in sorted(m[split],key=lambda c:str(c['video_id'])):
   path=work.fp(split,c['video_id']);assert file_hash(path)==s['feature_hashes'][str(path)]
   with np.load(path)as a:new={k:a[k].copy()for k in a.files}
   with np.load(work.prior.fp(split,c['video_id']))as a:prior={k:a[k].copy()for k in a.files}
   assert new.keys()==prior.keys()
   for k in new.keys()-{'decoded'}:assert np.array_equal(new[k],prior[k])
   assert new['decoded'].shape==prior['decoded'].shape and new['decoded'].dtype==np.float16 and np.isfinite(new['decoded']).all();assert len(new['labels'])==(256 if split=='train'else 2048)
   if split=='train':assert (new['xy'][:,2]==0).all()
   row=costs[(split,c['video_id'])];assert row['last_layer_bitwise_equal']and row['last_layer_values_sha256']==hashlib.sha256(prior['decoded'].tobytes()).hexdigest()and row['pooled_patch_dtype']=='torch.float32'and row['seconds']>0
   data[split].append(new);olddata[split].append(prior)
 assert set(s['feature_hashes'])=={str(path)for path in (ROOT/'features').rglob('*.npz')}
 model=load_b14();replay={}
 for split in ('train','val'):
  clip=m[split][0];i=next(i for i,c in enumerate(sorted(m[split],key=lambda c:str(c['video_id'])))if c['video_id']==clip['video_id']);new=data[split][i];oldcache=olddata[split][i];features,last=hook_features(model,clip['frames'][:1]if split=='train'else clip['frames'],new['xy']);assert np.array_equal(features,new['decoded'])and np.array_equal(last,oldcache['decoded']);replay[split]=clip['video_id'];print('verified independent last4 block hooks',split,clip['video_id'],flush=True)
 del model;torch.cuda.empty_cache();labels=np.concatenate([a['labels']for a in data['train']]);y=torch.tensor(labels,device='cuda');fg=torch.where(y>0)[0];bg=torch.where(y==0)[0]
 for seed in range(3):
  gen=torch.Generator(device='cuda').manual_seed(seed);ih=hashlib.sha256();hist=np.zeros(26,np.int64)
  for step in range(2000):
   f=fg[torch.randint(len(fg),(16,),generator=gen,device='cuda')];b=bg[torch.randint(len(bg),(16,),generator=gen,device='cuda')];ix=torch.cat([f,b]).cpu().numpy();ih.update(ix.tobytes());hist+=np.bincount(labels[ix],minlength=26)
  path=work.ckpt('mean_last4',seed);r=json.loads(path.with_suffix('.json').read_text());assert r==s['training_reports'][seed]and r['protocol']==p and ih.hexdigest()==r['sampled_indices_sha256']and hist.tolist()==r['sampled_class_counts']and hist.sum()==64000
  oldr=json.loads(work.ckpt('final',seed).with_suffix('.json').read_text());assert r['sampled_indices_sha256']==oldr['sampled_indices_sha256']and r['sampled_class_counts']==oldr['sampled_class_counts'];assert [a['step']for a in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda a:a['val_macro_f1']);assert best['step']==r['best_step']and best['val_macro_f1']==r['best_val'];checkpoint=torch.load(path,map_location='cpu',weights_only=True);assert checkpoint['protocol']==p and checkpoint['seed']==seed and checkpoint['steps']==2000 and checkpoint['best_step']==best['step']and sum(a.numel()for a in checkpoint['model'].values())==r['parameter_count']==19994 and file_hash(path)==r['checkpoint_sha256']
 target=np.concatenate([a['labels']for a in data['val']]);ts=np.concatenate([a['xy'][:,2]for a in data['val']]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in data['val']]);frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];results={};offsets={'final':p['fixed_final_offset']}
 for arm in work.ARMS:
  outputs=[]
  for seed in range(3):
   path=work.ckpt(arm,seed);assert file_hash(path)==frozen['checkpoint_sha256'][f'{arm}_{seed}'];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in (data['val']if arm=='mean_last4'else olddata['val'])]);outputs.append(values)
   with np.load(ROOT/arm/'logits'/f'seed{seed}_val.npz')as a:assert np.array_equal(values,a['logits'])and np.array_equal(a['labels'],target)and np.array_equal(a['times'],ts)and np.array_equal(a['video_ids'],ids)and str(a['checkpoint_sha256'])==file_hash(path)
   if arm=='mean_last4':assert abs(independent_eval(values,target,ts,0)['macro_f1_present']-s['training_reports'][seed]['best_val'])<1e-12
  if arm=='mean_last4':
   curve=[];best=-1;choice=None
   for offset in sorted(p['calibration_grid'],key=lambda v:(abs(v),v)):
    vals=[independent_eval(v,target,ts,offset)['macro_f1_present']for v in outputs];mean=float(np.mean(vals));curve.append({'offset':float(offset),'macro_f1_by_seed':vals,'mean_macro_f1':mean})
    if mean>best:best=mean;choice=float(offset)
   equal(curve,frozen['curves'][arm]);offsets[arm]=choice
  runs=[{'seed':seed,'raw_validation':independent_eval(v,target,ts,0),'calibrated_validation':independent_eval(v,target,ts,offsets[arm])}for seed,v in enumerate(outputs)];r={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([a['calibrated_validation']['macro_f1_present']for a in runs]))};equal(r,s['results'][arm]);results[arm]=r
 differences=[results['mean_last4']['runs'][seed]['calibrated_validation']['macro_f1_present']-results['final']['runs'][seed]['calibrated_validation']['macro_f1_present']for seed in range(3)];equal({'per_seed_validation_macro_difference':differences,'mean':float(np.mean(differences))},s['mean_last4_minus_final']);assert max(work.ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro'])==s['validation_selected_arm']==frozen['validation_selected_arm']and offsets==frozen['offsets']
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all100_query_metadata_and_last_layer_value_hashes_checked':True,'independent_block_hook_mean_and_last_feature_replays':replay,'all3_sampler_histogram_and_checkpoint_rules_recomputed':True,'all6_val_logits_and_calibration_contrast_selection_recomputed':True,'vendor_and_backbone_match_original58_fingerprints':True,'all_frozen_inputs_unchanged':True,'no_confirmation_or_deployment_change':True});print('DONE independent last4 mean feature audit',flush=True)
if __name__=='__main__':main()
