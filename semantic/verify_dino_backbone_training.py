"""Independent metadata, B14 feature, sampler and 12-output validation audit."""
from collections import Counter
import hashlib,json
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from semantic import dino_backbone_training as work
from semantic.dino_backbone_resource import load_b14
from semantic.verify_external_semantic_confirmation import metric,equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def independent_b14(model,paths,xy):
 arrays=[]
 for path in paths:
  with Image.open(path)as im:arrays.append(np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32))
 video=torch.stack([torch.from_numpy(a).permute(2,0,1)/255 for a in arrays]).cuda()
 mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 with torch.inference_mode():
  norm=(F.interpolate(video,size=(448,448),mode='bicubic',align_corners=False)-mean)/std
  with torch.autocast('cuda',dtype=torch.bfloat16):tokens=model.forward_features(norm)['x_norm_patchtokens'].reshape(len(paths),32,32,768).permute(0,3,1,2)
  features=torch.empty((len(xy),768),device='cuda')
  for t in np.unique(xy[:,2]):
   rows=np.flatnonzero(xy[:,2]==t);grid=(((torch.from_numpy(xy[rows,:2]).float().cuda()+.5)/128)*2-1)[None,None]
   features[torch.as_tensor(rows,device='cuda')]=F.grid_sample(tokens[int(t):int(t)+1].float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
 return features.half().cpu().numpy()

def independent_eval(values,labels,times,offset):
 shifted=values.copy();shifted[:,0]+=offset;pred=shifted.argmax(1);result,matrix=metric(pred,labels);pos=labels>0
 result.update(foreground_accuracy=float((pred[pos]==labels[pos]).mean()),foreground_to_background_rate=float((pred[pos]==0).mean()),predicted_class_counts=np.bincount(pred,minlength=26).tolist(),per_frame={str(t):metric(pred[times==t],labels[times==t])[0]for t in range(8)},confusion_matrix=matrix.tolist())
 return result

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];m=json.loads((ROOT/'manifest.json').read_text());assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==58
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 old=json.loads((work.batch32.ROOT/'summary.json').read_text());assert m=={split:old['manifest'][split]for split in ('train','val')}
 packages={};oldval=[]
 for split in ('train','val'):
  packages[split]=[]
  for clip in sorted(m[split],key=lambda c:str(c['video_id'])):
   path=work.fp(split,clip['video_id'])
   with np.load(path)as a:record={k:a[k].copy()for k in a.files}
   with np.load(work.rgb.feature_path('native_sam',split,clip['video_id']))as a:prior={k:a[k].copy()for k in a.files}
   ix=prior['xy'][:,2]==0 if split=='train'else np.ones(len(prior['labels']),bool)
   for k in ['labels','xy','t_tgt','t_cam']:assert np.array_equal(record[k],prior[k][ix])
   assert int(record['video_id'])==int(prior['video_id'])and record['decoded'].shape==(int(ix.sum()),768)and record['decoded'].dtype==np.float16 and np.isfinite(record['decoded']).all()
   assert file_hash(path)==s['feature_hashes'][str(path)];packages[split].append(record)
   if split=='val':oldval.append(prior)
 assert sum(len(a['labels'])for a in packages['train'])==20480 and sum(len(a['labels'])for a in packages['val'])==40960
 model=load_b14();features_replayed={}
 for split in ('train','val'):
  clip=m[split][0];record=next(a for a in packages[split]if int(a['video_id'])==clip['video_id']);paths=clip['frames'][:1]if split=='train'else clip['frames']
  features=independent_b14(model,paths,record['xy']);assert np.array_equal(features,record['decoded']);features_replayed[split]=clip['video_id']
 del model;torch.cuda.empty_cache()
 labels=np.concatenate([a['labels']for a in packages['train']]);times=np.concatenate([a['xy'][:,2]for a in packages['train']]);assert np.all(times==0)
 y=torch.tensor(labels,device='cuda');fg=torch.where(y>0)[0];bg=torch.where(y==0)[0];samplers={}
 for arm in work.ARMS[2:]:
  count=128 if arm.endswith('256')else 16
  for seed in range(3):
   gen=torch.Generator(device='cuda').manual_seed(seed);hist=torch.zeros(26,dtype=torch.long,device='cuda');digest=hashlib.sha256()
   for step in range(2000):
    a=fg[torch.randint(len(fg),(count,),generator=gen,device='cuda')];b=bg[torch.randint(len(bg),(count,),generator=gen,device='cuda')];ix=torch.cat([a,b]);digest.update(ix.cpu().numpy().tobytes());hist+=torch.bincount(y[ix],minlength=26)
   r=json.loads(work.ckpt(arm,seed).with_suffix('.json').read_text());assert r==s['training_reports'][arm][seed]
   assert r['protocol']==p and r['sampled_indices_sha256']==digest.hexdigest()and r['sampled_class_counts']==hist.cpu().tolist()and r['source_exposures']==4000*count
   reference=json.loads(work.ckpt('S14_gt256'if count==128 else'S14_gt32',seed).with_suffix('.json').read_text());assert reference['sampled_class_counts']==r['sampled_class_counts']
   if count==16:assert reference['sampled_indices_sha256']==digest.hexdigest()
   assert [h['step']for h in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda h:h['val_macro_f1']);assert r['best_step']==best['step']and r['best_val']==best['val_macro_f1']
   checkpoint=torch.load(work.ckpt(arm,seed),map_location='cpu',weights_only=True);assert checkpoint['best_step']==best['step']and checkpoint['steps']==2000 and checkpoint['seed']==seed and checkpoint['protocol']==p
   state=checkpoint['model'];assert state['classifier.weight'].shape==(26,768)and sum(a.numel()for a in state.values())==r['parameter_count']==19994
   assert file_hash(work.ckpt(arm,seed))==r['checkpoint_sha256'];samplers[f'{arm}_{seed}']=digest.hexdigest();print('verified capacity sampler',arm,seed,flush=True)
 frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];results={};offsets=dict(p['fixed_offsets'])
 for arm in work.ARMS:
  data=oldval if arm.startswith('S14')else packages['val'];target=np.concatenate([a['labels']for a in data]);time=np.concatenate([a['xy'][:,2]for a in data]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in data]);outputs=[]
  for seed in range(3):
   state=torch.load(work.ckpt(arm,seed),map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in data])
   with np.load(ROOT/arm/'logits'/f'dinov2_{seed}_val.npz')as a:
    assert np.array_equal(a['logits'],values)and np.array_equal(a['labels'],target)and np.array_equal(a['times'],time)and np.array_equal(a['video_ids'],ids)and str(a['checkpoint_sha256'])==file_hash(work.ckpt(arm,seed))
   outputs.append(values)
   if arm.startswith('B14'):assert abs(independent_eval(values,target,time,0)['macro_f1_present']-s['training_reports'][arm][seed]['best_val'])<1e-12
  if arm.startswith('B14'):
   curves=[];best=-1;choice=None
   for offset in sorted(p['calibration_grid'],key=lambda x:(abs(x),x)):
    vals=[independent_eval(a,target,time,offset)['macro_f1_present']for a in outputs];value=float(np.mean(vals));curves.append({'offset':float(offset),'macro_f1_by_seed':vals,'mean_macro_f1':value})
    if value>best:best=value;choice=float(offset)
   equal(curves,frozen['curves'][arm]);offsets[arm]=choice
  runs=[{'seed':i,'raw_validation':independent_eval(a,target,time,0),'calibrated_validation':independent_eval(a,target,time,offsets[arm])}for i,a in enumerate(outputs)]
  rebuilt={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))};equal(rebuilt,s['results'][arm]);results[arm]=rebuilt
  for seed in range(3):assert frozen['checkpoint_sha256'][f'{arm}_{seed}']==file_hash(work.ckpt(arm,seed))
  print('verified capacity three heads/calibration',arm,flush=True)
 assert offsets==frozen['offsets'];selected=max(work.ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro']);assert selected==s['validation_selected_arm']==frozen['validation_selected_arm']
 assert not s['confirmation_evaluated']and s['deployment_unchanged']==frozen['deployment_unchanged']=='native_gt'
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'metadata_packages_replayed':100,'source_and_validation_queries':[20480,40960],'B14_feature_video_replays':features_replayed,'six_sampler_index_hashes':samplers,'all_six_histories_best_checkpoint_rules_recomputed':True,'all_twelve_logits_metrics_calibrations_and_selection_recomputed':True,'old_inputs_unchanged':True,'new_confirmation_groups':0})
 print('DONE independent B14 capacity training verification',selected,flush=True)
if __name__=='__main__':main()
