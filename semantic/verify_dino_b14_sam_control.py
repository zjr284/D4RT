"""Independent B14 teacher metadata/features, sampler and factorial val audit."""
import hashlib,json
import numpy as np
import torch
from semantic import dino_b14_sam_control as work
from semantic.verify_dino_backbone_training import independent_b14,independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==69;m=json.loads((ROOT/'manifest.json').read_text())
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 packages=[];anchors=0;teacher_count=0
 for clip in sorted(m['train'],key=lambda c:str(c['video_id'])):
  path=work.fp(clip['video_id'])
  with np.load(path)as a:record={k:a[k].copy()for k in a.files}
  with np.load(work.rgb.feature_path('native_sam','train',clip['video_id']))as a:
   for k in ('labels','xy','t_tgt','t_cam','video_id'):assert np.array_equal(record[k],a[k])
  mask=record['xy'][:,2]==0
  with np.load(work.prior.fp('train',clip['video_id']))as a:
   assert np.array_equal(record['labels'][mask],a['labels'])and np.array_equal(record['xy'][mask],a['xy'])and np.array_equal(record['decoded'][mask],a['decoded'])
  assert int(mask.sum())==256 and record['decoded'].shape==(len(record['labels']),768)and record['decoded'].dtype==np.float16 and np.isfinite(record['decoded']).all();assert file_hash(path)==s['feature_hashes'][str(path)];anchors+=int(mask.sum());teacher_count+=len(record['labels']);packages.append(record)
 assert anchors==20480 and teacher_count==162560
 first=m['train'][0];record=next(a for a in packages if int(a['video_id'])==first['video_id']);model=work.load_b14();features=independent_b14(model,first['frames'],record['xy']);mask=record['xy'][:,2]==0;prefix=independent_b14(model,first['frames'][:1],record['xy'][mask]);features[mask]=prefix;assert np.array_equal(features,record['decoded']);del model;torch.cuda.empty_cache()
 labels=np.concatenate([a['labels']for a in packages]);times=np.concatenate([a['xy'][:,2]for a in packages]);y=torch.tensor(labels,dtype=torch.long,device='cuda');t=torch.tensor(times,dtype=torch.long,device='cuda');fg=torch.where(y>0)[0];bg=torch.where(y==0)[0];replays={}
 for seed in range(3):
  gen=torch.Generator(device='cuda').manual_seed(seed);hist=np.zeros((8,26),np.int64);ih=hashlib.sha256();seen=np.zeros(len(labels),bool)
  for step in range(2000):
   f=fg[torch.randint(len(fg),(128,),generator=gen,device='cuda')];b=bg[torch.randint(len(bg),(128,),generator=gen,device='cuda')];ix=torch.cat([f,b]).cpu().numpy();ih.update(ix.tobytes());seen[ix]=True;np.add.at(hist,(times[ix],labels[ix]),1)
  r=json.loads(work.ckpt('B14_sam',seed).with_suffix('.json').read_text());assert r==s['training_reports'][seed]and r['protocol']==p
  assert ih.hexdigest()==r['sampled_indices_sha256']and hist.tolist()==r['sampled_time_class_counts']and hist.sum(0).tolist()==r['sampled_class_counts']and int(hist[0].sum())==r['GT_anchor_exposures']and int(seen.sum())==r['unique_sampled_queries']and int(seen[times==0].sum())==r['unique_sampled_GT_anchor_queries']
  old=json.loads(work.ckpt('S14_sam',seed).with_suffix('.json').read_text());assert old['sampled_class_counts']==r['sampled_class_counts']and int(hist.sum())==512000 and int(hist[:,0].sum())==256000
  assert [a['step']for a in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda a:a['val_macro_f1']);assert best['step']==r['best_step']and best['val_macro_f1']==r['best_val'];checkpoint=torch.load(work.ckpt('B14_sam',seed),map_location='cpu',weights_only=True);assert checkpoint['best_step']==best['step']and checkpoint['protocol']==p and checkpoint['steps']==2000 and checkpoint['seed']==seed and sum(a.numel()for a in checkpoint['model'].values())==r['parameter_count']==19994 and file_hash(work.ckpt('B14_sam',seed))==r['checkpoint_sha256'];replays[str(seed)]=ih.hexdigest();print('verified B14 SAM sampling',seed,int(hist[0].sum()),flush=True)
 frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];offsets=dict(p['fixed_offsets']);results={}
 for arm in work.ARMS:
  val=work.rgb.read('native_sam','val',m)if arm.startswith('S14')else work.prior.read('val',m);target=np.concatenate([a['labels']for a in val]);ts=np.concatenate([a['xy'][:,2]for a in val]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in val]);outputs=[]
  for seed in range(3):
   path=work.ckpt(arm,seed);assert file_hash(path)==frozen['checkpoint_sha256'][f'{arm}_{seed}'];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in val]);outputs.append(values)
   with np.load(ROOT/arm/'logits'/f'seed{seed}_val.npz')as a:assert np.array_equal(values,a['logits'])and np.array_equal(target,a['labels'])and np.array_equal(ts,a['times'])and np.array_equal(ids,a['video_ids'])and str(a['checkpoint_sha256'])==file_hash(path)
   if arm=='B14_sam':assert abs(independent_eval(values,target,ts,0)['macro_f1_present']-s['training_reports'][seed]['best_val'])<1e-12
  if arm=='B14_sam':
   best=-1;choice=None;curves=[]
   for offset in sorted(p['calibration_grid'],key=lambda x:(abs(x),x)):
    vals=[independent_eval(a,target,ts,offset)['macro_f1_present']for a in outputs];value=float(np.mean(vals));curves.append({'offset':float(offset),'macro_f1_by_seed':vals,'mean_macro_f1':value})
    if value>best:best=value;choice=float(offset)
   equal(curves,frozen['curves'][arm]);offsets[arm]=choice
  runs=[{'seed':seed,'raw_validation':independent_eval(a,target,ts,0),'calibrated_validation':independent_eval(a,target,ts,offsets[arm])}for seed,a in enumerate(outputs)];r={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([a['calibrated_validation']['macro_f1_present']for a in runs]))};equal(r,s['results'][arm]);results[arm]=r;print('verified B14 SAM factorial val',arm,flush=True)
 terms={'SAM_B14':[('B14_sam',1),('B14_gt',-1)],'SAM_S14':[('S14_sam',1),('S14_gt',-1)],'interaction':[('B14_sam',1),('B14_gt',-1),('S14_sam',-1),('S14_gt',1)]}
 for name,term in terms.items():
  vals=[sum(weight*results[a]['runs'][seed]['calibrated_validation']['macro_f1_present']for a,weight in term)for seed in range(3)];equal({'per_seed_validation_macro_difference':vals,'mean':float(np.mean(vals))},s['validation_contrasts'][name])
 selected=max(work.ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro']);assert offsets==frozen['offsets']and selected==s['within_factorial_validation_choice']==frozen['within_factorial_validation_choice'];assert frozen['new_SAM_exceeds_current_global_validation_candidate']==(results['B14_sam']['mean_calibrated_validation_macro']>p['current_global_validation_macro'])
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all80_teacher_metadata_and_GT_anchors_recomputed':True,'full8frame_and_single_anchor_feature_video':first['video_id'],'new_three_sampling_hashes':replays,'all_three_best_checkpoint_rules_and12_validation_outputs_recomputed':True,'calibration_interaction_selection_recomputed':True,'all_frozen_inputs_unchanged':True,'no_new_confirmation':True});print('DONE independent B14 SAM factorial verification',flush=True)
if __name__=='__main__':main()
