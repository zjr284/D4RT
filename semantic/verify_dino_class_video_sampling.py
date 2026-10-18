"""Independent B14 sampling replay and nine-output validation reconstruction."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from semantic import dino_class_video_sampling as work
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.verify_dino_backbone_training import independent_eval
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==61
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 m=json.loads((ROOT/'manifest.json').read_text());source=work.prior.read('train',m);val=work.prior.read('val',m);y=np.concatenate([a['labels']for a in source]);vid=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int64)for a in source]);unique=np.unique(vid);classes=np.unique(y[y>0]);assert len(y)==20480 and len(classes)==25
 pool=[]
 for a in source:
  assert np.all(a['xy'][:,2]==0)and int((a['labels']>0).sum())==int((a['labels']==0).sum())==128
  for cid in np.unique(a['labels']):
   rows=a['labels']==cid;pool.append({'video_id':int(a['video_id']),'class_id':int(cid),'queries':int(rows.sum()),'distinct_xy':len(np.unique(a['xy'][rows,:2],axis=0))})
 assert pool==json.loads((ROOT/'source_pool_audit.json').read_text())and file_hash(ROOT/'source_pool_audit.json')==s['source_pool_audit_sha256']
 fg=torch.from_numpy(np.flatnonzero(y>0)).cuda();bg=torch.from_numpy(np.flatnonzero(y==0)).cuda();conditional={int(c):torch.from_numpy(np.flatnonzero(y==c)).cuda()for c in classes};pv={int(c):[(int(v),torch.from_numpy(np.flatnonzero((y==c)&(vid==v))).cuda())for v in np.unique(vid[y==c])]for c in classes};replayed={}
 for arm in work.ARMS[1:]:
  for seed in range(3):
   base=torch.Generator(device='cuda').manual_seed(seed);cls=torch.Generator(device='cuda').manual_seed(20261061+seed);detail=torch.Generator(device='cuda').manual_seed(20262061+seed)
   hashes={k:hashlib.sha256()for k in ['actual','base','bg','class']};hist=np.zeros(26,np.int64);fgcounts=np.zeros((80,26),np.int64);bgcounts=np.zeros(80,np.int64)
   for step in range(2000):
    bf=fg[torch.randint(len(fg),(16,),generator=base,device='cuda')];bb=bg[torch.randint(len(bg),(16,),generator=base,device='cuda')];choices=torch.randint(25,(16,),generator=cls,device='cuda');actual_fg=torch.empty(16,dtype=torch.long,device='cuda')
    for index,cid in enumerate(classes):
     destinations=torch.where(choices==index)[0]
     if not len(destinations):continue
     if arm=='class':
      candidates=conditional[int(cid)];draw=torch.randint(len(candidates),(len(destinations),),generator=detail,device='cuda');actual_fg[destinations]=candidates[draw]
     else:
      videos=pv[int(cid)];draw=torch.randint(len(videos),(len(destinations),),generator=detail,device='cuda')
      for j,(_,candidates)in enumerate(videos):
       dest=destinations[draw==j]
       if len(dest):actual_fg[dest]=candidates[torch.randint(len(candidates),(len(dest),),generator=detail,device='cuda')]
    actual=torch.cat([actual_fg,bb]).cpu().numpy();baseidx=torch.cat([bf,bb]).cpu().numpy();bgidx=bb.cpu().numpy();choice=choices.cpu().numpy()
    assert np.array_equal(y[actual[:16]],classes[choice])and np.all(y[bgidx]==0)
    for key,array in [('actual',actual),('base',baseidx),('bg',bgidx),('class',choice)]:hashes[key].update(array.tobytes())
    hist+=np.bincount(y[actual],minlength=26);np.add.at(fgcounts,(np.searchsorted(unique,vid[actual[:16]]),y[actual[:16]]),1);np.add.at(bgcounts,np.searchsorted(unique,vid[bgidx]),1)
   r=json.loads(work.ckpt(arm,seed).with_suffix('.json').read_text());assert r==s['training_reports'][arm][seed]and r['protocol']==p
   assert (hashes['actual'].hexdigest(),hashes['base'].hexdigest(),hashes['bg'].hexdigest(),hashes['class'].hexdigest())==(r['sampled_indices_sha256'],r['base_indices_sha256'],r['BG_indices_sha256'],r['class_choices_sha256'])
   assert hist.tolist()==r['sampled_class_counts']and fgcounts.tolist()==r['FG_video_class_counts']and bgcounts.tolist()==r['BG_video_counts']and unique.tolist()==r['video_ids']
   old=json.loads(work.ckpt('query',seed).with_suffix('.json').read_text());assert hashes['base'].hexdigest()==old['sampled_indices_sha256']and hist[0]==32000 and hist.sum()==64000
   assert [a['step']for a in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda a:a['val_macro_f1']);assert best['step']==r['best_step']and best['val_macro_f1']==r['best_val']
   checkpoint=torch.load(work.ckpt(arm,seed),map_location='cpu',weights_only=True);assert checkpoint['protocol']==p and checkpoint['best_step']==best['step']and checkpoint['seed']==seed and checkpoint['steps']==2000
   assert sum(t.numel()for t in checkpoint['model'].values())==r['parameter_count']==19994 and file_hash(work.ckpt(arm,seed))==r['checkpoint_sha256']
   replayed[f'{arm}_{seed}']={k:h.hexdigest()for k,h in hashes.items()};print('verified balanced sampler',arm,seed,flush=True)
 for seed in range(3):
  a=s['training_reports']['class'][seed];b=s['training_reports']['class_video'][seed]
  for key in ('sampled_class_counts','base_indices_sha256','BG_indices_sha256','class_choices_sha256','BG_video_counts'):assert a[key]==b[key]
 frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];target=np.concatenate([a['labels']for a in val]);times=np.concatenate([a['xy'][:,2]for a in val]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in val]);assert len(target)==40960
 results={};offsets={'query':-2.0}
 for arm in work.ARMS:
  outputs=[]
  for seed in range(3):
   state=torch.load(work.ckpt(arm,seed),map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in val]);outputs.append(values)
   with np.load(ROOT/arm/'logits'/f'dinov2_{seed}_val.npz')as a:
    assert np.array_equal(a['logits'],values)and np.array_equal(a['labels'],target)and np.array_equal(a['times'],times)and np.array_equal(a['video_ids'],ids)and str(a['checkpoint_sha256'])==file_hash(work.ckpt(arm,seed))
   if arm!='query':assert abs(independent_eval(values,target,times,0)['macro_f1_present']-s['training_reports'][arm][seed]['best_val'])<1e-12
   assert frozen['checkpoint_sha256'][f'{arm}_{seed}']==file_hash(work.ckpt(arm,seed))
  if arm!='query':
   curves=[];best=-1;choice=None
   for offset in sorted(p['calibration_grid'],key=lambda x:(abs(x),x)):
    vals=[independent_eval(v,target,times,offset)['macro_f1_present']for v in outputs];value=float(np.mean(vals));curves.append({'offset':float(offset),'macro_f1_by_seed':vals,'mean_macro_f1':value})
    if value>best:best=value;choice=float(offset)
   equal(curves,frozen['curves'][arm]);offsets[arm]=choice
  runs=[{'seed':i,'raw_validation':independent_eval(v,target,times,0),'calibrated_validation':independent_eval(v,target,times,offsets[arm])}for i,v in enumerate(outputs)];result={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([r['calibrated_validation']['macro_f1_present']for r in runs]))};equal(result,s['results'][arm]);results[arm]=result
  print('verified balanced three heads/calibration',arm,flush=True)
 selected=max(work.ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro']);assert offsets==frozen['offsets']and selected==s['validation_selected_arm']==frozen['validation_selected_arm']
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt'
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'source_pool_records_recomputed':len(pool),'six_sampler_hash_replays':replayed,'class_sequences_BG_and_baseline_indices_shared':True,'all_six_best_checkpoint_rules_and_nine_logits_calibrations_recomputed':True,'frozen_selection_recomputed':selected,'all_old_inputs_unchanged':True,'new_confirmation_groups':0})
 print('DONE independent class/video sampling verification',selected,flush=True)
if __name__=='__main__':main()
