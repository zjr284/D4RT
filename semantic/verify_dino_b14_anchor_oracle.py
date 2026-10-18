"""Independent raw-GT, exact sampler/loss statistics and fifteen-val-output audit."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from semantic import dino_b14_anchor_oracle as work
from semantic.ytvis import load_dataset
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==70
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 m=json.loads((ROOT/'manifest.json').read_text());source=work.prior.read_train(m);ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'));clips={c['video_id']:c for c in m['train']};truths=[];changed=0
 for a in source:
  c=clips[int(a['video_id'])]
  with np.load(work.oracle.ROOT/'labels'/f'{int(a["video_id"])}.npz')as o:
   assert np.array_equal(o['xy'],a['xy'])and np.array_equal(o['original_labels'],a['labels']);truth=o['corrected_labels'].copy()
  maps=np.stack([ds.gt_label_map(c,t)for t in range(8)]);xy=a['xy'];actual=maps[xy[:,2],xy[:,1],xy[:,0]].astype(np.int16);assert np.array_equal(actual,truth)and np.array_equal(actual[xy[:,2]==0],a['labels'][xy[:,2]==0]);truths.append(truth);changed+=int((truth!=a['labels']).sum());print('verified source oracle GT',int(a['video_id']),flush=True)
 labels=np.concatenate([a['labels']for a in source]);times=np.concatenate([a['xy'][:,2]for a in source]);truth=np.concatenate(truths);assert len(labels)==len(truth)==162560 and int((times==0).sum())==20480
 y=torch.tensor(labels,dtype=torch.long,device='cuda');fg=torch.where(y>0)[0];bg=torch.where(y==0)[0];replays={}
 for seed in range(3):
  gen=torch.Generator(device='cuda').manual_seed(seed);ih=hashlib.sha256();hist=np.zeros((8,26),np.int64);teacher=np.zeros(26,np.int64);truehist=np.zeros(26,np.int64);sourcehist=np.zeros(26,np.int64);nh=np.zeros(257,np.int64)
  for step in range(2000):
   f=fg[torch.randint(len(fg),(128,),generator=gen,device='cuda')];b=bg[torch.randint(len(bg),(128,),generator=gen,device='cuda')];ix=torch.cat([f,b]).cpu().numpy();ih.update(ix.tobytes());np.add.at(hist,(times[ix],labels[ix]),1);teacher+=np.bincount(labels[ix],minlength=26);truehist+=np.bincount(truth[ix],minlength=26);active=times[ix]==0;assert active.any();sourcehist+=np.bincount(labels[ix][active],minlength=26);nh[int(active.sum())]+=1
  reference=json.loads(work.ckpt('weak',seed).with_suffix('.json').read_text());assert ih.hexdigest()==reference['sampled_indices_sha256']and hist.tolist()==reference['sampled_time_class_counts'];replays[str(seed)]=ih.hexdigest()
  for arm in work.NEW:
   r=json.loads(work.ckpt(arm,seed).with_suffix('.json').read_text());assert r==s['training_reports'][arm][seed]and r['protocol']==p and r['sampled_indices_sha256']==ih.hexdigest()and r['teacher_time_class_counts']==hist.tolist()
   expected_all=truehist if arm=='oracle'else teacher;expected_nonzero=truehist if arm=='oracle'else sourcehist;dh=nh.copy()if arm=='source_active'else np.eye(1,257,256,dtype=np.int64).ravel()*2000
   assert r['all_sampled_loss_target_counts']==expected_all.tolist()and r['nonzero_loss_target_counts']==expected_nonzero.tolist()and r['nonzero_loss_exposures']==int(expected_nonzero.sum())and r['active_anchor_count_histogram']==nh.tolist()and r['loss_denominator_histogram']==dh.tolist()and r['GT_anchor_exposures']==int(np.arange(257)@nh)
   assert [h['step']for h in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda h:h['val_macro_f1']);assert r['best_step']==best['step']and r['best_val']==best['val_macro_f1'];checkpoint=torch.load(work.ckpt(arm,seed),map_location='cpu',weights_only=True);assert checkpoint['best_step']==best['step']and checkpoint['protocol']==p and checkpoint['steps']==2000 and checkpoint['seed']==seed and sum(a.numel()for a in checkpoint['model'].values())==r['parameter_count']==19994 and file_hash(work.ckpt(arm,seed))==r['checkpoint_sha256']
  print('verified anchor/oracle shared sampler and3 targets',seed,flush=True)
 val=work.prior.prior.read('val',m);target=np.concatenate([a['labels']for a in val]);ts=np.concatenate([a['xy'][:,2]for a in val]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in val]);frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];offsets=dict(p['fixed_offsets']);results={}
 for arm in work.ARMS:
  outputs=[]
  for seed in range(3):
   path=work.ckpt(arm,seed);assert file_hash(path)==frozen['checkpoint_sha256'][f'{arm}_{seed}'];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in val]);outputs.append(values)
   with np.load(ROOT/arm/'logits'/f'seed{seed}_val.npz')as a:assert np.array_equal(a['logits'],values)and np.array_equal(a['labels'],target)and np.array_equal(a['times'],ts)and np.array_equal(a['video_ids'],ids)and str(a['checkpoint_sha256'])==file_hash(path)
   if arm in work.NEW:assert abs(independent_eval(values,target,ts,0)['macro_f1_present']-s['training_reports'][arm][seed]['best_val'])<1e-12
  if arm in work.NEW:
   curves=[];best=-1;choice=None
   for offset in sorted(p['calibration_grid'],key=lambda x:(abs(x),x)):
    vals=[independent_eval(a,target,ts,offset)['macro_f1_present']for a in outputs];value=float(np.mean(vals));curves.append({'offset':float(offset),'macro_f1_by_seed':vals,'mean_macro_f1':value})
    if value>best:best=value;choice=float(offset)
   equal(curves,frozen['curves'][arm]);offsets[arm]=choice
  runs=[{'seed':seed,'raw_validation':independent_eval(a,target,ts,0),'calibrated_validation':independent_eval(a,target,ts,offsets[arm])}for seed,a in enumerate(outputs)];r={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([a['calibrated_validation']['macro_f1_present']for a in runs]))};equal(r,s['results'][arm]);results[arm]=r;print('verified anchor/oracle val',arm,flush=True)
 for left,right in [('source_active','weak'),('source_total','source_active'),('oracle','weak')]:
  vals=[results[left]['runs'][seed]['calibrated_validation']['macro_f1_present']-results[right]['runs'][seed]['calibrated_validation']['macro_f1_present']for seed in range(3)];equal({'per_seed_validation_macro_difference':vals,'mean':float(np.mean(vals))},s['validation_contrasts'][left+'_minus_'+right])
 assert tuple(p['eligible_validation_candidates'])==work.ELIGIBLE and 'oracle'not in work.ELIGIBLE and frozen['oracle_excluded_from_selection']and not s['oracle_deployable'];selected=max(work.ELIGIBLE,key=lambda a:results[a]['mean_calibrated_validation_macro']);assert selected==s['eligible_validation_choice']==frozen['validation_selected_arm']and offsets==frozen['offsets'];assert frozen['exceeds_previous_global_validation']==(results[selected]['mean_calibrated_validation_macro']>p['current_global_validation_macro'])
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all80_raw_GT_and_oracle_query_targets_recomputed':True,'corrected_teacher_targets':changed,'three_shared_sampler_hashes_for9_heads':replays,'all_masked_target_denominator_and_anchor_histograms_recomputed':True,'all9_best_checkpoint_rules_and15_validation_logits_recomputed':True,'all_calibrations_contrasts_and_nonoracle_selection_recomputed':True,'all_frozen_inputs_unchanged':True,'no_new_confirmation_or_oracle_deployment':True});print('DONE independent B14 anchor/oracle verification',selected,flush=True)
if __name__=='__main__':main()
