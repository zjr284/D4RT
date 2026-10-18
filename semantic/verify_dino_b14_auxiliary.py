"""Independent two-generator, fixed-anchor and fifteen-output R71 audit."""
import hashlib,json
import numpy as np
import torch
from semantic import dino_b14_auxiliary as work
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==71
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 m=json.loads((ROOT/'manifest.json').read_text());base=work.prior.prior.prior.read('train',m);whole=work.prior.read(m);later=[]
 for a,b in zip(whole,base):
  assert int(a['video_id'])==int(b['video_id']);keep=a['xy'][:,2]==0
  for key in ('decoded','labels','xy'):assert np.array_equal(a[key][keep],b[key])
  assert np.array_equal(a['truth'][keep],b['labels']);later.append({key:a[key][~keep]for key in ('decoded','labels','xy','truth')})
 assert sum(len(a['labels'])for a in base)==20480 and sum(len(a['labels'])for a in later)==142080
 by=np.concatenate([a['labels']for a in base]);sy=np.concatenate([a['labels']for a in later]);st=np.concatenate([a['xy'][:,2]for a in later]);truth=np.concatenate([a['truth']for a in later]);assert (st>0).all()
 def pools(labels):
  tensor=torch.tensor(labels,device='cuda',dtype=torch.long);return torch.where(tensor>0)[0],torch.where(tensor==0)[0]
 bp=pools(by);sp=pools(sy);replays={}
 for seed in range(3):
  reference=json.loads(work.ckpt('gt32',seed).with_suffix('.json').read_text());replays[str(seed)]={}
  for arm in work.NEW:
   bg=torch.Generator(device='cuda').manual_seed(seed);sg=torch.Generator(device='cuda').manual_seed(20261071+seed);bh=hashlib.sha256();sh=hashlib.sha256();basehist=np.zeros(26,np.int64);hist=np.zeros((8,26),np.int64);actual=np.zeros(26,np.int64)
   labels=by if arm=='extra_gt'else sy;times=np.zeros(len(by),np.int64)if arm=='extra_gt'else st;targets=truth if arm=='aux_oracle'else labels;pool=bp if arm=='extra_gt'else sp
   for step in range(2000):
    bi=torch.cat([b[torch.randint(len(b),(16,),generator=bg,device='cuda')]for b in bp]).cpu().numpy();si=torch.cat([a[torch.randint(len(a),(16,),generator=sg,device='cuda')]for a in pool]).cpu().numpy();bh.update(bi.tobytes());sh.update(si.tobytes());basehist+=np.bincount(by[bi],minlength=26);np.add.at(hist,(times[si],labels[si]),1);actual+=np.bincount(targets[si],minlength=26)
   r=json.loads(work.ckpt(arm,seed).with_suffix('.json').read_text());assert r==s['training_reports'][arm][seed]and r['protocol']==p
   assert bh.hexdigest()==reference['sampled_indices_sha256']==r['base_indices_sha256']and basehist.tolist()==reference['sampled_class_counts']==r['base_GT_class_counts'];assert sh.hexdigest()==r['supplement_indices_sha256']and hist.tolist()==r['supplement_sampling_time_class_counts']and actual.tolist()==r['supplement_actual_target_counts']
   assert r['supplement_pool']==('source_GT'if arm=='extra_gt'else'later_teacher')and r['base_exposures']==r['supplement_exposures']==64000 and r['total_exposures']==128000 and basehist.sum()==hist.sum()==actual.sum()==64000 and basehist[0]==hist[:,0].sum()==32000
   assert [h['step']for h in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda h:h['val_macro_f1']);assert r['best_step']==best['step']and r['best_val']==best['val_macro_f1'];checkpoint=torch.load(work.ckpt(arm,seed),map_location='cpu',weights_only=True);assert checkpoint['protocol']==p and checkpoint['steps']==2000 and checkpoint['seed']==seed and checkpoint['best_step']==best['step']and sum(v.numel()for v in checkpoint['model'].values())==r['parameter_count']==19994 and file_hash(work.ckpt(arm,seed))==r['checkpoint_sha256']
   replays[str(seed)][arm]={'base_indices_sha256':bh.hexdigest(),'supplement_indices_sha256':sh.hexdigest()}
  a=s['training_reports']['aux_sam'][seed];b=s['training_reports']['aux_oracle'][seed]
  for key in ('base_indices_sha256','supplement_indices_sha256','base_GT_class_counts','supplement_sampling_time_class_counts'):assert a[key]==b[key]
  print('verified fixed GT anchors and auxiliary samplers',seed,flush=True)
 val=work.prior.prior.prior.read('val',m);target=np.concatenate([a['labels']for a in val]);ts=np.concatenate([a['xy'][:,2]for a in val]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in val]);frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];offsets=dict(p['fixed_offsets']);results={}
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
  runs=[{'seed':seed,'raw_validation':independent_eval(a,target,ts,0),'calibrated_validation':independent_eval(a,target,ts,offsets[arm])}for seed,a in enumerate(outputs)];r={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([a['calibrated_validation']['macro_f1_present']for a in runs]))};equal(r,s['results'][arm]);results[arm]=r;print('verified auxiliary val',arm,flush=True)
 for left,right in [('aux_sam','extra_gt'),('aux_oracle','aux_sam'),('aux_sam','gt32')]:
  vals=[results[left]['runs'][seed]['calibrated_validation']['macro_f1_present']-results[right]['runs'][seed]['calibrated_validation']['macro_f1_present']for seed in range(3)];equal({'per_seed_validation_macro_difference':vals,'mean':float(np.mean(vals))},s['validation_contrasts'][left+'_minus_'+right])
 assert tuple(p['eligible_candidates'])==work.ELIGIBLE and 'aux_oracle'not in work.ELIGIBLE and frozen['oracle_excluded_from_selection']and not s['oracle_deployable'];selected=max(work.ELIGIBLE,key=lambda a:results[a]['mean_calibrated_validation_macro']);teacher=max(('weak','aux_sam'),key=lambda a:results[a]['mean_calibrated_validation_macro']);assert selected==s['eligible_validation_choice']==frozen['validation_selected_arm']and teacher==s['validation_selected_teacher_recipe']==frozen['validation_selected_teacher_recipe']and offsets==frozen['offsets'];assert frozen['teacher_matched_GT_reference']==('extra_gt'if teacher=='aux_sam'else'B14_GT256_from_round58');assert frozen['exceeds_previous_global_validation']==(results[selected]['mean_calibrated_validation_macro']>p['current_global_validation_macro'])
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'80_source_prefixes_and_later_only_pools_verified':True,'all9_two_generator_sampler_replays':replays,'all9_GT_anchor_hashes_match_old_GT32':True,'all_actual_target_and_exposure_histograms_recomputed':True,'all9_best_checkpoint_rules_and15_val_logits_recomputed':True,'all_calibrations_contrasts_and_nonoracle_teacher_selection_recomputed':True,'all_frozen_inputs_unchanged':True,'no_new_confirmation_or_oracle_deployment':True});print('DONE independent B14 auxiliary verification',selected,teacher,flush=True)
if __name__=='__main__':main()
