"""Independent two-affine inference, initialization and matched sampling checks."""
import hashlib,json
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_nonlinear_head as work
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def two_layer(x,state,relu):
 with torch.inference_mode():
  h=F.linear(torch.from_numpy(x).cuda(),state['input.weight'].cuda(),state['input.bias'].cuda())
  if relu:h=torch.maximum(h,torch.zeros_like(h))
  return F.linear(h,state['output.weight'].cuda(),state['output.bias'].cuda()).cpu().numpy()

def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==91 and p['new_heads']==6
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert file_hash(ROOT/'unit_checks.log')==p['unit_checks_sha256']and 'PASS'in(ROOT/'unit_checks.log').read_text()
 assert file_hash(ROOT/'future_confirmation_manifest.json')==p['future_confirmation_manifest_sha256']==file_hash(work.candidates.ROOT/'future_confirmation_manifest.json')
 m=json.loads((ROOT/'manifest.json').read_text());stats={'b14_gt32':p['statistics']};data={split:work.prior.read(work.REFERENCE,split,m,stats)for split in ('train',)+work.SPLITS};labels=np.concatenate([a['labels']for a in data['train']]);assert len(labels)==20480 and all((a['xy'][:,2]==0).all()for a in data['train']);assert file_hash(p['statistics']['path'])==p['statistics']['sha256']
 foreground=torch.tensor(np.flatnonzero(labels>0),device='cuda');background=torch.tensor(np.flatnonzero(labels==0),device='cuda');hashes={}
 for seed in range(3):
  torch.manual_seed(seed);first=torch.nn.Linear(768,32);last=torch.nn.Linear(32,26);initial={'input.weight':first.weight.detach(),'input.bias':first.bias.detach(),'output.weight':last.weight.detach(),'output.bias':last.bias.detach()};gen=torch.Generator(device='cuda').manual_seed(seed);digest=hashlib.sha256();hist=np.zeros(26,np.int64)
  for _ in range(2000):
   f=foreground[torch.randint(len(foreground),(16,),generator=gen,device='cuda')];b=background[torch.randint(len(background),(16,),generator=gen,device='cuda')];ix=torch.cat([f,b]).cpu().numpy();digest.update(ix.tobytes());hist+=np.bincount(labels[ix],minlength=26)
  hashes[str(seed)]=digest.hexdigest();reference=json.loads(work.ckpt(work.REFERENCE,seed).with_suffix('.json').read_text());assert digest.hexdigest()==reference['sampled_indices_sha256']and hist.tolist()==reference['sampled_class_counts']and hist.sum()==64000
  for arm in work.NEW:
   path=work.ckpt(arm,seed);r=json.loads(path.with_suffix('.json').read_text());assert r==s['training_reports'][arm][seed]and r['protocol']==p and r['sampled_indices_sha256']==digest.hexdigest()and r['sampled_class_counts']==hist.tolist()
   assert[row['step']for row in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda a:a['balanced_raw_val_macro']);assert best['step']==r['best_step']and best['balanced_raw_val_macro']==r['best_balanced_raw_val']
   cp=torch.load(path,map_location='cpu',weights_only=True);assert cp['protocol']==p and cp['steps']==2000 and cp['seed']==seed and cp['best_step']==r['best_step']and cp['nonlinear']==(arm=='relu32')and sum(v.numel()for v in cp['model'].values())==r['head_parameters']==25466 and file_hash(path)==r['checkpoint_sha256']
   for key,value in initial.items():assert torch.equal(cp['initial_state'][key],value)and r['initial_parameter_sha256'][key]==hashlib.sha256(value.numpy().tobytes()).hexdigest()
  assert s['training_reports']['factor32'][seed]['initial_parameter_sha256']==s['training_reports']['relu32'][seed]['initial_parameter_sha256']
 frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];results={};fold_checks=[]
 for arm in work.ARMS:
  outputs={}
  for split in work.SPLITS:
   y=np.concatenate([a['labels']for a in data[split]]);times=np.concatenate([a['xy'][:,2]for a in data[split]]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in data[split]]);outputs[split]=[]
   for seed in range(3):
    path=work.ckpt(arm,seed);assert file_hash(path)==frozen['checkpoint_sha256'][f'{arm}_{seed}'];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)if arm==work.REFERENCE else two_layer(a['decoded'],state,arm=='relu32')for a in data[split]])
    with np.load(ROOT/arm/'logits'/f'seed{seed}_{split}.npz')as a:
     for key,value in [('logits',values),('labels',y),('times',times),('video_ids',ids)]:assert np.array_equal(a[key],value)
     assert str(a['checkpoint_sha256'])==file_hash(path)
    outputs[split].append({'logits':values,'labels':y,'times':times})
    if arm in work.NEW and split=='balanced_val':assert abs(independent_eval(values,y,times,0)['macro_f1_present']-s['training_reports'][arm][seed]['best_balanced_raw_val'])<1e-12
    if arm=='factor32'and split=='balanced_val':
     a=state['input.weight'].cuda();b=state['input.bias'].cuda();w=state['output.weight'].cuda();v=state['output.bias'].cuda();fold_w=w@a;fold_b=w@b+v
     with torch.inference_mode():folded=np.concatenate([F.linear(torch.from_numpy(row['decoded']).cuda(),fold_w,fold_b).cpu().numpy()for row in data[split]])
     maximum=float(np.abs(folded-values).max());tolerance=2e-4+2e-5*float(np.abs(values).max());record={'maximum_folded_FP32_logit_difference':maximum,'preregistered_tolerance':tolerance,'label_changes_due_to_numerics':int(np.sum(folded.argmax(-1)!=values.argmax(-1)))};assert maximum<=tolerance;equal(record,s['factorized_affine_equivalence'][seed])
     x=data[split][0]['decoded'][:64].astype(np.float64);aa=state['input.weight'].numpy().astype(np.float64);bb=state['input.bias'].numpy().astype(np.float64);ww=state['output.weight'].numpy().astype(np.float64);vv=state['output.bias'].numpy().astype(np.float64);direct=(x@aa.T+bb)@ww.T+vv;equivalent=x@(ww@aa).T+(ww@bb+vv);error=float(np.abs(direct-equivalent).max());assert error<1e-9;fold_checks.append({'seed':seed,'FP32':record,'FP64_first64_maximum_error':error})
  offsets={'raw':0.}
  for split,profile in [('balanced_val','balanced_profile'),('spatial_val','spatial_profile')]:
   if arm==work.REFERENCE:offsets[profile]=p['fixed_reference_offsets'][profile];continue
   best=-1;choice=None;curve=[]
   for offset in sorted(p['calibration_grid'],key=lambda v:(abs(v),v)):
    per=[independent_eval(a['logits'],a['labels'],a['times'],offset)['macro_f1_present']for a in outputs[split]];mean=float(np.mean(per));curve.append({'offset':float(offset),'macro_f1_by_seed':per,'mean_macro_f1':mean})
    if mean>best:best=mean;choice=float(offset)
   equal(curve,frozen['curves'][arm][profile]);offsets[profile]=choice
  runs=[{'seed':seed,**{split:{profile:independent_eval(outputs[split][seed]['logits'],outputs[split][seed]['labels'],outputs[split][seed]['times'],offset)for profile,offset in offsets.items()}for split in work.SPLITS}}for seed in range(3)];results[arm]={'offsets':offsets,'runs':runs,'mean_balanced_profile_macro':float(np.mean([a['balanced_val']['balanced_profile']['macro_f1_present']for a in runs]))};equal(results[arm],s['results'][arm]);print('verified matched head output/profile',arm,flush=True)
 pairs=[('relu32','factor32'),('factor32',work.REFERENCE),('relu32',work.REFERENCE)];contrasts={}
 for left,right in pairs:
  contrasts[left+'_minus_'+right]={}
  for split in work.SPLITS:
   contrasts[left+'_minus_'+right][split]={}
   for profile in ('raw','balanced_profile','spatial_profile'):
    diff=[results[left]['runs'][seed][split][profile]['macro_f1_present']-results[right]['runs'][seed][split][profile]['macro_f1_present']for seed in range(3)];contrasts[left+'_minus_'+right][split][profile]={'per_seed_macro_difference':diff,'mean':float(np.mean(diff))}
 equal(contrasts,s['contrasts']);selected=max(work.ARMS,key=lambda a:results[a]['mean_balanced_profile_macro']);eligible=selected in work.NEW and min(contrasts[selected+'_minus_'+work.REFERENCE]['balanced_val']['balanced_profile']['per_seed_macro_difference'])>0
 assert selected==s['validation_selected_arm']==frozen['validation_selected_arm']and eligible==s['new_arm_eligible_for_separate_confirmation']==frozen['new_arm_eligible_for_separate_confirmation']and frozen['future_confirmation_manifest_sha256']==p['future_confirmation_manifest_sha256']and frozen['offsets']=={a:r['offsets']for a,r in results.items()}and s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt'
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'six_head_initializations_independently_reconstructed_and_matched':True,'three_complete_sampler_replays':hashes,'six_matched25466_parameter_budgets_and_checkpoints_verified':True,'all18_raw_logits_four_new_shared_calibration_curves_metrics_contrasts_selection_recomputed':True,'factorized_affine_checks':fold_checks,'unchanged_training_statistics_future_candidate_identity_and_old_inputs':True,'no_new_confirmation_or_deployment_change':True});print('DONE independent matched linear versus ReLU source control',selected,eligible,flush=True)
if __name__=='__main__':main()
