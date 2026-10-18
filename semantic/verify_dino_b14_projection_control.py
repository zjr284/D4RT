"""Independent sampling, FP64 projection prefixes, diagnostics and val audit."""
import hashlib,json
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_b14_projection_control as work
from semantic import dino_backbone_training as capacity
from semantic import dino_b14_anchor_oracle as oracle
from semantic.model import SemanticHead
from semantic.verify_dino_backbone_training import independent_eval
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def conflicting_update_check():
 param=torch.nn.Parameter(torch.tensor([.3,-.2],device='cuda'));optimizer=torch.optim.AdamW([param],lr=.001,weight_decay=.01);expected=np.array([.3,-.2],np.float64);m=np.zeros(2);v=np.zeros(2)
 for step,(b,t)in enumerate([([1.,0.],[-2.,1.]),([0.,1.],[2.,-3.]),([1.,1.],[-4.,0.])],1):
  base=np.array(b,np.float64);supp=np.array(t,np.float64);alpha=min(float(base@supp),0)/(float(base@base)+1e-12);g=base+supp-alpha*base;assert alpha<0
  projected,_=work.project(torch.tensor(b,device='cuda'),torch.tensor(t,device='cuda'));optimizer.zero_grad(set_to_none=True);param.grad=torch.tensor(b,device='cuda')+projected;optimizer.step();m=.9*m+.1*g;v=.999*v+.001*g*g;expected=expected*(1-.001*.01)-.001*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8);assert np.max(np.abs(param.detach().cpu().numpy()-expected))<1e-7

def main():
 torch.backends.cuda.matmul.allow_tf32=False;conflicting_update_check()
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==75 and p['head_parameters']==19994;assert file_hash(ROOT/'unit_checks.log')==p['unit_checks_sha256']and 'PASS'in(ROOT/'unit_checks.log').read_text()
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 m=json.loads((ROOT/'manifest.json').read_text());base=capacity.read('train',m);whole=oracle.read(m);later=[]
 for a,b in zip(whole,base):
  mask=a['xy'][:,2]==0;assert int(a['video_id'])==int(b['video_id'])
  for key in ('decoded','labels','xy'):assert np.array_equal(a[key][mask],b[key])
  later.append({key:a[key][~mask]for key in ('decoded','labels','xy')})
 bx=torch.from_numpy(np.concatenate([a['decoded']for a in base])).float().cuda();by=np.concatenate([a['labels']for a in base]);ly=np.concatenate([a['labels']for a in later]);lt=np.concatenate([a['xy'][:,2]for a in later]);lx=torch.from_numpy(np.concatenate([a['decoded']for a in later])).float().cuda();base_y=torch.tensor(by,dtype=torch.long,device='cuda');pool_b=(torch.where(base_y>0)[0],torch.where(base_y==0)[0]);prefix_max=0.;replays={}
 for arm in work.NEW:
  sy,st,x=(by,np.zeros(len(by),np.int64),bx)if arm=='project_extra_gt'else(ly,lt,lx);target=torch.tensor(sy,dtype=torch.long,device='cuda');pool_s=(torch.where(target>0)[0],torch.where(target==0)[0])
  for seed in range(3):
   bg=torch.Generator(device='cuda').manual_seed(seed);sg=torch.Generator(device='cuda').manual_seed(20261071+seed);bh=hashlib.sha256();sh=hashlib.sha256();hist=np.zeros(26,np.int64);supphist=np.zeros((8,26),np.int64);prefix=[]
   for step in range(2000):
    bi=torch.cat([q[torch.randint(len(q),(16,),generator=bg,device='cuda')]for q in pool_b]);si=torch.cat([q[torch.randint(len(q),(16,),generator=sg,device='cuda')]for q in pool_s]);ib,is_=bi.cpu().numpy(),si.cpu().numpy();bh.update(ib.tobytes());sh.update(is_.tobytes());hist+=np.bincount(by[ib],minlength=26);np.add.at(supphist,(st[is_],sy[is_]),1)
    if step<3:prefix.append((bi,si))
   path=work.ckpt(arm,seed);r=json.loads(path.with_suffix('.json').read_text());assert r==s['training_reports'][arm][seed]and r['protocol']==p and r['base_indices_sha256']==bh.hexdigest()and r['supplement_indices_sha256']==sh.hexdigest()and r['base_GT_class_counts']==hist.tolist()and r['supplement_time_class_counts']==supphist.tolist()and hist.sum()==supphist.sum()==64000
   reference=json.loads(work.prior.ckpt('extra_gt'if arm=='project_extra_gt'else'aux_sam',seed).with_suffix('.json').read_text());assert bh.hexdigest()==reference['base_indices_sha256']and sh.hexdigest()==reference['supplement_indices_sha256'];assert file_hash(path.with_suffix('.diagnostics.npz'))==r['diagnostic_sha256']and file_hash(path.with_suffix('.prefix.pt'))==r['prefix_sha256']
   with np.load(path.with_suffix('.diagnostics.npz'))as a:d=a['values'].copy()
   assert (d[:3,4]==0).all();assert d.shape==(2000,8)and np.isfinite(d).all()and np.array_equal(d[:,0],np.arange(1,2001))and (d[:,[2,3,6]]>=0).all();expected=np.minimum(d[:,1],0)/(d[:,2]+1e-12);assert np.allclose(expected,d[:,4],rtol=2e-5,atol=2e-6);scale=np.sqrt(d[:,2]*d[:,6]).clip(1e-12);assert (d[:,5]>=-2e-5*scale).all();assert np.all(np.abs(d[:,5]-(d[:,1]-d[:,4]*d[:,2]))<=2e-5*np.sqrt(d[:,2]*d[:,3])+2e-6);assert np.allclose(d[:,6],d[:,3]-2*d[:,4]*d[:,1]+d[:,4]**2*d[:,2],rtol=2e-5,atol=2e-6)
   assert r['projection_applied_steps']==int((d[:,1]<0).sum())and r['post_projection_negative_dot_beyond_tolerance_steps']==0 and r['actual_AdamW_descent_dot_base_negative_steps']==int((d[:,7]<0).sum())
   snapshots=torch.load(path.with_suffix('.prefix.pt'),map_location='cpu',weights_only=True);assert len(snapshots)==4;torch.manual_seed(seed);model=SemanticHead(hidden_dim=768,num_classes=26).cuda();params=list(model.parameters());assert all(torch.equal(v.cpu(),snapshots[0][k])for k,v in model.state_dict().items());optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
   for step,(bi,si)in enumerate(prefix):
    optimizer.zero_grad(set_to_none=True);lb=F.cross_entropy(model(bx[bi]),base_y[bi]);ls=F.cross_entropy(model(x[si]),target[si]);gb=np.concatenate([q.detach().cpu().numpy().reshape(-1)for q in torch.autograd.grad(lb,params)]).astype(np.float64);gs=np.concatenate([q.detach().cpu().numpy().reshape(-1)for q in torch.autograd.grad(ls,params)]).astype(np.float64);alpha=min(float(gb@gs),0)/(float(gb@gb)+1e-12);combined=(gb+gs-alpha*gb).astype(np.float32);start=0
    for q in params:q.grad=torch.from_numpy(combined[start:start+q.numel()].copy()).cuda().reshape_as(q);start+=q.numel()
    optimizer.step();error=max(float((v.detach().cpu()-snapshots[step+1][k]).abs().max())for k,v in model.state_dict().items());prefix_max=max(prefix_max,error);assert error<=2e-6
   assert [a['step']for a in r['history']]==list(range(50,2001,50));best=max(r['history'],key=lambda a:a['val_macro_f1']);assert r['best_step']==best['step']and r['best_val']==best['val_macro_f1'];checkpoint=torch.load(path,map_location='cpu',weights_only=True);assert checkpoint['protocol']==p and checkpoint['seed']==seed and checkpoint['steps']==2000 and checkpoint['best_step']==best['step']and sum(a.numel()for a in checkpoint['model'].values())==r['parameter_count']==19994 and file_hash(path)==r['checkpoint_sha256'];replays[f'{arm}_{seed}']={'base_sha256':bh.hexdigest(),'supplement_sha256':sh.hexdigest()};print('verified projection full sampler, diagnostics and3 updates',arm,seed,flush=True)
 val=capacity.read('val',m);target=np.concatenate([a['labels']for a in val]);ts=np.concatenate([a['xy'][:,2]for a in val]);ids=np.concatenate([np.full(len(a['labels']),int(a['video_id']),np.int32)for a in val]);frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256'];offsets=dict(p['fixed_offsets']);results={}
 for arm in work.ARMS:
  outputs=[]
  for seed in range(3):
   path=work.ckpt(arm,seed);assert file_hash(path)==frozen['checkpoint_sha256'][f'{arm}_{seed}'];state=torch.load(path,map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(a['decoded'],state)for a in val]);outputs.append(values)
   with np.load(ROOT/arm/'logits'/f'seed{seed}_val.npz')as a:assert np.array_equal(values,a['logits'])and np.array_equal(a['labels'],target)and np.array_equal(a['times'],ts)and np.array_equal(a['video_ids'],ids)and str(a['checkpoint_sha256'])==file_hash(path)
   if arm in work.NEW:assert abs(independent_eval(values,target,ts,0)['macro_f1_present']-s['training_reports'][arm][seed]['best_val'])<1e-12
  if arm in work.NEW:
   curve=[];best=-1;choice=None
   for offset in sorted(p['calibration_grid'],key=lambda v:(abs(v),v)):
    vals=[independent_eval(value,target,ts,offset)['macro_f1_present']for value in outputs];mean=float(np.mean(vals));curve.append({'offset':float(offset),'macro_f1_by_seed':vals,'mean_macro_f1':mean})
    if mean>best:best=mean;choice=float(offset)
   equal(curve,frozen['curves'][arm]);offsets[arm]=choice
  runs=[{'seed':seed,'raw_validation':independent_eval(value,target,ts,0),'calibrated_validation':independent_eval(value,target,ts,offsets[arm])}for seed,value in enumerate(outputs)];r={'offset':offsets[arm],'runs':runs,'mean_calibrated_validation_macro':float(np.mean([a['calibrated_validation']['macro_f1_present']for a in runs]))};equal(r,s['results'][arm]);results[arm]=r;print('verified projected val',arm,flush=True)
 for left,right in [('project_sam','aux_sam'),('project_extra_gt','extra_gt'),('project_sam','project_extra_gt'),('project_sam','gt32')]:
  vals=[results[left]['runs'][seed]['calibrated_validation']['macro_f1_present']-results[right]['runs'][seed]['calibrated_validation']['macro_f1_present']for seed in range(3)];equal({'per_seed_validation_macro_difference':vals,'mean':float(np.mean(vals))},s['validation_contrasts'][left+'_minus_'+right])
 assert max(work.ARMS,key=lambda a:results[a]['mean_calibrated_validation_macro'])==s['validation_selected_arm']==frozen['validation_selected_arm']and offsets==frozen['offsets']
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all6_full_sampler_and_budget_replays':replays,'all12000_step_projection_algebra_and_update_statistics_verified':True,'independent_FP64_three_step_AdamW_prefixes':6,'first_three_historical_steps_do_not_trigger_projection':True,'three_conflicting_synthetic_projection_AdamW_updates_checked_in_FP64':True,'maximum_prefix_parameter_absolute_difference':prefix_max,'all6_best_checkpoint_rules_and15_val_logits_recomputed':True,'all_calibrations_four_contrasts_and_selection_recomputed':True,'all_frozen_inputs_unchanged':True,'no_confirmation_or_deployment_change':True});print('DONE independent projection control verification',prefix_max,flush=True)
if __name__=='__main__':main()
