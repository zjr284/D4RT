"""Independent pools/RNG and autograd audit of all7200 analytic CE gradients."""
import hashlib,json
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_b14_gradient_audit as work
from semantic import dino_backbone_training as capacity
from semantic import dino_b14_anchor_oracle as oracle
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def close(a,b):assert np.allclose(a,b,rtol=2e-5,atol=2e-6),float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def summarize(a):
 out={'base_mean_loss':float(a['losses'][:,0].mean()),'base_mean_gradient_norm':float(a['norms'][:,0].mean()),'oracle_SAM_mean_relative_gradient_change':float(a['oracle_SAM_relative_gradient_change'].mean()),'supplement_vs_base':{}}
 for k,name in enumerate(('extra_GT','later_SAM','later_oracle')):
  b,c=a['mean_gradients'][[0,k+1]];out['supplement_vs_base'][name]={'mean_loss':float(a['losses'][:,k+1].mean()),'mean_batch_cosine':float(a['cosines'][:,k].mean()),'negative_dot_fraction':float((a['dots'][:,k]<0).mean()),'mean_supplement_to_base_norm_ratio':float(a['ratios'][:,k].mean()),'supplement_norm_exceeds_base_fraction':float((a['norms'][:,k+1]>a['norms'][:,0]).mean()),'cosine_of_mean_gradients':float(np.dot(b,c)/max(float(np.linalg.norm(b)*np.linalg.norm(c)),1e-12))}
 return out

def nested(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:nested(a[k],b[k])
 else:close(a,b)

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==74 and p['analyzed_prefix_steps']==200 and p['replayed_steps']==2000
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 m=json.loads((ROOT/'manifest.json').read_text());base=capacity.read('train',m);whole=oracle.read(m);later=[]
 for a,b in zip(whole,base):
  assert int(a['video_id'])==int(b['video_id']);mask=a['xy'][:,2]==0
  for key in ('decoded','labels','xy'):assert np.array_equal(a[key][mask],b[key])
  later.append({k:a[k][~mask]for k in ('decoded','labels','truth')})
 bx=torch.from_numpy(np.concatenate([a['decoded']for a in base])).float().cuda();by=torch.tensor(np.concatenate([a['labels']for a in base]),dtype=torch.long,device='cuda');sx=torch.from_numpy(np.concatenate([a['decoded']for a in later])).float().cuda();sy=torch.tensor(np.concatenate([a['labels']for a in later]),dtype=torch.long,device='cuda');truth=torch.tensor(np.concatenate([a['truth']for a in later]),dtype=torch.long,device='cuda');fg=[torch.where(y>0)[0]for y in (by,by,sy)];bg=[torch.where(y==0)[0]for y in (by,by,sy)];count=0;largest_gradient_error=0.
 for seed in range(3):
  generators=[torch.Generator(device='cuda').manual_seed(k)for k in (seed,20261071+seed,20261071+seed)];hashes=[hashlib.sha256()for _ in range(3)];ix=[[],[],[]]
  for step in range(2000):
   for kind in range(3):
    f=fg[kind][torch.randint(len(fg[kind]),(16,),generator=generators[kind],device='cuda')];b=bg[kind][torch.randint(len(bg[kind]),(16,),generator=generators[kind],device='cuda')];sample=torch.cat([f,b]).cpu().numpy();hashes[kind].update(sample.tobytes())
    if step<200:ix[kind].append(sample)
  assert [h.hexdigest()for h in hashes]==s['sampler_replays'][str(seed)]['full_2000_step_hashes'];ix=[np.stack(a)for a in ix]
  with np.load(ROOT/'indices'/f'seed{seed}.npz')as a:assert all(np.array_equal(a[k],value)for k,value in zip(('base','extra','teacher'),ix))
  indices=[torch.from_numpy(a).cuda()for a in ix];assert int((sy[indices[2]]!=truth[indices[2]]).sum())==s['sampler_replays'][str(seed)]['analyzed_teacher_target_changes']
  for state in p['states']:
   r=s['results'][f'{state}_{seed}'];assert file_hash(r['output_path'])==r['output_sha256'];checkpoint=torch.load(work.prior.ckpt(state,seed),map_location='cpu',weights_only=True)['model'];weight=checkpoint['classifier.weight'].cuda().requires_grad_(True);bias=checkpoint['classifier.bias'].cuda().requires_grad_(True);rows={k:[]for k in ('losses','norms','cosines','dots','ratios','oracle_SAM_relative_gradient_change')};mean=torch.zeros(4,19994,device='cuda')
   for step in range(200):
    bi,ei,ti=(a[step]for a in indices);sets=((bx[bi],by[bi]),(bx[ei],by[ei]),(sx[ti],sy[ti]),(sx[ti],truth[ti]));gradients=[];losses=[]
    for x,y in sets:
     loss=F.cross_entropy(F.linear(x,weight,bias),y);gw,gb=torch.autograd.grad(loss,(weight,bias));g=torch.cat([gw.flatten(),gb]);gradients.append(g);losses.append(loss.detach());analytic,aloss=work.gradient(x,y,weight.detach(),bias.detach());largest_gradient_error=max(largest_gradient_error,float((g-analytic).abs().max()));assert torch.allclose(g,analytic,rtol=2e-5,atol=2e-6);count+=1
    g=torch.stack(gradients);n=torch.linalg.vector_norm(g,dim=1);d=(g[1:]*g[0]).sum(1);mean+=g/200;rows['losses'].append(torch.stack(losses).cpu().numpy());rows['norms'].append(n.cpu().numpy());rows['dots'].append(d.cpu().numpy());rows['cosines'].append((d/(n[1:]*n[0]).clamp_min(1e-12)).cpu().numpy());rows['ratios'].append((n[1:]/n[0].clamp_min(1e-12)).cpu().numpy());rows['oracle_SAM_relative_gradient_change'].append(float(torch.linalg.vector_norm(g[3]-g[2])/n[2].clamp_min(1e-12)))
   arrays={key:np.asarray(value)for key,value in rows.items()};arrays['mean_gradients']=mean.cpu().numpy()
   with np.load(r['output_path'])as a:
    for key,value in arrays.items():close(value,a[key])
    nested(summarize({key:a[key]for key in a.files}),r['statistics'])
   nested(summarize(arrays),r['statistics']);print('verified autograd gradients and summaries',state,seed,flush=True)
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert count==7200 and s['new_confirmation_groups']==0 and not s['optimizer_causal_claim']and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all80_source_and_later_pools_checked':True,'three_full_2000_step_three_generator_hashes_recomputed':True,'all200_step_prefix_indices_and_corrected_targets_verified':True,'independent_autograd_gradients_recomputed':count,'maximum_analytic_vs_autograd_absolute_gradient_error':largest_gradient_error,'all_per_batch_metrics_mean_gradients_and_aggregates_recomputed':True,'all_frozen_inputs_unchanged':True,'no_training_or_confirmation':True});print('DONE independent gradient audit',count,largest_gradient_error,flush=True)
if __name__=='__main__':main()
