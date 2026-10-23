"""Manual source canonicalization, unchanged geometry, and isolated gradient replay."""
from pathlib import Path
import gc,json
import numpy as np
import torch
from semantic.model import SemanticTail,GEOMETRY_KEYS
from semantic.ytvis import load_video
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha
from tools.load_opend4rt import load_model,DEFAULT_VARIANT
ROOT=Path('semantic_artifacts/source_query_adapter_probe')
def thash(q):
 import hashlib
 return hashlib.sha256(q.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());assert p['round']==104 and p['steps']==5 and sha(ROOT/'protocol.json')==s['protocol_sha256']
 for path,h in p['input_sha256'].items():assert sha(path)==h
 for path,h in s['artifacts_sha256'].items():assert sha(path)==h
 manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text());clips=sorted([c for c in manifest['train']if c['video_id']in p['video_ids']],key=lambda c:int(c['video_id']));rng=np.random.Generator(np.random.PCG64(104));items=[]
 for i,c in enumerate(clips):
  vid=int(c['video_id'])
  with np.load(Path('semantic_artifacts/features_ovis/sam/train')/f'{vid}.npz')as a:xy=a['xy'].copy();labels=a['labels'].copy()
  idx=np.concatenate((np.flatnonzero((xy[:,2]==0)&(labels>0))[:32],np.flatnonzero((xy[:,2]==0)&(labels==0))[:32]));assert idx.tolist()==p['query_indices'][i];xy=xy[idx];q={'u':torch.tensor((xy[:,0]+.5)/128,device='cuda',dtype=torch.float32)[None],'v':torch.tensor((xy[:,1]+.5)/128,device='cuda',dtype=torch.float32)[None],'t_src':torch.tensor(xy[:,2],device='cuda',dtype=torch.long)[None],'t_tgt':torch.tensor(rng.integers(8,size=(1,64)),device='cuda'),'t_cam':torch.tensor(rng.integers(8,size=(1,64)),device='cuda')};items.append({'video':load_video(c)[None].cuda().to(torch.bfloat16),'query':q,'labels':torch.tensor(labels[idx].astype(np.int64),device='cuda')})
 pair={'video':torch.cat([r['video']for r in items]),'query':{k:torch.cat([r['query'][k]for r in items])for k in items[0]['query']}};cases=items+[pair]
 torch.manual_seed(0);base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml');tail=SemanticTail(base,26).cuda().eval();base=base.to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False);assert {n:thash(q)for n,q in base.named_parameters()}==s['base_parameter_sha256_before_and_after'];assert not{q.untyped_storage().data_ptr()for q in base.parameters()}&{q.untyped_storage().data_ptr()for q in tail.parameters()}
 def install(state):
  with torch.no_grad():
   for n,q in tail.named_parameters():
    if n in state:q.copy_(state[n].to(q.device))
 def semantics(batch):
  video=batch['video'];q=batch['query'];src=q['t_src']
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   memory=base.encode_video(video,aspect_ratio=batch.get('aspect_ratio'));x=base.query_embedder(video=video,u=q['u'].to(video.dtype),v=q['v'].to(video.dtype),t_src=src,t_tgt=src,t_cam=src)
   for block in base.decoder.blocks[:-2]:x=block(x,memory)
  with torch.autocast('cuda',dtype=torch.bfloat16):return tail(x,memory)['semantic_logits']
 for phase in ('before','after'):
  state=torch.load(ROOT/('initial_adapter.pt'if phase=='before'else'final_adapter.pt'),map_location='cpu',weights_only=True);install(state)
  for j,r in enumerate(cases):
   original={'video':r['video'],'query':r['query']};tc={'video':r['video'],'query':dict(r['query'])};tc['query']['t_tgt']=(tc['query']['t_tgt']+1)%8;tc['query']['t_cam']=(tc['query']['t_cam']+3)%8;src={'video':r['video'],'query':dict(r['query'])};src['query']['t_src']=(src['query']['t_src']+1)%8;aspect={**original,'aspect_ratio':torch.tensor([1.2]if j<2 else[1.2,.8],device='cuda',dtype=torch.bfloat16)};observed={}
   for name,batch in {'original':original,'target_camera':tc,'source':src,'aspect':aspect}.items():
    record=next(t for t in s['records']if t['phase']==phase and t['case']==j and t['variant']==name);assert sha(record['output_path'])==record['sha256']
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):geometry=base(batch);z=semantics(batch)
    with np.load(record['output_path'])as saved:
     assert np.array_equal(z.float().cpu().numpy(),saved['semantic_logits'])
     for k in GEOMETRY_KEYS:assert np.array_equal(geometry[k].float().cpu().numpy(),saved[k])and record['geometry_max_difference'][k]==0
     observed[name]=saved['semantic_logits'].copy()
    if phase=='after':
     with np.load(ROOT/f'before_case{j}_{name}.npz')as before,np.load(record['output_path'])as after:
      for k in GEOMETRY_KEYS:assert np.array_equal(before[k],after[k])
   assert np.array_equal(observed['original'],observed['target_camera'])
  print('Verified manual canonical semantics and twelve geometry cases',phase,flush=True)
 initial=torch.load(ROOT/'initial_adapter.pt',map_location='cpu',weights_only=True);final=torch.load(ROOT/'final_adapter.pt',map_location='cpu',weights_only=True);install(initial);batch={'video':items[0]['video'],'query':items[0]['query']};z=semantics(batch)[0].float().log_softmax(-1);loss=-z[torch.arange(64,device='cuda'),items[0]['labels']].mean();loss.backward();stored=torch.load(ROOT/'first_gradients.pt',map_location='cpu',weights_only=True)['gradients'];grads={n:q.grad for n,q in tail.named_parameters()if q.grad is not None};assert set(grads)==set(stored);gradient_error=0.
 for n,g in grads.items():assert n.endswith(('.a','.b'))or n.startswith('semantic_head.');torch.testing.assert_close(g.cpu(),stored[n],rtol=5e-4,atol=5e-6);gradient_error=max(gradient_error,float((g.cpu()-stored[n]).abs().max()))
 assert all(q.grad is None for q in base.parameters());saved=torch.load(ROOT/'before_final.pt',map_location='cpu',weights_only=True);install(saved['adapter']);opt=torch.optim.AdamW([{'params':[q for n,q in tail.named_parameters()if q.requires_grad and not n.startswith('semantic_head.')],'lr':.0001},{'params':[q for n,q in tail.named_parameters()if q.requires_grad and n.startswith('semantic_head.')],'lr':.001}],weight_decay=.01);opt.load_state_dict(saved['optimizer']);z=semantics(batch)[0].float().log_softmax(-1);loss=-z[torch.arange(64,device='cuda'),items[0]['labels']].mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step();parameter_error=0.
 for n,q in tail.named_parameters():
  if n in final:torch.testing.assert_close(q.detach().cpu(),final[n],rtol=3e-4,atol=2e-6);parameter_error=max(parameter_error,float((q.detach().cpu()-final[n]).abs().max()))
 assert any(not torch.equal(initial[n],final[n])for n in final if n.endswith('.b'));assert {n:thash(q)for n,q in base.named_parameters()}==s['base_parameter_sha256_before_and_after'];assert len(s['records'])==24
 for path,h in p['input_sha256'].items():assert sha(path)==h
 atomic_json(ROOT/'verification.json',{'round':104,'summary_sha256':sha(ROOT/'summary.json'),'verifier_sha256':sha('semantic/verify_source_query_adapter_probe.py'),'all24_geometry_outputs_equal_frozen_base_and_before_after':True,'manual_source_canonical_semantics_all24_match_exact':True,'target_camera_semantic_invariance_both_stages_B1_B2':True,'only_tail_gradient_white_list_and_base_no_grad':True,'first_gradient_max_abs_error':gradient_error,'last_optimizer_parameter_max_abs_error':parameter_error,'base_parameters_and_all_inputs_unchanged':True,'no_accuracy_or_new_confirmation_claim':True});print('DONE independent source-query adapter probe verification',flush=True)
if __name__=='__main__':main()
