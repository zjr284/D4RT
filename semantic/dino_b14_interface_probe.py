"""Frozen B14 integration correctness on two already-observed validation clips."""
import fcntl,json,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
from semantic.dino_b14_interface import RECIPES,resolve_recipe,load_b14_research_adapter
from semantic.dino_native_interface import load_source_pair
from semantic.dino_interface import cuda_bf16
from semantic.dino_interface_probe import queries,measured,delta
from semantic import dino_backbone_training as capacity
from semantic.dino_backbone_resource import WEIGHTS,VENDOR
from semantic.model import GEOMETRY_KEYS
from semantic.semantic_feature_control import save_npz
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from tools.load_opend4rt import DEFAULT_VARIANT
ROOT=Path('semantic_artifacts/dino_b14_interface_probe')

def freeze():
 m=json.loads((capacity.ROOT/'manifest.json').read_text());clips=sorted(m['val'],key=lambda c:str(c['video_id']))[:2];assert [c['video_id']for c in clips]==[109,130]
 specs={};paths=[WEIGHTS,DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml',capacity.ROOT/'manifest.json',Path('semantic/dino_b14_interface.py'),Path('semantic/dino_b14_interface_probe.py'),Path('semantic/OVIS_DINO_B14_INTERFACE_PROTOCOL.md'),Path('tests/test_dino_b14_interface.py'),Path('semantic/dino_native_interface.py'),Path('semantic/dino_interface.py'),Path('semantic/dino_interface_probe.py'),Path('semantic/model.py'),Path('semantic/semantic_feature_control.py'),Path('tools/load_opend4rt.py')]+sorted(VENDOR.rglob('*.py'))
 for recipe in RECIPES:
  specs[recipe]={}
  for seed in range(3):
   path,sha,offset,source=resolve_recipe(recipe,seed);specs[recipe][str(seed)]={'path':str(path),'sha256':sha,'offset':offset,'source':str(source)};paths.extend([path,source/'summary.json',source/'verification.json',source/'frozen_selection.json'])
 for c in clips:paths.extend([capacity.fp('val',c['video_id'])]+[Path(frame)for frame in c['frames']])
 unit=subprocess.run(['.venv/bin/python','tests/test_dino_b14_interface.py'],capture_output=True,text=True,check=True);(ROOT/'unit_checks.log').write_text(unit.stdout+unit.stderr);assert 'PASS B14768'in unit.stdout
 p={'round':73,'video_ids':[109,130],'recipes':list(RECIPES),'head_specs':specs,'queries_full':2048,'geometry_query_indices':list(range(0,2048,32)),'extra_queries':8,'real_batch_sizes':[1,2],'time_change':'target+1 camera+3 mod8','feature_dim':768,'feature_size':448,'geometry_array_storage':'FP32 upcast preserving original values','training_updates':0,'calibration_updates':0,'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'unit_checks_sha256':file_hash(ROOT/'unit_checks.log'),'input_sha256':{str(path):file_hash(path)for path in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'manifest.json',{'val':clips});return clips,p

def array(value):return value.detach().float().cpu().numpy()
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed B14 interface probe exists')
  torch.backends.cuda.matmul.allow_tf32=False;clips,p=freeze();adapter=load_b14_research_adapter('gt32',0);items=[];full=[];cases=[];singles={}
  for clip in clips:
   vid=clip['video_id']
   with np.load(capacity.fp('val',vid))as a:cache={k:a[k].copy()for k in a.files}
   geometry,native=load_source_pair([clip['frames']]);query=queries(cache,np.arange(2048),extra=False);features=adapter.native_features(native.cuda(),query);cached=torch.from_numpy(cache['decoded']).float().cuda()[None];feature_error=delta(features,cached);save_npz(ROOT/'online_features'/f'{vid}.npz',decoded=array(features[0]),xy=cache['xy'])
   for recipe in RECIPES:
    for seed in range(3):
     spec=p['head_specs'][recipe][str(seed)];adapter.background_offset=spec['offset'];adapter.load_head(spec['path'],spec['sha256']);out=adapter.semantic_outputs(features);ref=adapter.semantic_outputs(cached);path=ROOT/'full_semantics'/f'{recipe}_{seed}_{vid}.npz';save_npz(path,raw=array(out['semantic_raw_logits'][0]),calibrated=array(out['semantic_logits'][0]),pred=out['semantic_labels'][0].cpu().numpy(),cached_raw=array(ref['semantic_raw_logits'][0]),cached_pred=ref['semantic_labels'][0].cpu().numpy());full.append({'video_id':vid,'recipe':recipe,'seed':seed,'feature_error':feature_error,'raw_logit_error':delta(out['semantic_raw_logits'],ref['semantic_raw_logits']),'label_changes':int((out['semantic_labels']!=ref['semantic_labels']).sum()),'output_path':str(path),'output_sha256':file_hash(path)})
   batch={'video':geometry.cuda().bfloat16(),'source_frames':[clip['frames']],'query':queries(cache,np.arange(0,2048,32),extra=True)};items.append((clip,cache,batch));print('B14 full-query features',vid,feature_error,flush=True)
  groups=[('B1',[items[0]]),('B1',[items[1]]),('B2',items)]
  for kind,group in groups:
   ids=[a[0]['video_id']for a in group];batch={'video':torch.cat([a[2]['video']for a in group]),'source_frames':[a[0]['frames']for a in group],'query':{k:torch.cat([a[2]['query'][k]for a in group])for k in group[0][2]['query']}};q=batch['query'];changed={k:v.clone()for k,v in q.items()};changed['t_tgt']=(changed['t_tgt']+1)%8;changed['t_cam']=(changed['t_cam']+3)%8;altered_batch={**batch,'query':changed};video_before=batch['video'].clone();q_before={k:v.clone()for k,v in q.items()};path_before=[list(x)for x in batch['source_frames']]
   def geometry(b):
    with torch.inference_mode(),cuda_bf16(batch['video'].device):return adapter.base(b)
   reference,base_cost=measured(lambda:geometry(batch));changed_reference=geometry(altered_batch)
   for recipe in RECIPES:
    for seed in range(3):
     spec=p['head_specs'][recipe][str(seed)];adapter.background_offset=spec['offset'];adapter.load_head(spec['path'],spec['sha256']);result,cost=measured(lambda:adapter(batch));altered=adapter(altered_batch);geom={k:delta(result[k],reference[k])for k in GEOMETRY_KEYS};changed_geom={k:delta(altered[k],changed_reference[k])for k in GEOMETRY_KEYS};time=delta(result['semantic_logits'],altered['semantic_logits']);assert all(a['bitwise_equal']for a in [*geom.values(),*changed_geom.values()])and time['bitwise_equal']
     adapter.load_head(spec['path'],spec['sha256']);reloaded=adapter.semantic_outputs(adapter.native_features(adapter.source_rgb(batch),q));assert torch.equal(reloaded['semantic_logits'],result['semantic_logits'])and torch.equal(result['semantic_labels'],result['semantic_logits'].argmax(-1));assert all(torch.isfinite(result[k]).all()for k in GEOMETRY_KEYS+('semantic_logits','semantic_raw_logits'))
     package={f'{prefix}__{k}':array(out[k])for prefix,out in [('reference',reference),('actual',result),('changed_reference',changed_reference),('changed_actual',altered)]for k in GEOMETRY_KEYS};package.update(raw=array(result['semantic_raw_logits']),calibrated=array(result['semantic_logits']),pred=result['semantic_labels'].cpu().numpy(),changed_raw=array(altered['semantic_raw_logits']),changed_calibrated=array(altered['semantic_logits']),reloaded_calibrated=array(reloaded['semantic_logits']));package.update({'query_'+k:v.cpu().numpy()for k,v in q.items()});path=ROOT/'cases'/f'{kind}_{ids[0]}_{recipe}_{seed}.npz';save_npz(path,**package)
     row={'kind':kind,'video_ids':ids,'recipe':recipe,'seed':seed,'queries_per_video':72,'geometry':geom,'changed_time_geometry':changed_geom,'semantic_time':time,'reload_equal':True,'base_forward_cost':base_cost,'full_interface_cost':cost,'output_path':str(path),'output_sha256':file_hash(path)}
     if kind=='B1':singles[(ids[0],recipe,seed)]=package['raw']
     else:row['B2_vs_B1_raw_logits']=delta(result['semantic_raw_logits'],torch.from_numpy(np.concatenate([singles[(vid,recipe,seed)]for vid in ids])).cuda())
     cases.append(row);atomic_json(ROOT/'progress.json',{'cases':cases});print('B14 live geometry verified',kind,ids,recipe,seed,flush=True)
   assert torch.equal(video_before,batch['video'])and path_before==batch['source_frames']and all(torch.equal(q_before[k],q[k])for k in q)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  assert all(not x.requires_grad for x in adapter.parameters())and not adapter.training
  atomic_json(ROOT/'summary.json',{'protocol':p,'full_query_semantics':full,'cases':cases,'online_feature_sha256':{str(path):file_hash(path)for path in sorted((ROOT/'online_features').glob('*.npz'))},'all_model_parameters_frozen':True,'all_inputs_unchanged':True,'new_confirmation_groups':0,'deployment_unchanged':'native_gt','limits':['already observed two OVIS validation clips, not new generalization','explicit research recipes, no automatic deployment change','same-batch geometry preservation, not geometry accuracy or cross-batch equality','one sequential invocation per case, not production throughput']});print('DONE B14 explicit research interface',len(cases),len(full),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
