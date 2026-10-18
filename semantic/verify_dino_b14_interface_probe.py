"""Independent B14 source feature/head arrays and live geometry reference replay."""
import gc,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from semantic import dino_b14_interface_probe as work
from semantic.dino_b14_interface import resolve_recipe
from semantic.dino_backbone_resource import load_b14
from semantic.verify_dino_backbone_training import independent_b14
from semantic.verify_external_semantic_confirmation import equal,linear
from semantic.dino_interface_probe import delta
from semantic.model import GEOMETRY_KEYS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from tools.load_opend4rt import DEFAULT_VARIANT,load_model
ROOT=work.ROOT

def compare_stats(a,b,reported):equal(delta(torch.from_numpy(a).cuda(),torch.from_numpy(b).cuda()),reported)
def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==73 and p['video_ids']==[109,130]and p['recipes']==['gt32','class_video','aux_sam'];assert file_hash(ROOT/'unit_checks.log')==p['unit_checks_sha256']and 'PASS B14768'in(ROOT/'unit_checks.log').read_text()
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 clips=json.loads((ROOT/'manifest.json').read_text())['val'];assert [c['video_id']for c in clips]==p['video_ids'];full=s['full_query_semantics'];cases=s['cases'];assert len(full)==18 and len(cases)==27
 assert {(r['video_id'],r['recipe'],r['seed'])for r in full}=={(vid,recipe,seed)for vid in p['video_ids']for recipe in p['recipes']for seed in range(3)}
 assert [(r['kind'],r['video_ids'],r['recipe'],r['seed'])for r in cases]==[(kind,ids,recipe,seed)for kind,ids in [('B1',[109]),('B1',[130]),('B2',[109,130])]for recipe in p['recipes']for seed in range(3)]
 packages={};online={}
 for clip in clips:
  vid=clip['video_id'];path=ROOT/'online_features'/f'{vid}.npz';assert file_hash(path)==s['online_feature_sha256'][str(path)]
  with np.load(work.capacity.fp('val',vid))as a:packages[vid]={k:a[k].copy()for k in a.files}
  with np.load(path)as a:assert np.array_equal(a['xy'],packages[vid]['xy']);online[vid]=a['decoded'].copy();assert online[vid].shape==(2048,768)and np.array_equal(online[vid],online[vid].astype(np.float16).astype(np.float32))and np.isfinite(online[vid]).all()
 model=load_b14();first=clips[0];xy=packages[109]['xy'].astype(np.float32);xy[:,:2]=(torch.from_numpy(xy[:,:2]/127).cuda()*127).cpu().numpy();replay=independent_b14(model,first['frames'],xy);assert np.array_equal(replay.astype(np.float32),online[109]);del model;gc.collect();torch.cuda.empty_cache()
 for r in full:
  vid,recipe,seed=r['video_id'],r['recipe'],r['seed'];spec=p['head_specs'][recipe][str(seed)];path,sha,offset,source=resolve_recipe(recipe,seed);assert (str(path),sha,offset,str(source))==(spec['path'],spec['sha256'],spec['offset'],spec['source']);assert file_hash(r['output_path'])==r['output_sha256'];state=torch.load(path,map_location='cpu',weights_only=True)['model'];raw=linear(online[vid],state);cached=linear(packages[vid]['decoded'],state);cal=raw.copy();cal[:,0]+=offset;oldcal=cached.copy();oldcal[:,0]+=offset
  with np.load(r['output_path'])as a:
   assert np.array_equal(a['raw'],raw)and np.array_equal(a['cached_raw'],cached)and np.array_equal(a['calibrated'],cal)and np.array_equal(a['pred'],cal.argmax(-1))and np.array_equal(a['cached_pred'],oldcal.argmax(-1));assert r['label_changes']==int((cal.argmax(-1)!=oldcal.argmax(-1)).sum())
  compare_stats(online[vid],packages[vid]['decoded'].astype(np.float32),r['feature_error']);compare_stats(raw,cached,r['raw_logit_error'])
 print('verified first full B14 feature replay and18 complete semantic arrays',flush=True)
 loaded={};singles={}
 for r in cases:
  assert file_hash(r['output_path'])==r['output_sha256']and r['queries_per_video']==72;data=np.load(r['output_path']);a={k:data[k].copy()for k in data.files};data.close();key=(r['kind'],tuple(r['video_ids']),r['recipe'],r['seed']);loaded[key]=a;offset=p['head_specs'][r['recipe']][str(r['seed'])]['offset'];cal=a['raw'].copy();cal[...,0]+=offset;assert np.array_equal(cal,a['calibrated'])and np.array_equal(cal.argmax(-1),a['pred'])and np.array_equal(cal,a['changed_calibrated'])and np.array_equal(a['raw'],a['changed_raw'])and np.array_equal(cal,a['reloaded_calibrated'])and r['reload_equal']
  for b,vid in enumerate(r['video_ids']):
   cache=packages[vid];xy=cache['xy'][::32];uv=np.array([[0,0],[1,0],[0,1],[1,1],[.5,.5],[.125,.875],[.333,.667],[.1,.9]],np.float32);t=np.arange(8)
   expected={'u':np.r_[xy[:,0].astype(np.float32)/127,uv[:,0]],'v':np.r_[xy[:,1].astype(np.float32)/127,uv[:,1]],'t_src':np.r_[xy[:,2],t],'t_tgt':np.r_[cache['t_tgt'][::32],(t+1)%8],'t_cam':np.r_[cache['t_cam'][::32],(t+3)%8]}
   for k,value in expected.items():assert np.array_equal(a['query_'+k][b],value)
  for k in GEOMETRY_KEYS:
   assert np.array_equal(a['reference__'+k],a['actual__'+k])and np.array_equal(a['changed_reference__'+k],a['changed_actual__'+k])and np.isfinite(a['actual__'+k]).all();compare_stats(a['actual__'+k],a['reference__'+k],r['geometry'][k]);compare_stats(a['changed_actual__'+k],a['changed_reference__'+k],r['changed_time_geometry'][k])
  compare_stats(a['calibrated'],a['changed_calibrated'],r['semantic_time']);assert a['query_t_src'].min()==0 and a['query_t_src'].max()==7
  for keycost in ('base_forward_cost','full_interface_cost'):assert r[keycost]['seconds']>0 and r[keycost]['peak_cuda_allocated_bytes']>0
  if r['kind']=='B1':singles[(r['video_ids'][0],r['recipe'],r['seed'])]=a['raw']
  else:compare_stats(a['raw'],np.concatenate([singles[(vid,r['recipe'],r['seed'])]for vid in r['video_ids']]),r['B2_vs_B1_raw_logits'])
 print('verified27 saved real interface cases including324 geometry comparisons',flush=True)
 # Recreate geometry RGB independently of the adapter, then rerun the bare model.
 base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False);replayed=[]
 for kind,group in [('B1',[clips[0]]),('B2',clips)]:
  videos=[]
  for clip in group:
   frames=[]
   for path in clip['frames']:
    with Image.open(path)as image:pixels=np.asarray(image.convert('RGB').resize((128,128),Image.Resampling.BILINEAR),np.float32)
    frames.append(torch.from_numpy(pixels.copy()).permute(2,0,1)/255)
   videos.append(torch.stack(frames))
  video=torch.stack(videos).cuda().bfloat16();ids=tuple(c['video_id']for c in group);a=loaded[(kind,ids,'gt32',0)];q={k:torch.from_numpy(a['query_'+k]).cuda()for k in ('u','v','t_src','t_tgt','t_cam')};changed={k:v.clone()for k,v in q.items()};changed['t_tgt']=(q['t_tgt']+1)%8;changed['t_cam']=(q['t_cam']+3)%8
  for prefix,query in [('reference',q),('changed_reference',changed)]:
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=base({'video':video,'query':query})
   for k in GEOMETRY_KEYS:assert np.array_equal(out[k].float().cpu().numpy(),a[prefix+'__'+k])
  replayed.append({'kind':kind,'video_ids':list(ids),'six_fields_original_and_changed_bitwise_equal':True});print('independent bare D4RT replay',kind,ids,flush=True)
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 d=json.loads((ROOT/'reload_diagnostic.json').read_text());assert d['protocol']['original_summary_sha256']==file_hash(ROOT/'summary.json')and file_hash('semantic/dino_b14_reload_diagnostic.py')==d['protocol']['script_sha256']and len(d['cases'])==12
 for row in d['cases']:
  values=list(row['fields'].values())
  if row['arm']=='parameter_requires_grad_false':assert all(x['bitwise_equal']and x['max_absolute_difference']==0 for x in values)
  else:assert all(not x['bitwise_equal']and x['max_absolute_difference']>0 for x in values)
 assert s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt'and s['all_inputs_unchanged']and s['all_model_parameters_frozen'];atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'first109_B14_features_independently_recomputed':True,'all18_full_semantic_arrays_recomputed':True,'all27_geometry_time_reload_cases_verified_from_arrays':True,'bare_geometry_independent_replays':replayed,'geometry_saved_array_comparisons':324,'geometry_independent_array_replays':24,'initial_unfrozen_reference_failure_preserved_and_frozen_reference_corrected':True,'bounded_parameter_freeze_toggle_diagnosis_sha256':file_hash(ROOT/'reload_diagnostic.json'),'all_three_source_val_recipes_and9_head_fingerprints_verified':True,'all_frozen_inputs_unchanged':True,'no_new_confirmation_or_deployment_change':True});print('DONE independent B14 interface verification',flush=True)
if __name__=='__main__':main()
