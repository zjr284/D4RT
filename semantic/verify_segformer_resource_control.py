"""Independent official model integrity, private environment and RGB feature audit."""
import hashlib,importlib.metadata,inspect,json,subprocess,sys
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from semantic import segformer_features as features
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=features.ROOT

def manual_maps(model,alias,video):
 mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  encoded=model.segformer(pixel_values=(video-mean)/std,output_hidden_states=True,return_dict=True);states=encoded.hidden_states;last=states[-1]
  if last.ndim==3:last=last.transpose(1,2).reshape(len(video),256,14,14)
  result={alias+'_encoder':last.float()}
  if alias=='ade':
   maps=[]
   for state,projection in zip(states,model.decode_head.linear_c):
    h,w=state.shape[-2:];value=projection(state).permute(0,2,1).reshape(len(video),256,h,w);maps.append(F.interpolate(value,size=states[0].shape[-2:],mode='bilinear',align_corners=False))
   value=model.decode_head.linear_fuse(torch.cat(list(reversed(maps)),dim=1));value=model.decode_head.batch_norm(value);value=model.decode_head.activation(value);value=model.decode_head.dropout(value);result['ade_decoder']=value.float()
 return result

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==81 and Path(sys.prefix).resolve()==(ROOT/'env').resolve()
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 code='import importlib.metadata,json;print(json.dumps({k:importlib.metadata.version(k)for k in '+repr(tuple(p['original_environment']))+'}))';parent=json.loads(subprocess.check_output(['.venv/bin/python','-c',code],text=True));assert parent==p['original_environment']==s['original_environment_after'];assert str(Path(torch.__file__).resolve()).startswith(str(Path(p['original_site_packages']).resolve()))
 import transformers,huggingface_hub,safetensors
 assert transformers.__version__=='4.57.6'and huggingface_hub.__version__=='0.36.0'and str(Path(transformers.__file__).resolve()).startswith(str((ROOT/'env').resolve()))and str(Path(huggingface_hub.__file__).resolve()).startswith(str((ROOT/'env').resolve()));assert file_hash(ROOT/'env_install.json')==s['env_install_report_sha256'];install=json.loads((ROOT/'env_install.json').read_text());assert not {'torch','torchvision','numpy','pillow'}&{r['metadata']['name'].lower().replace('_','-')for r in install['install']}
 for record in install['install']:assert urlsplit(record['download_info']['url']).hostname=='files.pythonhosted.org'and record['download_info']['archive_info']['hashes']['sha256']
 specs=json.loads((ROOT/'model_specs.json').read_text());assert specs==s['model_specs']and file_hash(ROOT/'model_specs.json')==s['model_specs_sha256'];license_sha=file_hash(ROOT/'sources/NVIDIA_LICENSE.txt')
 for alias,spec in specs.items():
  api_path=ROOT/'sources'/f'{alias}_api.json';api=json.loads(api_path.read_text());files={x['rfilename']:x for x in api['siblings']};assert api['sha']==spec['revision']and api['id']==spec['repo_id']and not api.get('private')and not api.get('gated')and file_hash(api_path)==spec['API_sha256'];lfs=files[spec['weight_name']]['lfs'];assert lfs['sha256']==spec['expected_sha256']and lfs['size']==spec['expected_size']and file_hash(spec['weight_path'])==spec['expected_sha256']and Path(spec['weight_path']).stat().st_size==spec['expected_size']and spec['license_sha256']==license_sha
  for name,key in [('config.json','config_path'),('preprocessor_config.json','preprocessor_path'),('README.md','README_path')]:
   data=Path(spec[key]).read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==files[name]['blobId']
  config=json.loads(Path(spec['config_path']).read_text());assert config['hidden_sizes']==[32,64,160,256]and config['depths']==[2,2,2,2]and len(config['id2label'])==(1000 if alias=='mit' else 150)
 events=json.loads((ROOT/'network_events.json').read_text());budget=json.loads((ROOT/'network_budget.json').read_text());assert len(events)<=256 and all(b['started_unix']-a['started_unix']>=1-1e-3 for a,b in zip(events,events[1:]))and all(row['started_unix']<=budget['deadline_unix']for row in events)
 for row in events:assert row['host']in('huggingface.co','hf.co','raw.githubusercontent.com')or row['host'].endswith('.huggingface.co')or row['host'].endswith('.hf.co')
 ledger=json.loads((ROOT/'weight_requests.json').read_text());assert sum(r['received_bytes']for r in ledger)<=64*1024**2
 for alias,spec in specs.items():
  rows=[r for r in ledger if r['alias']==alias];assert len(rows)<=128;pos=0;parts=[];etag=None;modified=None
  for row in rows:
   assert row['range_start']==pos
   if row.get('path'):data=Path(row['path']).read_bytes();assert len(data)==row['received_bytes']and file_hash(row['path'])==row['sha256'];parts.append(data)
   else:assert row['received_bytes']==0
   if row.get('headers'):
    h=row['headers']
    if row['status']==200:assert pos==0 and int(h['Content-Length'])==spec['expected_size']
    else:
     start,endtotal=h['Content-Range'].removeprefix('bytes ').split('-');end,total=endtotal.split('/');assert row['status']==206 and int(start)==pos and int(total)==spec['expected_size']and int(h['Content-Length'])==int(end)-int(start)+1
    if etag is None:etag=h['ETag'];modified=h['Last-Modified']
    else:assert etag==h['ETag']and(modified is None or modified==h['Last-Modified'])
   pos+=row['received_bytes']
  assert pos==spec['expected_size']and b''.join(parts)==Path(spec['weight_path']).read_bytes()
 clip=json.loads((ROOT/'source_clip.json').read_text());assert clip['clip']['video_id']==258;frames=[]
 for path in clip['clip']['frames']:
  with Image.open(path)as image:array=np.array(image.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),dtype=np.float32)
  frames.append(torch.from_numpy(array).permute(2,0,1)/255)
 video=torch.stack(frames).cuda()
 with np.load(clip['query_cache'])as a:xy=a['xy'][np.arange(0,2048,32)].copy()
 pre=json.loads((ROOT/'preflight.json').read_text());assert file_hash(ROOT/'preflight.json')==s['preflight_sha256'];parameter_shapes={};manual_differences={}
 for alias in ('mit','ade'):
  model=features.load_official(alias);parameter_shapes[alias]={name:tuple(value.shape)for name,value in model.segformer.state_dict().items()};assert all(not value.requires_grad for value in model.parameters())and all(not module.training for module in model.modules());maps=manual_maps(model,alias,video);single=manual_maps(model,alias,video[:1])
  for arm,map_ in maps.items():
   r=pre['results'][arm];assert file_hash(r['output_path'])==r['output_sha256']and list(map_.shape)==r['map_shape'];out=torch.empty(64,256,device='cuda')
   with torch.inference_mode():
    for t in sorted(set(map(int,xy[:,2]))):
     rows=np.flatnonzero(xy[:,2]==t);grid=(((torch.from_numpy(xy[rows,:2]).float().cuda()+.5)/128)*2-1)[None,None];out[torch.from_numpy(rows).cuda()]=F.grid_sample(map_[t:t+1],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
   quant=out.half().cpu().numpy()
   with np.load(r['output_path'])as a:assert np.array_equal(a['xy'],xy)and np.array_equal(a['decoded'],quant)and a['decoded'].dtype==np.float16 and np.isfinite(a['decoded']).all()
   diff=(map_[0]-single[arm][0]).abs();assert abs(float(diff.max())-r['single_vs_T8_first_frame_max_difference'])<1e-6 and abs(float(diff.mean())-r['single_vs_T8_first_frame_mean_difference'])<1e-6;assert abs(float((out-out.half().float()).abs().max())-r['FP16_quantization_max_difference'])<1e-6 and sum(v.numel()for v in model.parameters())==r['cost']['parameters']and r['cost']['seconds']>0;manual_differences[arm]=float(diff.max());print('verified independent SegFormer source feature',arm,flush=True)
  del model;torch.cuda.empty_cache()
 assert parameter_shapes['mit']==parameter_shapes['ade'];assert not pre['training_updates']and not pre['new_confirmation_groups']and pre['no_GT_labels_read_or_scored']
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['all_original_inputs_unchanged']and s['no_OVIS_head_training_or_GT_scoring']and s['new_confirmation_groups']==0 and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'both_official_revisions_LFS_sizes_SHA256_and_git_blob_configs_verified':True,'all_anonymous_HTTP_segments_budget_and_hashes_verified':True,'private_env_pins_PyPI_wheels_and_parent_environment_unchanged':True,'MIT_and_ADE_encoder_state_shapes_identical':True,'all3_source64_point_features_independently_recomputed':True,'ADE_decoder_manually_rebuilt_from_frozen_multiscale_modules':True,'T1_vs_T8_max_differences_retained':manual_differences,'transformers_model_definition_sha256':file_hash(inspect.getfile(transformers.SegformerForImageClassification)),'all_original_inputs_unchanged':True,'no_training_or_new_confirmation':True});print('DONE independent SegFormer/MIT isolated resource verification',flush=True)
if __name__=='__main__':main()
