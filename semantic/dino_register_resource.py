"""Bounded official register B14 weights and fixed RGB-only resource checks."""
import fcntl,gc,hashlib,json,os,re,shutil,sys,time,urllib.request,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_backbone_resource as prior
from semantic import dino_backbone_training as capacity
from semantic.coco_endpoint_probe import NoRedirect
from semantic.semantic_feature_control import patch_at_queries,save_npz
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_register_resource');WEIGHTS=ROOT/'weights/dinov2_vitb14_reg4_pretrain.pth';URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth';VENDOR=prior.VENDOR

def load_register():
 os.environ['XFORMERS_DISABLED']='1';sys.path.insert(0,str(VENDOR.resolve()))
 from dinov2.hub.backbones import dinov2_vitb14_reg
 model=dinov2_vitb14_reg(pretrained=False);model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
 assert model.num_register_tokens==4 and model.interpolate_antialias and model.interpolate_offset==0
 return model.cuda().eval().requires_grad_(False)

def freeze():
 m=json.loads((capacity.ROOT/'manifest.json').read_text());clip=m['val'][0];assert clip['video_id']==258
 paths=sorted(VENDOR.rglob('*.py'))+[VENDOR/n for n in ('LICENSE','MODEL_CARD.md','README.md')]+[prior.WEIGHTS,capacity.ROOT/'manifest.json',capacity.fp('val',258),Path('semantic/dino_register_resource.py'),Path('semantic/OVIS_DINO_REGISTER_RESOURCE_PROTOCOL.md')]+[Path(x)for x in clip['frames']]
 assert file_hash(prior.WEIGHTS)=='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
 p={'round':89,'official_url':URL,'max_HEAD':1,'max_GET':4,'max_received_bytes':400*1024**2,'socket_timeout_seconds':30,'total_deadline_seconds':900,'RGB_video_id':258,'no_GT_classification_or_new_confirmation':True,'input_sha256':{str(path):file_hash(path)for path in paths}}
 assert shutil.disk_usage(ROOT).free>=1024**3
 atomic_json(ROOT/'protocol.json',p);atomic_json(ROOT/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,'disk_free_bytes':shutil.disk_usage(ROOT).free})
 return p,clip

def acquire(p):
 assert not WEIGHTS.exists()and not(ROOT/'requests.json').exists()
 opener=urllib.request.build_opener(NoRedirect());ledger=[];start=time.time();deadline=start+900;atomic_json(ROOT/'budget.json',{'started_unix':start,'deadline_unix':deadline})
 def open_request(method,headers):
  row={'method':method,'url':URL,'request_headers':headers,'started_unix':time.time(),'received_bytes':0};ledger.append(row);atomic_json(ROOT/'requests.json',ledger)
  response=opener.open(urllib.request.Request(URL,method=method,headers={'Accept-Encoding':'identity',**headers}),timeout=30);row['status']=response.status;row['headers']={k:response.headers.get(k)for k in ('Content-Length','Content-Range','Content-Encoding','ETag','Last-Modified')};atomic_json(ROOT/'requests.json',ledger);return response,row
 response,head=open_request('HEAD',{})
 with response:
  assert response.status==200;total=int(head['headers']['Content-Length']);assert 0<total<=p['max_received_bytes'];version={k:head['headers'][k]for k in ('ETag','Last-Modified')}
 current=0;parts=[];received=0
 for attempt in range(4):
  if current==total:break
  assert time.time()<deadline;headers={}if current==0 else{'Range':f'bytes={current}-{total-1}'};row=None;path=ROOT/f'segment{attempt}.data';before=current
  try:
   response,row=open_request('GET',headers)
   with response:
    h=row['headers'];assert h['Content-Encoding']in(None,'identity')and all(h[k]==v for k,v in version.items()if v is not None)
    if current==0:assert response.status==200 and int(h['Content-Length'])==total
    else:assert response.status==206 and h['Content-Range']==f'bytes {current}-{total-1}/{total}'and int(h['Content-Length'])==total-current
    row['range_start']=current;row['requested_end']=total-1
    with path.open('xb')as output:
     while current<total:
      chunk=response.read(min(65536,total-current))
      if not chunk:break
      assert received+len(chunk)<=p['max_received_bytes']and time.time()<deadline;output.write(chunk);received+=len(chunk);current+=len(chunk);row['received_bytes']+=len(chunk)
      if current//(32*1024**2)>(current-len(chunk))//(32*1024**2):print('official registers received MiB',current//1024**2,flush=True);atomic_json(ROOT/'requests.json',ledger)
  except Exception as e:
   if row is None:row=ledger[-1]
   row['error_type']=type(e).__name__;row['error']=str(e)
   if getattr(e,'code',None)in(400,401,403,404,429):atomic_json(ROOT/'requests.json',ledger);raise
  finally:
   if path.exists():row['path']=str(path);row['sha256']=file_hash(path);parts.append(path)
   atomic_json(ROOT/'requests.json',ledger)
  print('register resource segment',attempt,current-before,'remaining',total-current,flush=True)
  gets=[x for x in ledger if x['method']=='GET']
  if len(gets)>=2 and all(x['received_bytes']==0 for x in gets[-2:]):raise RuntimeError('Two consecutive zero-byte downloads; preserve bounded failure')
 assert current==total,'Bounded download incomplete; preserve all segments'
 WEIGHTS.parent.mkdir(parents=True,exist_ok=True);partial=WEIGHTS.with_suffix('.partial.pth')
 with partial.open('xb')as out:
  for path in parts:
   with path.open('rb')as handle:shutil.copyfileobj(handle,out)
 assert partial.stat().st_size==total
 with zipfile.ZipFile(partial)as z:assert z.testzip()is None;entries=len(z.infolist())
 etag=(version['ETag']or'').strip('"');md5=None
 if re.fullmatch('[0-9a-fA-F]{32}',etag):
  digest=hashlib.md5()
  with partial.open('rb')as f:
   for chunk in iter(lambda:f.read(1024**2),b''):digest.update(chunk)
  assert digest.hexdigest()==etag.lower();md5=True
 partial.replace(WEIGHTS);atomic_json(ROOT/'acquisition.json',{'url':URL,'weight_bytes':total,'weight_sha256':file_hash(WEIGHTS),'recorded_SHA_is_not_author_signature':True,'GET_requests':sum(x['method']=='GET'for x in ledger),'HEAD_requests':1,'received_bytes':received,'requests_sha256':file_hash(ROOT/'requests.json'),'zip_entries':entries,'all_ZIP_CRC_passed':True,'single_part_ETag_MD5_verified':md5})

def preflight(clip):
 with np.load(capacity.fp('val',258))as a:xy=a['xy'][::32].copy();baseline=a['decoded'][::32].copy()
 assert len(xy)==64;video=prior.rgb.load_native_video(clip).cuda();reports={}
 for alias in ('plain','registers'):
  model=prior.load_b14()if alias=='plain'else load_register();rows=[];features={}
  for n in (1,8):
   image=video[:n];mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];points=xy[xy[:,2]<n]
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=model.forward_features((F.interpolate(image.float(),size=(448,448),mode='bicubic',align_corners=False)-mean)/std)
   patches=output['x_norm_patchtokens'];registers=output['x_norm_regtokens'];assert patches.shape==(n,1024,768)and registers.shape==(n,0 if alias=='plain'else 4,768)and torch.isfinite(patches).all()and torch.isfinite(registers).all()
   with torch.inference_mode():point_features=patch_at_queries(patches.reshape(n,32,32,768).permute(0,3,1,2),points).float().cpu().numpy()
   torch.cuda.synchronize();elapsed=time.perf_counter()-start;quantized=point_features.astype(np.float16);features[n]=quantized
   save_npz(ROOT/'preflight_features'/f'{alias}_T{n}.npz',xy=points,decoded=quantized,patchtokens=patches.float().cpu().numpy().astype(np.float16),registertokens=registers.float().cpu().numpy().astype(np.float16))
   rows.append({'frames':n,'patch_shape':list(patches.shape),'register_shape':list(registers.shape),'class_shape':list(output['x_norm_clstoken'].shape),'point_count':len(points),'FP16_quantization_maximum_difference':float(np.abs(point_features-quantized.astype(np.float32)).max()),'seconds':elapsed,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()})
  if alias=='plain':assert np.array_equal(features[8],baseline)
  reports[alias]={'backbone_parameters':sum(a.numel()for a in model.parameters()),'register_tokens':model.num_register_tokens,'interpolate_antialias':model.interpolate_antialias,'interpolate_offset':model.interpolate_offset,'runs':rows,'T1_T8_source0_maximum_FP16_difference':float(np.abs(features[1].astype(np.float32)-features[8][xy[:,2]==0].astype(np.float32)).max())}
  del model;gc.collect();torch.cuda.empty_cache()
 atomic_json(ROOT/'preflight.json',reports)

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists()or(ROOT/'requests.json').exists():raise RuntimeError('Existing register resource run; inspect instead of repeat')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  p,clip=freeze();acquire(p);preflight(clip)
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'acquisition':json.loads((ROOT/'acquisition.json').read_text()),'preflight':json.loads((ROOT/'preflight.json').read_text()),'preflight_sha256':file_hash(ROOT/'preflight.json'),'feature_hashes':{str(path):file_hash(path)for path in sorted((ROOT/'preflight_features').glob('*.npz'))},'new_heads':0,'new_confirmations':0,'deployment_unchanged':'native_gt'});print('DONE official register B14 RGB-only resource preflight',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
