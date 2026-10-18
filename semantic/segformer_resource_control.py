"""Bounded anonymous official weights and isolated SegFormer dependency setup."""
import fcntl,hashlib,importlib.metadata,inspect,json,re,subprocess,sys,sysconfig,time,urllib.request,venv
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
import torch,numpy,PIL,huggingface_hub
from semantic import coco_spatial_confirmation as previous
from semantic import dino_backbone_training as capacity
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/segformer_resource_control');REPOS={'mit':'nvidia/mit-b0','ade':'nvidia/segformer-b0-finetuned-ade-512-512'};PACKAGES=('torch','numpy','pillow','torchvision','huggingface-hub','pycocotools','matplotlib');ENV=ROOT/'env'

def environment():
 return {name:importlib.metadata.version(name)for name in PACKAGES}

def freeze():
 s=json.loads((previous.ROOT/'summary.json').read_text());assert file_hash(previous.ROOT/'summary.json')==json.loads((previous.ROOT/'verification.json').read_text())['summary_sha256'];m=json.loads((capacity.ROOT/'manifest.json').read_text());clip=m['val'][0];assert clip['video_id']==258
 paths=[previous.ROOT/'summary.json',previous.ROOT/'verification.json',capacity.ROOT/'manifest.json',capacity.fp('val',258),Path('semantic/segformer_resource_control.py'),Path('semantic/segformer_features.py'),Path('semantic/segformer_resource_worker.py'),Path('semantic/OVIS_SEGFORMER_RESOURCE_PROTOCOL.md'),Path('semantic/semantic_feature_control.py'),Path(torch.__file__),Path(numpy.__file__),Path(PIL.__file__),Path(huggingface_hub.__file__)]+[Path(x)for x in clip['frames']]
 p={'round':81,'repositories':REPOS,'purpose':'resources and RGB-only preflight, no OVIS training or GT scores','transformers_pin':'4.57.6','private_hub_pin':'0.36.0','max_HTTP_GET':256,'max_model_bytes':64*1024**2,'max_each_model_bytes':32*1024**2,'max_metadata_bytes':8*1024**2,'network_deadline_seconds':900,'max_segments_each_model':128,'minimum_GET_start_interval':1,'pip_timeout_seconds':300,'original_environment':environment(),'original_site_packages':sysconfig.get_path('purelib'),'input_sha256':{str(x):file_hash(x)for x in paths}}
 if(ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==p
 else:atomic_json(ROOT/'protocol.json',p)
 atomic_json(ROOT/'source_clip.json',{'clip':clip,'query_cache':str(capacity.fp('val',258))});return p

def install(p):
 if not(ENV/'bin/python').exists():venv.EnvBuilder(with_pip=True,system_site_packages=False).create(ENV)
 site=ENV/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages';assert site.exists();(site/'parent_d4rt_readonly.pth').write_text(p['original_site_packages']+'\n')
 if not(ROOT/'env_install.json').exists():
  with(ROOT/'env_install.log').open('w')as log:subprocess.run([str(ENV/'bin/python'),'-m','pip','--isolated','install','--index-url','https://pypi.org/simple','--only-binary=:all:','--no-cache-dir','--disable-pip-version-check','--timeout','30','--retries','1','--report',str(ROOT/'env_install.json'),'transformers==4.57.6','huggingface_hub==0.36.0'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 assert environment()==p['original_environment']
 code="import json,transformers,safetensors,torch;from transformers import SegformerForImageClassification,SegformerForSemanticSegmentation;print(json.dumps({'transformers':transformers.__version__,'torch':torch.__version__,'safetensors':safetensors.__version__}))"
 with(ROOT/'env_import.log').open('w')as log:subprocess.run([str(ENV/'bin/python'),'-c',code],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
 print('Isolated dependencies imported; parent environment unchanged',flush=True)

class Network:
 def __init__(self,p):
  self.p=p;self.events=json.loads((ROOT/'network_events.json').read_text())if(ROOT/'network_events.json').exists()else[];self.metadata_bytes=0;self.model_bytes=sum(r.get('received_bytes',0)for r in json.loads((ROOT/'weight_requests.json').read_text()))if(ROOT/'weight_requests.json').exists()else 0;bp=ROOT/'network_budget.json'
  if not bp.exists():atomic_json(bp,{'started_unix':time.time(),'deadline_unix':time.time()+900})
  self.budget=json.loads(bp.read_text());owner=self
  class Redirect(urllib.request.HTTPRedirectHandler):
   def redirect_request(self,req,fp,code,msg,headers,newurl):
    owner.register(newurl,'redirect');return super().redirect_request(req,fp,code,msg,headers,newurl)
  self.opener=urllib.request.build_opener(Redirect())
 def register(self,url,kind):
  u=urlsplit(url);host=u.hostname or'';assert u.scheme=='https'and (host in ('huggingface.co','hf.co','raw.githubusercontent.com')or host.endswith('.huggingface.co')or host.endswith('.hf.co'))and not u.username and not u.password;assert len(self.events)<256 and time.time()<self.budget['deadline_unix']
  delay=1-(time.time()-(self.events[-1]['started_unix']if self.events else self.budget['started_unix']))
  if delay>0:time.sleep(delay)
  self.events.append({'kind':kind,'host':host,'path':u.path,'started_unix':time.time()});atomic_json(ROOT/'network_events.json',self.events)
 def open(self,url,headers=None):
  self.register(url,'GET');return self.opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity',**(headers or{})}),timeout=30)
 def small(self,url,path):
  if path.exists():return path.read_bytes()
  try:
   with self.open(url)as response:assert response.status==200;data=response.read(2*1024**2+1);assert len(data)<=2*1024**2
  except Exception as e:raise RuntimeError(f'Official metadata retrieval failed {path.name}: {type(e).__name__} HTTP{getattr(e,"code",None)}')from None
  self.metadata_bytes+=len(data);assert self.metadata_bytes<=8*1024**2;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return data

def resources(p):
 net=Network(p);license_path=ROOT/'sources/NVIDIA_LICENSE.txt';license_bytes=net.small('https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE',license_path);assert b'non-commercially'in license_bytes and b'research or evaluation'in license_bytes;specs={}
 for alias,repo in REPOS.items():
  api_path=ROOT/'sources'/f'{alias}_api.json';api=json.loads(net.small(f'https://huggingface.co/api/models/{repo}?blobs=true',api_path));assert not api.get('private')and not api.get('gated');revision=api['sha'];assert re.fullmatch('[0-9a-f]{40}',revision);files={a['rfilename']:a for a in api['siblings']};weight_name='model.safetensors'if'model.safetensors'in files else'pytorch_model.bin';record=files[weight_name];lfs=record['lfs'];assert 0<lfs['size']<=32*1024**2 and re.fullmatch('[0-9a-f]{64}',lfs['sha256']);config_paths={}
  for name in ('config.json','preprocessor_config.json','README.md'):
   path=ROOT/'sources'/alias/name;data=net.small(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',path);assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==files[name]['blobId'];config_paths[name]=str(path)
  config=json.loads(Path(config_paths['config.json']).read_text());processor=json.loads(Path(config_paths['preprocessor_config.json']).read_text());assert config['hidden_sizes']==[32,64,160,256]and config['depths']==[2,2,2,2];assert processor['image_mean']==[.485,.456,.406]and processor['image_std']==[.229,.224,.225]
  specs[alias]={'repo_id':repo,'revision':revision,'weight_name':weight_name,'weight_format':'safetensors'if weight_name.endswith('.safetensors')else'pytorch_bin_weights_only','expected_size':lfs['size'],'expected_sha256':lfs['sha256'],'config_path':config_paths['config.json'],'preprocessor_path':config_paths['preprocessor_config.json'],'README_path':config_paths['README.md'],'API_sha256':file_hash(api_path),'config_sha256':file_hash(config_paths['config.json']),'preprocessor_sha256':file_hash(config_paths['preprocessor_config.json']),'weight_path':str(ROOT/'weights'/alias/weight_name),'license_sha256':file_hash(license_path)}
 atomic_json(ROOT/'model_specs.json',specs)
 ledger=json.loads((ROOT/'weight_requests.json').read_text())if(ROOT/'weight_requests.json').exists()else[]
 for alias,spec in specs.items():
  destination=Path(spec['weight_path'])
  if destination.exists():assert destination.stat().st_size==spec['expected_size']and file_hash(destination)==spec['expected_sha256'];continue
  rows=[r for r in ledger if r['alias']==alias];pieces=[];pos=0
  for row in rows:
   if row.get('path'):raw=Path(row['path']).read_bytes();assert len(raw)==row['received_bytes']and file_hash(row['path'])==row['sha256'];pieces.append(raw);pos+=len(raw)
  public=f'https://huggingface.co/{spec["repo_id"]}/resolve/{spec["revision"]}/{spec["weight_name"]}';current=public;etag=None;modified=None
  for attempt in range(len(rows),128):
   if pos==spec['expected_size']:break
   row={'alias':alias,'attempt':attempt,'public_source_url':public,'range_start':pos,'received_bytes':0};ledger.append(row);atomic_json(ROOT/'weight_requests.json',ledger);chunk_path=ROOT/'downloads'/f'{alias}_{attempt}.data';chunk_path.parent.mkdir(exist_ok=True)
   try:
    with net.open(current,{'Range':f'bytes={pos}-{spec["expected_size"]-1}'}if pos else{})as response:
     current=response.geturl();h={key:response.headers.get(key)for key in ('Content-Length','Content-Range','Content-Encoding','ETag','Last-Modified')};row.update(status=response.status,headers=h,response_host=urlsplit(current).hostname);assert h['Content-Encoding']in(None,'identity')
     if response.status==200:assert pos==0 and int(h['Content-Length'])==spec['expected_size']
     else:
      match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',h['Content-Range']or'');assert response.status==206 and match and int(match[1])==pos and int(match[3])==spec['expected_size']and int(h['Content-Length'])==int(match[2])-int(match[1])+1
     if etag is None:etag=h['ETag'];modified=h['Last-Modified']
     else:assert h['ETag']==etag and (modified is None or h['Last-Modified']==modified)
     atomic_json(ROOT/'weight_requests.json',ledger)
     with chunk_path.open('xb')as out:
      while pos<spec['expected_size']:
       block=response.read(min(65536,spec['expected_size']-pos))
       if not block:break
       assert net.model_bytes+len(block)<=64*1024**2 and time.time()<net.budget['deadline_unix'];out.write(block);pos+=len(block);net.model_bytes+=len(block);row['received_bytes']+=len(block)
   except Exception as e:row.update(error_type=type(e).__name__,HTTP_code=getattr(e,'code',None))
   finally:
    if chunk_path.exists():row.update(path=str(chunk_path),sha256=file_hash(chunk_path));pieces.append(chunk_path.read_bytes())
    atomic_json(ROOT/'weight_requests.json',ledger)
   if row.get('HTTP_code')in(400,401,403,404,429):raise RuntimeError(f'Official model download HTTP{row["HTTP_code"]}; preserve bounded prefix')
   if len(ledger)>=2 and all(r['received_bytes']==0 and r.get('error_type')for r in ledger[-2:]):raise RuntimeError('Two consecutive zero-byte model download failures')
  data=b''.join(pieces);assert pos==len(data)==spec['expected_size']and hashlib.sha256(data).hexdigest()==spec['expected_sha256'];destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data);print('Official weights complete',alias,spec['expected_size'],spec['expected_sha256'],flush=True)
 return specs

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed SegFormer resource experiment exists')
  p=freeze();install(p);specs=resources(p)
  with(ROOT/'preflight.log').open('w')as log:subprocess.run([str(ENV/'bin/python'),'-m','semantic.segformer_resource_worker'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
  assert environment()==p['original_environment']
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'model_specs_sha256':file_hash(ROOT/'model_specs.json'),'model_specs':specs,'env_install_report_sha256':file_hash(ROOT/'env_install.json'),'preflight_sha256':file_hash(ROOT/'preflight.json'),'original_environment_after':environment(),'all_original_inputs_unchanged':True,'no_OVIS_head_training_or_GT_scoring':True,'new_confirmation_groups':0,'deployment_unchanged':'native_gt'});print('DONE official SegFormer/MIT isolated resource experiment',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
