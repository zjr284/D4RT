"""Single HEAD + at most two bounded suffix requests for official B14."""
import fcntl,gc,hashlib,json,re,shutil,time,urllib.request,zipfile
from pathlib import Path
import torch
from semantic.dino_backbone_resource import URL,WEIGHTS,load_b14,tokens_b14
from semantic import dino_native_rgb_control as rgb
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_backbone_recovery');PRIOR=Path('semantic_artifacts/dino_backbone_capacity');PREFIX=WEIGHTS.with_name('dinov2_vitb14_pretrain.partial.pth');LIMIT=2*1024**2

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists()or(ROOT/'requests.json').exists():raise RuntimeError('Prior recovery exists; do not repeat requests')
  old=json.loads((PRIOR/'summary.json').read_text());assert PREFIX.stat().st_size==old['partial_bytes']and file_hash(PREFIX)==old['partial_sha256']
  inputs={str(x):file_hash(x)for x in [PRIOR/'summary.json',PRIOR/'verification.json',PREFIX,Path('semantic/dino_backbone_recovery.py'),Path('semantic/OVIS_DINO_BACKBONE_RECOVERY_PROTOCOL.md')]}
  p={'round':57,'url':URL,'max_HEAD':1,'max_GET':2,'max_received_bytes':LIMIT,'input_sha256':inputs};atomic_json(ROOT/'protocol.json',p);requests=[];received=0;start=time.monotonic()
  def record_request(method,headers=None):
   req=urllib.request.Request(URL,method=method,headers={'Accept-Encoding':'identity',**(headers or{})});response=urllib.request.urlopen(req,timeout=60)
   row={'method':method,'status':response.status,'headers':{k:response.headers.get(k)for k in ['Content-Length','Content-Range','Content-Encoding','ETag','Last-Modified']},'received_bytes':0};requests.append(row);atomic_json(ROOT/'requests.json',requests);return response,row
  response,head=record_request('HEAD')
  with response:total=int(head['headers']['Content-Length']);assert response.status==200
  current=PREFIX.stat().st_size;assert 0<total-current<=LIMIT
  segments=[]
  for attempt in range(2):
   if current==total:break
   assert time.monotonic()-start<180
   response,row=record_request('GET',{'Range':f'bytes={current}-{total-1}'})
   with response:
    assert response.status==206;h=row['headers'];assert h['Content-Range']==f'bytes {current}-{total-1}/{total}'and h['Content-Encoding']in(None,'identity')
    for key in ['ETag','Last-Modified']:
     if head['headers'][key]is not None:assert h[key]==head['headers'][key]
    assert int(h['Content-Length'])==total-current
    path=ROOT/f'suffix{attempt}.data';row['range_start']=current;row['requested_end']=total-1
    with path.open('xb')as out:
     while True:
      chunk=response.read(min(65536,LIMIT-received+1))
      if not chunk:break
      assert received+len(chunk)<=LIMIT and current+len(chunk)<=total;out.write(chunk);received+=len(chunk);current+=len(chunk);row['received_bytes']+=len(chunk)
      assert time.monotonic()-start<180
    row['sha256']=file_hash(path);row['local_path']=str(path);row['complete_requested_range']=current==total;segments.append(path);atomic_json(ROOT/'requests.json',requests)
   print('B14 suffix',attempt,row['received_bytes'],'remaining',total-current,flush=True)
  assert current==total,'Bounded suffix requests exhausted before full file'
  candidate=ROOT/'candidate.pth'
  with candidate.open('xb')as out:
   for part in [PREFIX,*segments]:
    with part.open('rb')as f:shutil.copyfileobj(f,out)
  assert candidate.stat().st_size==total and file_hash(PREFIX)==old['partial_sha256']
  with zipfile.ZipFile(candidate)as z:assert z.testzip()is None;entries=len(z.infolist())
  h=file_hash(candidate);assert not WEIGHTS.exists();candidate.replace(WEIGHTS)
  model=load_b14();manifest=json.loads((rgb.ROOT/'manifest.json').read_text());clip=manifest['val'][0];video=rgb.load_native_video(clip).cuda();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();tokens=tokens_b14(model,video);torch.cuda.synchronize()
  pre={'video_id':clip['video_id'],'rgb_shape':list(video.shape),'patch_shape':list(tokens.shape),'backbone_parameters':sum(a.numel()for a in model.parameters()),'seconds':time.perf_counter()-t,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0),'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=rgb.prior.resolution.VENDOR,text=True).strip(),'weight_sha256':h,'reads_RGB_only':True};atomic_json(ROOT/'resource_preflight.json',pre)
  for path,hash0 in inputs.items():assert file_hash(path)==hash0
  atomic_json(ROOT/'summary.json',{'protocol':p,'weight_bytes':total,'weight_sha256':h,'zip_entries':entries,'zip_all_CRC_passed':True,'requests_sha256':file_hash(ROOT/'requests.json'),'HEAD_requests':1,'GET_requests':len(segments),'received_suffix_bytes':received,'prefix_preserved':True,'preflight_sha256':file_hash(ROOT/'resource_preflight.json'),'new_heads_trained':0,'new_confirmation_groups':0,'deployment_unchanged':'native_gt'})
  print('DONE bounded B14 recovery',total,h,pre,flush=True);del model,video,tokens;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
