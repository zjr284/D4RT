"""Bounded official ViT-B/14 resource acquisition and fixed RGB preflight."""
import gc,json,os,sys,time,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_native_rgb_control as rgb
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_backbone_capacity')
WEIGHTS=Path('semantic_artifacts/weights/dinov2_vitb14_pretrain.pth')
URL='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
VENDOR=rgb.prior.resolution.VENDOR

def load_b14():
 os.environ['XFORMERS_DISABLED']='1';sys.path.insert(0,str(VENDOR.resolve()))
 from dinov2.hub.backbones import dinov2_vitb14
 model=dinov2_vitb14(pretrained=False);model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
 return model.cuda().eval().requires_grad_(False)
@torch.inference_mode()
def tokens_b14(model,video):
 mean=torch.tensor([.485,.456,.406],device=video.device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=video.device)[None,:,None,None]
 image=(F.interpolate(video.float(),size=(448,448),mode='bicubic',align_corners=False)-mean)/std
 with torch.autocast('cuda',dtype=torch.bfloat16):tokens=model.forward_features(image)['x_norm_patchtokens']
 assert tokens.shape==(len(video),1024,768)and torch.isfinite(tokens).all()
 return tokens.reshape(len(video),32,32,768).permute(0,3,1,2)
def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'resource_preflight.json').exists():raise RuntimeError('Completed resource preflight exists')
 record={'url':URL,'maximum_bytes':400*1024*1024,'max_requests':1,'socket_timeout_seconds':60,'total_timeout_seconds':300,'protocol_sha256':file_hash('semantic/OVIS_DINO_BACKBONE_CAPACITY_PROTOCOL.md'),'resource_code_sha256':file_hash('semantic/dino_backbone_resource.py')}
 atomic_json(ROOT/'resource_protocol.json',record)
 if not WEIGHTS.exists():
  tmp=WEIGHTS.with_suffix('.partial.pth')
  if tmp.exists():raise RuntimeError('Prior partial exists; review before another request')
  n=0;t=time.monotonic()
  with urllib.request.urlopen(URL,timeout=60)as response:
   record['http_status']=response.status;record['content_length']=response.headers.get('Content-Length');assert response.status==200
   with tmp.open('xb')as out:
    while True:
     chunk=response.read(1024*1024)
     if not chunk:break
     n+=len(chunk);assert n<=record['maximum_bytes']and time.monotonic()-t<300;out.write(chunk)
     if n%(32*1024*1024)==0:print('official B14 received MiB',n//1024**2,flush=True)
  if record['content_length']is not None:assert n==int(record['content_length'])
  tmp.replace(WEIGHTS);record['download_bytes']=n;record['download_seconds']=time.monotonic()-t
 else:record['reused_existing_weight']=True
 record['weight_sha256']=file_hash(WEIGHTS);record['weight_bytes']=WEIGHTS.stat().st_size;atomic_json(ROOT/'acquisition.json',record)
 model=load_b14();m=json.loads((rgb.ROOT/'manifest.json').read_text());clip=m['val'][0];video=rgb.load_native_video(clip).cuda()
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();tokens=tokens_b14(model,video);torch.cuda.synchronize()
 report={'acquisition_sha256':file_hash(ROOT/'acquisition.json'),'weight_sha256':record['weight_sha256'],'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip(),'video_id':clip['video_id'],'rgb_shape':list(video.shape),'patch_shape':list(tokens.shape),'backbone_parameters':sum(p.numel()for p in model.parameters()),'seconds':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'device_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0),'preflight_reads_RGB_only':True}
 atomic_json(ROOT/'resource_preflight.json',report);print('DONE B14 preflight',report,flush=True);del model,video,tokens;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'resource_failure.json',{'type':type(e).__name__,'message':str(e)});raise
