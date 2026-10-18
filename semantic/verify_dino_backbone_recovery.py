"""Offline requests/bytes/ZIP/strict-model and real RGB preflight verification."""
import hashlib,json,os,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from semantic.dino_backbone_recovery import ROOT,PREFIX,WEIGHTS
from semantic import dino_native_rgb_control as rgb
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 req=json.loads((ROOT/'requests.json').read_text());assert file_hash(ROOT/'requests.json')==s['requests_sha256'];assert len(req)<=3 and req[0]['method']=='HEAD'and req[0]['status']==200
 total=int(req[0]['headers']['Content-Length']);current=PREFIX.stat().st_size;received=0;getrows=req[1:];assert len(getrows)<=2
 with WEIGHTS.open('rb')as f,PREFIX.open('rb')as prefix:
  for chunk in iter(lambda:prefix.read(1024*1024),b''):assert f.read(len(chunk))==chunk
  for r in getrows:
   assert r['method']=='GET'and r['status']==206 and r['range_start']==current
   assert r['headers']['Content-Range']==f'bytes {current}-{total-1}/{total}'and int(r['headers']['Content-Length'])==total-current
   for key in ['ETag','Last-Modified']:
    if req[0]['headers'][key]is not None:assert r['headers'][key]==req[0]['headers'][key]
   part=Path(r['local_path']);assert file_hash(part)==r['sha256']and part.stat().st_size==r['received_bytes']
   assert f.read(r['received_bytes'])==part.read_bytes();current+=r['received_bytes'];received+=r['received_bytes']
  assert not f.read(1)
 assert current==total==s['weight_bytes']==WEIGHTS.stat().st_size and received<=p['max_received_bytes']and received==s['received_suffix_bytes']
 assert s['GET_requests']==len(getrows)and s['HEAD_requests']==1 and file_hash(WEIGHTS)==s['weight_sha256']
 with zipfile.ZipFile(WEIGHTS)as z:assert z.testzip()is None and len(z.infolist())==s['zip_entries']
 pre=json.loads((ROOT/'resource_preflight.json').read_text());assert file_hash(ROOT/'resource_preflight.json')==s['preflight_sha256']and pre['weight_sha256']==s['weight_sha256']
 os.environ['XFORMERS_DISABLED']='1';sys.path.insert(0,str(rgb.prior.resolution.VENDOR.resolve()))
 from dinov2.hub.backbones import dinov2_vitb14
 model=dinov2_vitb14(pretrained=False);model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True);model.cuda().eval().requires_grad_(False)
 assert sum(x.numel()for x in model.parameters())==pre['backbone_parameters']==86580480
 clip=json.loads((rgb.ROOT/'manifest.json').read_text())['val'][0];assert clip['video_id']==pre['video_id']==258
 frames=[]
 for path in clip['frames']:
  with Image.open(path)as im:a=np.array(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32)
  frames.append(torch.from_numpy(a).permute(2,0,1)/255)
 video=torch.stack(frames).cuda();mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):tokens=model.forward_features((video-mean)/std)['x_norm_patchtokens']
 assert list(tokens.shape)==[8,1024,768]and torch.isfinite(tokens).all()and pre['patch_shape']==[8,768,32,32]
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'actual_suffix_bytes':received,'requests':len(req),'all_ranges_and_file_segments_compared':True,'full_ZIP_CRC_independently_rechecked':True,'strict_official_structure_and_frozen_8frame_RGB_forward_passed':True,'prefix_and_frozen_inputs_unchanged':True,'no_new_training_or_confirmation':True})
 print('DONE independent B14 recovery verification',received,total,flush=True)
if __name__=='__main__':main()
