"""Independent segment assembly, strict loading and manual final-normalization slicing."""
import hashlib,json,re,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from semantic import dino_register_resource as work
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==89
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert work.URL in(work.VENDOR/'README.md').read_text() and 'Apache'in(work.VENDOR/'LICENSE').read_text()
 rows=json.loads((ROOT/'requests.json').read_text());acq=s['acquisition'];assert acq==json.loads((ROOT/'acquisition.json').read_text())and file_hash(ROOT/'requests.json')==acq['requests_sha256']
 assert rows[0]['method']=='HEAD'and sum(r['method']=='HEAD'for r in rows)==1 and sum(r['method']=='GET'for r in rows)==acq['GET_requests']<=4
 head=rows[0]['headers'];total=int(head['Content-Length']);version={k:head[k]for k in ('ETag','Last-Modified')};budget=json.loads((ROOT/'budget.json').read_text());assert 0<total<=400*1024**2
 current=0;digest=hashlib.sha256();md5=hashlib.md5()
 with work.WEIGHTS.open('rb')as full:
  for r in rows:
   assert r['url']==work.URL and r['started_unix']<=budget['deadline_unix']and r['received_bytes']>=0
   if r['method']=='HEAD':assert r['received_bytes']==0 and r['status']==200;continue
   h=r.get('headers')
   if h:
    assert h['Content-Encoding']in(None,'identity')and all(h[k]==v for k,v in version.items()if v is not None)
    if current==0:assert r['status']==200 and int(h['Content-Length'])==total and r['request_headers']=={}
    else:assert r['status']==206 and h['Content-Range']==f'bytes {current}-{total-1}/{total}'and int(h['Content-Length'])==total-current and r['request_headers']=={'Range':f'bytes={current}-{total-1}'}
   if r.get('path'):
    path=Path(r['path']);assert file_hash(path)==r['sha256']and path.stat().st_size==r['received_bytes'];assert r['range_start']==current
    with path.open('rb')as part:
     for chunk in iter(lambda:part.read(1024**2),b''):assert full.read(len(chunk))==chunk;digest.update(chunk);md5.update(chunk);current+=len(chunk)
   else:assert r['received_bytes']==0
  assert full.read(1)==b''
 assert current==total==work.WEIGHTS.stat().st_size==acq['weight_bytes']==acq['received_bytes']and digest.hexdigest()==acq['weight_sha256']==file_hash(work.WEIGHTS)
 etag=(version['ETag']or'').strip('"')
 if re.fullmatch('[0-9a-fA-F]{32}',etag):assert md5.hexdigest()==etag.lower()and acq['single_part_ETag_MD5_verified']is True
 else:assert acq['single_part_ETag_MD5_verified']is None
 with zipfile.ZipFile(work.WEIGHTS)as z:assert z.testzip()is None and len(z.infolist())==acq['zip_entries']
 assert acq['recorded_SHA_is_not_author_signature']and acq['all_ZIP_CRC_passed']
 manifest=json.loads((work.capacity.ROOT/'manifest.json').read_text());clip=manifest['val'][0];assert clip['video_id']==258
 arrays=[]
 for path in clip['frames']:
  with Image.open(path)as im:arrays.append(np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32))
 video=torch.stack([torch.from_numpy(a).permute(2,0,1)/255 for a in arrays]).cuda()
 with np.load(work.capacity.fp('val',258))as a:xy=a['xy'][::32].copy();legacy=a['decoded'][::32].copy()
 assert file_hash(ROOT/'preflight.json')==s['preflight_sha256'];preflight=json.loads((ROOT/'preflight.json').read_text());assert preflight==s['preflight'];replays=[]
 for alias in ('plain','registers'):
  model=work.prior.load_b14()if alias=='plain'else work.load_register();reg=0 if alias=='plain'else 4;record=preflight[alias]
  assert model.num_register_tokens==reg and record['register_tokens']==reg and model.interpolate_antialias==record['interpolate_antialias']and model.interpolate_offset==record['interpolate_offset']and sum(v.numel()for v in model.parameters())==record['backbone_parameters']
  points={}
  for n,r in zip((1,8),record['runs']):
   selected=xy[xy[:,2]<n];mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    image=(F.interpolate(video[:n],size=(448,448),mode='bicubic',align_corners=False)-mean)/std
    hidden=model.prepare_tokens_with_masks(image)
    for block in model.blocks:hidden=block(hidden)
    normalized=model.norm(hidden);patch=normalized[:,1+reg:];register=normalized[:,1:1+reg];cls=normalized[:,0]
   assert list(patch.shape)==r['patch_shape']==[n,1024,768]and list(register.shape)==r['register_shape']==[n,reg,768]and list(cls.shape)==r['class_shape']
   maps=patch.reshape(n,32,32,768).permute(0,3,1,2).float();values=torch.empty((len(selected),768),dtype=torch.float32,device='cuda')
   with torch.inference_mode():
    for t in sorted(set(map(int,selected[:,2]))):
     rows_ix=np.flatnonzero(selected[:,2]==t);coordinates=torch.from_numpy(selected[rows_ix,:2].astype(np.float32)).cuda();grid=(((coordinates+.5)/128)*2-1)[None,None]
     values[torch.tensor(rows_ix,device='cuda')]=F.grid_sample(maps[t:t+1],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
   float_values=values.cpu().numpy();points[n]=float_values.astype(np.float16);path=ROOT/'preflight_features'/f'{alias}_T{n}.npz';assert file_hash(path)==s['feature_hashes'][str(path)]
   with np.load(path)as a:assert np.array_equal(a['xy'],selected)and np.array_equal(a['decoded'],points[n])and np.array_equal(a['patchtokens'],patch.float().cpu().numpy().astype(np.float16))and np.array_equal(a['registertokens'],register.float().cpu().numpy().astype(np.float16))
   assert r['point_count']==len(selected)and abs(r['FP16_quantization_maximum_difference']-float(np.abs(float_values-points[n].astype(np.float32)).max()))<1e-12 and r['seconds']>0 and r['peak_cuda_allocated_bytes']>0
   replays.append({'alias':alias,'frames':n,'points':len(selected)})
  assert abs(record['T1_T8_source0_maximum_FP16_difference']-float(np.abs(points[1].astype(np.float32)-points[8][xy[:,2]==0].astype(np.float32)).max()))<1e-12
  if alias=='plain':assert np.array_equal(points[8],legacy)
  del model;torch.cuda.empty_cache()
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['new_heads']==0 and s['new_confirmations']==0 and s['deployment_unchanged']=='native_gt'
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all_segments_reassembled_bytewise_and_official_URL_LEDGER_verified':True,'complete_recorded_SHA_ETag_if_MD5_and_all_ZIP_CRC_verified':True,'strict_register_and_plain_weight_loading_verified':True,'independent_manual_blocks_final_norm_register_patch_slicing_and_grid_replays':replays,'plain_T8_reproduces_old64_points_bitwise':True,'all_frozen_inputs_unchanged':True,'no_GT_scoring_training_new_confirmation_or_deployment_change':True});print('DONE independent official register resource verification',flush=True)
if __name__=='__main__':main()
