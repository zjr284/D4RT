"""Offline independent audit of bounded partial COCO RGB acquisition."""
from collections import Counter
import hashlib,io,json,struct
from pathlib import Path
import numpy as np
from PIL import Image
from semantic import coco_cohort as work
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 p=json.loads((ROOT/'protocol.json').read_text());m=json.loads((ROOT/'manifest.json').read_text());records=json.loads((ROOT/'image_records.json').read_text());requests=json.loads((ROOT/'requests.json').read_text());failure=json.loads((ROOT/'failure.json').read_text())
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert len(records)==51 and len(requests)==71 and not(ROOT/'materials.json').exists()and not(ROOT/'frozen_eligibility.json').exists()and not Path('semantic_artifacts/coco_semantic_confirmation').exists()
 assert [r['image_id']for r in records]==[im['id']for im in m['candidate'][:51]]and all(r['role']=='candidate'for r in records)
 assert m['candidate'][51]['id']==460379 and failure['message']=='Image 460379 not complete after at most two requests; no replacement'
 source=json.loads((work.META/'instances_val2017.json').read_bytes());images={im['id']:im for im in source['images']};licenses={l['id']:l for l in source['licenses']};groups={}
 for row in requests:groups.setdefault(row['image_id'],[]).append(row)
 assert all(len(rows)<=2 for rows in groups.values())and len(requests)<=p['max_requests']
 for r in records:
  raw=Path(r['path']).read_bytes();im=images[r['image_id']];rows=groups[r['image_id']];parts=[];position=0;total=None;version=None
  for row in rows:
   assert row['range_start']==position
   if row.get('path'):
    data=Path(row['path']).read_bytes();assert len(data)==row['received_bytes']and file_hash(row['path'])==row['sha256'];h=row['headers']
    if position==0:assert row['status']==200;total=int(h['Content-Length']);version=row['version']
    else:
     assert row['status']==206 and h['Content-Range']==f'bytes {position}-{total-1}/{total}'and row['version']==version
    position+=len(data);parts.append(data)
   else:assert row['received_bytes']==0 and row.get('error_type')
  assert raw==b''.join(parts)and len(raw)==position==total and hashlib.sha256(raw).hexdigest()==r['file_sha256']and len(raw)<=p['per_image_byte_cap']
  if r['S3_ETag_MD5_verified']:assert hashlib.md5(raw).hexdigest()==r['S3_ETag'].strip('"')
  assert r['license_id']==im['license']and r['license']==licenses[im['license']]
  with Image.open(io.BytesIO(raw))as image:
   rgb=image.convert('RGB');rgb.load();assert rgb.size==(im['width'],im['height'])==(r['width'],r['height']);assert hashlib.sha256(struct.pack('<II',*rgb.size)+rgb.tobytes()).hexdigest()==r['pixel_sha256'];g=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.BILINEAR).resize((9,8),Image.Resampling.BILINEAR));bits=''.join('1'if g[y,x+1]>g[y,x]else'0'for y in range(8)for x in range(8));assert f'{int(bits,2):016x}'==r['dhash64']
 assert [r['received_bytes']for r in groups[460379]]==[0,0]and '[Errno 104]'in groups[460379][0]['error']and '[Errno 111]'in groups[460379][1]['error']
 received=sum(r['received_bytes']for r in requests);assert received==9220422 and received<=p['byte_cap']
 for arm in work.ARMS:
  for seed in range(3):assert file_hash(work.ckpt(arm,seed))==p['models'][arm]['checkpoint_sha256'][str(seed)]
 errors=Counter(r['error_type']for r in requests if r.get('error_type'))
 summary={'protocol':p,'manifest_sha256':file_hash(ROOT/'manifest.json'),'scope':'partial RGB acquisition failure, not complete cohort or model evaluation','planned_candidate':80,'planned_reserve':80,'completed_candidate':51,'completed_reserve':0,'missing_images':109,'received_RGB_bytes':received,'GET_attempts':len(requests),'request_error_counts':dict(errors),'failed_image_id':460379,'last_two_requests_zero_body':True,'failure':failure,'all_completed_ETag_MD5_verified':all(r['S3_ETag_MD5_verified']for r in records),'image_records_sha256':file_hash(ROOT/'image_records.json'),'requests_sha256':file_hash(ROOT/'requests.json'),'no_replacement_or_GT_or_model_evaluation':True,'duplicate_screen_not_completed':True,'deployment_unchanged':'native_gt'}
 atomic_json(ROOT/'summary.json',summary);atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all51_completed_JPEG_parts_MD5_pixels_dimensions_and_licenses_recomputed':True,'all71_attempts_and_failed_boundary_rechecked':True,'all21_heads_and_frozen_inputs_unchanged':True,'no_false_RGB_eligibility_or_confirmation':True})
 print('VERIFIED partial COCO acquisition',len(records),len(requests),received,dict(errors),flush=True)
if __name__=='__main__':main()
