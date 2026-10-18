"""Independent fresh cohort, official RGB/ledger and duplicate-screen audit."""
from collections import Counter,defaultdict
import hashlib,io,json,sqlite3,struct
from pathlib import Path
import numpy as np
from PIL import Image
from semantic import coco_spatial_cohort as work
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 p=json.loads((ROOT/'protocol.json').read_text());m=json.loads((ROOT/'manifest.json').read_text());s=json.loads((ROOT/'summary.json').read_text());assert p['round']==79 and s['protocol']==p and s['no_candidate_GT_or_model_evaluation']
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 payload=json.loads((work.META/'instances_val2017.json').read_bytes());mapping={int(k):int(v)for k,v in json.loads((work.META/'summary.json').read_text())['shared_name_mapping'].items()};previous=json.loads((work.old.ROOT/'manifest.json').read_text());excluded={im['id']for role in ('candidate','reserve')for im in previous[role]};images=sorted(payload['images'],key=lambda im:im['id']);rank=[images[i]['id']for i in np.random.default_rng(20261079).permutation(len(images))];byid={im['id']:im for im in images};classes=sorted(set(mapping.values())-{1});pools={c:set()for c in classes}
 for ann in payload['annotations']:
  cid=mapping.get(ann['category_id'])
  if cid in pools and not ann['iscrowd']and ann['area']>0:pools[cid].add(ann['image_id'])
 seen=set(excluded);expected={};union=set().union(*pools.values())
 for role in ('candidate','reserve'):
  chosen=[]
  for cycle in range(2):
   for cid in classes:
    eligible=[i for i in rank if i not in seen and i in pools[cid]]
    if eligible and len(chosen)<40:chosen.append(eligible[0]);seen.add(eligible[0])
  for i in rank:
   if len(chosen)<40 and i not in seen and i in union:chosen.append(i);seen.add(i)
  expected[role]=[byid[i]for i in chosen];assert expected[role]==m[role]
 assert m['seed']==20261079 and m['excluded_previously_evaluated_COCO_ids']==sorted(excluded)
 expected_counts={role:{str(c):sum(im['id']in pools[c]for im in expected[role])for c in classes}for role in expected};assert expected_counts==m['metadata_class_image_counts'];records=json.loads((ROOT/'image_records.json').read_text());assert [r['image_id']for r in records]==[im['id']for im in m['candidate']];licenses={x['id']:x for x in payload['licenses']};requests=json.loads((ROOT/'requests.json').read_text());budget=json.loads((ROOT/'budget.json').read_text());assert len(requests)<=160 and sum(x['received_bytes']for x in requests)<=64*1024**2
 assert all(b['started_unix']-a['started_unix']>=1-1e-3 for a,b in zip(requests,requests[1:]))and all(x['started_unix']<=budget['deadline_unix']for x in requests);byreq=defaultdict(list)
 for row in requests:byreq[row['image_id']].append(row)
 for r,im in zip(records,m['candidate']):
  rows=byreq[r['image_id']];assert 1<=len(rows)<=4;data=[];pos=0;total=None;version=None
  for row in rows:
   assert row['url']=='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+im['file_name']and row['range_start']==pos
   if row.get('path'):
    raw=Path(row['path']).read_bytes();assert len(raw)==row['received_bytes']and file_hash(row['path'])==row['sha256'];data.append(raw)
   else:assert row['received_bytes']==0
   if row.get('headers'):
    h=row['headers'];assert h['Content-Encoding']in(None,'identity')
    if pos==0:assert row['status']==200;total=int(h['Content-Length']);version={key:h[key]for key in ('ETag','Last-Modified')};assert 0<total<=8*1024**2
    else:assert row['status']==206 and h['Content-Range']==f'bytes {pos}-{total-1}/{total}'and int(h['Content-Length'])==total-pos and all(h[key]==value for key,value in version.items()if value is not None)
   pos+=row['received_bytes']
  raw=b''.join(data);assert total==pos==len(raw)and raw==Path(r['path']).read_bytes()and hashlib.md5(raw).hexdigest()==r['S3_ETag'].strip('"').lower()
  with Image.open(io.BytesIO(raw))as image:assert image.format=='JPEG';image.verify()
  with Image.open(io.BytesIO(raw))as image:rgb=image.convert('RGB');rgb.load();width,height=rgb.size;pixel=struct.pack('<II',width,height)+rgb.tobytes();small=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.BILINEAR).resize((9,8),Image.Resampling.BILINEAR));dhash=np.packbits((small[:,1:]>small[:,:-1]).ravel(),bitorder='big').tobytes().hex()
  assert(width,height)==(im['width'],im['height'])==(r['width'],r['height'])and hashlib.sha256(raw).hexdigest()==r['file_sha256']and hashlib.sha256(pixel).hexdigest()==r['pixel_sha256']and dhash==r['dhash64'];assert r['license_id']==im['license']and r['license']==licenses[im['license']]and r['S3_ETag_MD5_verified'];print('verified new official COCO JPEG/segments',r['image_id'],flush=True)
 conn=sqlite3.connect(f'file:{work.old.INDEX/"index.sqlite"}?mode=ro',uri=True);native=[json.loads(x[0])for x in conn.execute('SELECT record FROM frames ORDER BY path')];conn.close();refs=[{'dataset':'OVIS_all42149','path':str(Path('semantic_artifacts/ovis/train')/r['path']),**{key:r[key]for key in ('file_sha256','pixel_sha256','dhash64')}}for r in native];assert len(refs)==42149;seen_paths=set()
 for source in p['previous_LVVIS_RGB_sources']:
  for r in json.loads(Path(source).read_text()):
   if r['path']not in seen_paths:refs.append({'dataset':'available_LVVIS_old_RGB','path':r['path'],**{key:r[key]for key in ('file_sha256','pixel_sha256','dhash64')}});seen_paths.add(r['path'])
 for r in json.loads((work.old.ROOT/'image_records.json').read_text()):refs.append({'dataset':'COCO_previously_evaluated160','path':r['path'],'image_id':r['image_id'],**{key:r[key]for key in ('file_sha256','pixel_sha256','dhash64')}})
 pairs=[]
 def append(left,right,distance):pairs.append({'pair_id':len(pairs),'image_id':left['image_id'],'external_path':left['path'],'external_sha256':left['file_sha256'],'reference_dataset':right.get('dataset','new_candidate_COCO'),'reference_path':right['path'],'reference_sha256':right['file_sha256'],'reference_image_id':right.get('image_id'),'distance':distance,'exact_file':left['file_sha256']==right['file_sha256'],'exact_pixels':left['pixel_sha256']==right['pixel_sha256']})
 for r in records:
  for ref in refs:
   distance=(int(r['dhash64'],16)^int(ref['dhash64'],16)).bit_count()
   if distance<=6:append(r,ref,distance)
 for i,r in enumerate(records):
  for ref in records[i+1:]:
   distance=(int(r['dhash64'],16)^int(ref['dhash64'],16)).bit_count()
   if distance<=6:append(r,ref,distance)
 assert pairs==json.loads((ROOT/'pairs.json').read_text());materials=json.loads((ROOT/'materials.json').read_text());assert materials['reference_counts']==dict(Counter(r['dataset']for r in refs))and materials['pair_count']==len(pairs)and materials['exact_pairs']==sum(r['exact_file']or r['exact_pixels']for r in pairs)and file_hash(ROOT/'materials.json')==s['materials_sha256']
 review=json.loads((ROOT/'rgb_review.json').read_text());elig=json.loads((ROOT/'frozen_eligibility.json').read_text());assert review['materials_sha256']==file_hash(ROOT/'materials.json')and file_hash(ROOT/'rgb_review.json')==s['rgb_review_sha256']and file_hash(ROOT/'frozen_eligibility.json')==s['frozen_eligibility_sha256'];assert set(review['pair_ids_reviewed'])=={row['pair_id']for row in pairs};assert elig['candidate']==[im['id']for im in m['candidate']if im['id']not in review['excluded_image_ids']]and elig['metadata_only_reserve']==[im['id']for im in m['reserve']]
 overview=json.loads((ROOT/'overview.json').read_text());assert file_hash(overview['path'])==overview['sha256']==review['viewed_overview_sha256']and overview['image_ids']==[r['image_id']for r in records]and overview['script_sha256']==file_hash('semantic/coco_spatial_overview.py');assert not any((ROOT/'images'/im['file_name']).exists()for im in m['reserve'])
 source=json.loads((work.calibration.ROOT/'frozen_selection.json').read_text())
 for arm,config in p['models'].items():
  assert config['old_offset']==source['old_offsets'][arm]and config['spatial_offset']==source['spatial_offsets'][arm]and config['checkpoint_sha256']==source['models'][arm]['checkpoint_sha256']
  for seed in range(3):assert file_hash(work.models.ckpt(arm,seed))==config['checkpoint_sha256'][str(seed)]
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert not(ROOT/'queries').exists()and not(ROOT/'features').exists()and not(ROOT/'logits').exists();atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'candidate40_and_metadata_reserve40_reconstructed_excluding160':True,'all40_official_JPEG_MD5_pixel_dhash_license_and_segment_ledgers_verified':True,'received_bytes':sum(r['received_bytes']for r in requests),'GET_requests':len(requests),'duplicate_pairs_independently_recomputed':len(pairs),'all_reference_counts_recomputed':materials['reference_counts'],'single_Codex_RGB_overview_and_eligibility_frozen':True,'all21_heads_and_old_spatial_offsets_verified':True,'all_frozen_inputs_unchanged':True,'no_GT_masks_queries_features_or_model_evaluation':True});print('DONE independent new spatial COCO resource verification',flush=True)
if __name__=='__main__':main()
