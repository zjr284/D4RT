"""Independent COCO split, all JPEG/request and complete match/review audit."""
from collections import Counter,defaultdict
import hashlib,io,json,sqlite3,struct
from pathlib import Path
from PIL import Image
import numpy as np
from semantic import coco_cohort as work
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 source=json.loads((work.META/'instances_val2017.json').read_bytes());m=json.loads((ROOT/'manifest.json').read_text());assert file_hash(ROOT/'manifest.json')==s['manifest_sha256']
 normalize=lambda name:' '.join(name.lower().replace('_',' ').split());names={normalize(c['name']):c['id']for c in json.loads(work.meta.OVIS.read_bytes())['categories']};mapping={c['id']:names[normalize(c['name'])]for c in source['categories']if normalize(c['name'])in names}
 present=defaultdict(set)
 for a in source['annotations']:
  cid=mapping.get(a['category_id'])
  if cid is not None and cid!=1 and not a['iscrowd']and a['area']>0:present[cid].add(a['image_id'])
 rank=np.random.default_rng(20261065).permutation(np.array(sorted(im['id']for im in source['images']))).tolist();chosen=set();roles={};classes=sorted(set(mapping.values())-{1});images={im['id']:im for im in source['images']}
 for role in ('candidate','reserve'):
  ids=[]
  for round_index in range(5):
   for cid in classes:
    available=[i for i in rank if i in present[cid]and i not in chosen]
    if available and len(ids)<80:ids.append(available[0]);chosen.add(available[0])
  for i in rank:
   if len(ids)<80 and i not in chosen and any(i in present[c]for c in classes):ids.append(i);chosen.add(i)
  roles[role]=ids;assert [images[i]for i in ids]==m[role]and len(ids)==80
 assert not set(roles['candidate'])&set(roles['reserve'])
 counts={role:{str(cid):sum(i in present[cid]for i in roles[role])for cid in classes}for role in roles};assert counts==m['role_class_image_counts']==s['metadata_role_class_image_counts']and {str(c):len(present[c])for c in classes}==m['metadata_pool_sizes']
 requests=json.loads((ROOT/'requests.json').read_text());records=json.loads((ROOT/'image_records.json').read_text());materials=json.loads((ROOT/'materials.json').read_text());assert file_hash(ROOT/'materials.json')==s['materials_sha256']and file_hash(ROOT/'image_records.json')==materials['image_records_sha256']
 assert len(requests)==s['actual_GET_requests']<=320 and sum(r['received_bytes']for r in requests)==s['received_RGB_bytes']<=256*1024**2 and len(records)==s['images_acquired']==160
 assert [(r['role'],r['image_id'])for r in records]==[(role,i)for role in ('candidate','reserve')for i in roles[role]]
 licenses={l['id']:l for l in source['licenses']};requestgroups=defaultdict(list)
 for row in requests:requestgroups[row['image_id']].append(row)
 for r in records:
  im=images[r['image_id']];rows=requestgroups[r['image_id']];assert 1<=len(rows)<=2;parts=[];position=0;total=None;version=None
  for attempt,row in enumerate(rows):
   assert row['attempt']==attempt and row['range_start']==position and row['role']==r['role']and row['url']==f'https://s3.amazonaws.com/images.cocodataset.org/val2017/{im["file_name"]}'
   if row.get('path'):
    data=Path(row['path']).read_bytes();assert len(data)==row['received_bytes']and file_hash(row['path'])==row['sha256'];h=row['headers'];assert h['Content-Encoding']in(None,'identity')
    if position==0:assert row['status']==200 and int(h['Content-Length'])==row['total_bytes'];total=row['total_bytes'];version=row['version']
    else:
     assert row['status']==206 and h['Content-Range']==f'bytes {position}-{total-1}/{total}'and int(h['Content-Length'])==total-position
     for key,value in version.items():
      if value is not None:assert h[key]==value
    parts.append(data);position+=len(data)
   else:assert row['received_bytes']==0 and row.get('error_type')
  raw=Path(r['path']).read_bytes();assert raw==b''.join(parts)and len(raw)==position==total<=5*1024**2 and hashlib.sha256(raw).hexdigest()==r['file_sha256']
  if r['S3_ETag_MD5_verified']:assert hashlib.md5(raw).hexdigest()==r['S3_ETag'].strip('"').lower()
  assert r['license_id']==im['license']and r['license']==licenses[im['license']]and r['flickr_url']==im['flickr_url']and r['original_coco_url']==im['coco_url']
  with Image.open(io.BytesIO(raw))as original:
   rgb=original.convert('RGB');rgb.load();assert rgb.size==(im['width'],im['height'])==(r['width'],r['height']);assert hashlib.sha256(struct.pack('<II',*rgb.size)+rgb.tobytes()).hexdigest()==r['pixel_sha256'];gray=np.array(rgb.convert('L').resize((32,32),Image.Resampling.BILINEAR).resize((9,8),Image.Resampling.BILINEAR))
   bits=''.join('1'if gray[y,x+1]>gray[y,x]else'0'for y in range(8)for x in range(8));assert f'{int(bits,2):016x}'==r['dhash64']
  print('verified COCO RGB',r['role'],r['image_id'],flush=True)
 conn=sqlite3.connect(f'file:{work.INDEX/"index.sqlite"}?mode=ro',uri=True);native=[json.loads(r[0])for r in conn.execute('SELECT record FROM frames ORDER BY path')];conn.close();assert len(native)==42149
 refs=[{'dataset':'OVIS','path':str(Path('semantic_artifacts/ovis/train')/r['path']),**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in native];old=[]
 for path in work.PREVIOUS_RGB:old+=json.loads(path.read_text())
 assert len(old)==1040;refs+=[{'dataset':'LVVIS_seen130','path':r['path'],**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in old];pairs=[]
 def add(left,right,dataset,distance):
  pairs.append({'pair_id':len(pairs),'image_id':left['image_id'],'role':left['role'],'external_path':left['path'],'external_sha256':left['file_sha256'],'reference_dataset':dataset,'reference_path':right['path'],'reference_sha256':right['file_sha256'],'reference_image_id':right.get('image_id'),'reference_role':right.get('role'),'distance':distance,'exact_file':left['file_sha256']==right['file_sha256'],'exact_pixels':left['pixel_sha256']==right['pixel_sha256']})
 for left in records:
  for right in refs:
   d=(int(left['dhash64'],16)^int(right['dhash64'],16)).bit_count()
   if d<=6:assert file_hash(right['path'])==right['file_sha256'];add(left,right,right['dataset'],d)
 for i,left in enumerate(records):
  for right in records[i+1:]:
   d=(int(left['dhash64'],16)^int(right['dhash64'],16)).bit_count()
   if d<=6:add(left,right,'new_coco',d)
 assert pairs==json.loads((ROOT/'pairs.json').read_text())and file_hash(ROOT/'pairs.json')==materials['pairs_sha256']and len(pairs)==s['near_pairs']==materials['pair_count']
 assert sum(r['exact_file']or r['exact_pixels']for r in pairs)==s['exact_pairs']==materials['exact_pairs']and dict(Counter(r['reference_dataset']for r in pairs))==materials['pair_sources']
 for sheet in materials['sheets']:
  assert file_hash(sheet['path'])==sheet['sha256']
  for panel in sheet['panels']:assert file_hash(panel['path'])==panel['sha256']
 decisions=json.loads((ROOT/'review_decisions.json').read_text());assert file_hash(ROOT/'review_decisions.json')==s['decisions_sha256'];assert [d['pair_id']for d in decisions['decisions']]==list(range(len(pairs)));bad=set()
 for pair,d in zip(pairs,decisions['decisions']):
  assert d['judgment']in('distinct','same','uncertain')and d['reason']
  if pair['exact_file']or pair['exact_pixels']:assert d['judgment']!='distinct'
  if d['judgment']!='distinct':
   bad.add(pair['image_id'])
   if pair['reference_dataset']=='new_coco':bad.add(pair['reference_image_id'])
 eligible=json.loads((ROOT/'frozen_eligibility.json').read_text());assert file_hash(ROOT/'frozen_eligibility.json')==s['eligibility_sha256']and eligible['no_replacement']and eligible['excluded_image_ids']==s['excluded_image_ids']==sorted(bad)
 for role in roles:assert eligible[role]==[i for i in roles[role]if i not in bad]and len(eligible[role])==s[role+'_images']
 # Every head and offset is tied to its previously frozen validation selection.
 cap=json.loads((work.capacity.ROOT/'frozen_selection.json').read_text());balanced=json.loads((work.balanced.ROOT/'frozen_selection.json').read_text());sam=json.loads(Path('semantic_artifacts/external_semantic_confirmation/frozen_selection.json').read_text())['selection']['native_sam']
 for arm in work.ARMS:
  if arm in work.capacity.ARMS:offset=cap['offsets'][arm];hashes={str(i):cap['checkpoint_sha256'][f'{arm}_{i}']for i in range(3)}
  elif arm.startswith('B14_class'):
   key=arm[4:];offset=balanced['offsets'][key];hashes={str(i):balanced['checkpoint_sha256'][f'{key}_{i}']for i in range(3)}
  else:offset=sam['offset'];hashes=sam['checkpoint_sha256']
  assert p['models'][arm]=={'offset':offset,'checkpoint_sha256':hashes}
  for seed in range(3):assert file_hash(work.ckpt(arm,seed))==hashes[str(seed)]
 assert file_hash('semantic/finalize_coco_cohort.py')==s['finalizer_sha256']and s['no_GT_masks_or_model_outputs']and s['LVVIS40_reserve_unchanged']and not Path('semantic_artifacts/coco_semantic_confirmation').exists()
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all160_split_images_and_licenses_recomputed':True,'JPEG_requests_and_MD5_pixels_hashes_recomputed':len(records),'complete_matching_pairs_recomputed':len(pairs),'all_review_decisions_and_frozen_roles_recomputed':True,'all21_heads_and_offsets_unchanged':True,'no_new_model_evaluation_or_GT_decoding':True})
 print('DONE independent COCO cohort verification',len(pairs),flush=True)
if __name__=='__main__':main()
