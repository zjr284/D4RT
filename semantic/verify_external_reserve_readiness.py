"""Independent reserve identity, RGB fingerprints and complete cross-library pairs."""
from collections import Counter
import hashlib,json,sqlite3,struct
from pathlib import Path
import numpy as np
from PIL import Image
from semantic import external_reserve_readiness as work
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def main():
 p=json.loads((ROOT/'protocol.json').read_text());s=json.loads((ROOT/'summary.json').read_text());assert p['round']==92 and s['protocol']==p and s['no_new_GT_masks_features_or_model_evaluation']
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 old=json.loads(work.original.COHORT.read_text());old_elig=json.loads((work.original.REVIEW/'frozen_eligibility.json').read_text());m=json.loads((ROOT/'manifest.json').read_text());assert m['confirmation']==old['provisional_reserve']and[c['group']for c in m['confirmation']]==old_elig['reserve']and len(m['confirmation'])==40 and m['original_reserve_activated']and m['original_manifest_sha256']==file_hash(work.original.COHORT)
 groups=set(old_elig['reserve']);assert not groups&set(old_elig['candidate_confirmation'])
 for folder in Path('semantic_artifacts').glob('external*/features'):
  assert not groups&{path.stem for path in folder.rglob('*.npz')}
 records=json.loads((ROOT/'image_records.json').read_text());original={r['file_name']:r for r in json.loads((work.original.RGB/'image_records.json').read_text())};assert len(records)==320 and records==[original[im['file_name']]for clip in m['confirmation']for im in clip['frames']]
 for row in records:
  raw=Path(row['path']).read_bytes()
  with Image.open(row['path'])as im:rgb=im.convert('RGB');rgb.load();width,height=rgb.size;pixels=struct.pack('<II',width,height)+rgb.tobytes();small=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.BILINEAR).resize((9,8),Image.Resampling.BILINEAR));dhash=np.packbits((small[:,1:]>small[:,:-1]).ravel(),bitorder='big').tobytes().hex()
  assert hashlib.sha256(raw).hexdigest()==row['file_sha256']and hashlib.sha256(pixels).hexdigest()==row['pixel_sha256']and dhash==row['dhash64']and(width,height)==(row['width'],row['height'])
 conn=sqlite3.connect(f'file:{work.coco.prior.old.INDEX/"index.sqlite"}?mode=ro',uri=True);native=[json.loads(row[0])for row in conn.execute('SELECT record FROM frames ORDER BY path')];conn.close();assert len(native)==42149
 refs=[{'dataset':'OVIS_all42149','path':str(Path('semantic_artifacts/ovis/train')/r['path']),**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in native];excluded={r['file_name']for r in records};seen=set()
 for source in work.LV_SOURCES:
  for row in json.loads(source.read_text()):
   if row['file_name']not in excluded and row['path']not in seen:refs.append({'dataset':'LVVIS_old_RGB_excluding_reserve','path':row['path'],'group':row['group'],**{k:row[k]for k in ('file_sha256','pixel_sha256','dhash64')}});seen.add(row['path'])
 for source in work.COCO_SOURCES:
  for row in json.loads(source.read_text()):refs.append({'dataset':'COCO_previously_evaluated240','path':row['path'],'image_id':row['image_id'],**{k:row[k]for k in ('file_sha256','pixel_sha256','dhash64')}})
 pairs=[]
 def append(left,right,distance):
  assert file_hash(right['path'])==right['file_sha256'];pairs.append({'pair_id':len(pairs),'group':left['group'],'image_id':left['image_id'],'external_path':left['path'],'reference_dataset':right.get('dataset','new_reserve_cross_group'),'reference_group':right.get('group'),'reference_path':right['path'],'distance':int(distance),'exact_file':left['file_sha256']==right['file_sha256'],'exact_pixels':left['pixel_sha256']==right['pixel_sha256']})
 for left in records:
  for right in refs:
   distance=(int(left['dhash64'],16)^int(right['dhash64'],16)).bit_count()
   if distance<=6:append(left,right,distance)
 for i,left in enumerate(records):
  for right in records[i+1:]:
   if left['group']==right['group']:continue
   distance=(int(left['dhash64'],16)^int(right['dhash64'],16)).bit_count()
   if distance<=6:append(left,right,distance)
 assert pairs==json.loads((ROOT/'pairs.json').read_text());materials=json.loads((ROOT/'materials.json').read_text());assert materials['pair_count']==len(pairs)and materials['pairs_sha256']==file_hash(ROOT/'pairs.json')and materials['reference_counts']==dict(Counter(r['dataset']for r in refs))and materials['exact_pairs']==sum(r['exact_file']or r['exact_pixels']for r in pairs)and materials['images']==320 and materials['groups']==40
 for sheet in materials['near_sheets']+materials['all320_RGB_overviews']:
  assert file_hash(sheet['path'])==sheet['sha256']
  for panel in sheet['panels']:assert file_hash(panel['path'])==panel['sha256']
 overview_paths=[panel['path']for sheet in materials['all320_RGB_overviews']for panel in sheet['panels']];assert overview_paths==[r['path']for r in records]
 viewed=json.loads((ROOT/'rgb_review.json').read_text());eligible=json.loads((ROOT/'frozen_eligibility.json').read_text());assert file_hash(ROOT/'materials.json')==s['materials_sha256']==viewed['materials_sha256']and file_hash(ROOT/'rgb_review.json')==s['rgb_review_sha256']==eligible['rgb_review_sha256']and file_hash(ROOT/'frozen_eligibility.json')==s['frozen_eligibility_sha256']
 assert viewed['viewed_overview_sha256']==[r['sha256']for r in materials['all320_RGB_overviews']]and viewed['viewed_near_sheet_sha256']==[r['sha256']for r in materials['near_sheets']]and set(viewed['pair_ids_reviewed'])=={r['pair_id']for r in pairs}
 assert eligible['accepted_groups']==[c['group']for c in m['confirmation']if c['group']not in viewed['excluded_groups']]and eligible['excluded_groups']==viewed['excluded_groups']and eligible['no_replacement']
 source=json.loads((work.original.RGB/'summary.json').read_text());bs=json.loads((work.b14.ROOT/'summary.json').read_text())
 for arm,c in p['models'].items():
  if arm in work.ARMS[:4]:assert c['offset']==source['protocol']['selection'][arm]['offset']and c['checkpoint_sha256']==source['protocol']['selection'][arm]['checkpoint_sha256']
  else:assert c['offset']==bs['results']['b14_gt32'if arm=='B14_raw'else'z_b14_gt32']['offsets']['balanced_profile']
  for seed in range(3):assert file_hash(work.ckpt(arm,seed))==c['checkpoint_sha256'][str(seed)]
 assert file_hash(work.original.rgb.prior.resolution.WEIGHTS)=='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'and file_hash(work.B14_WEIGHTS)=='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'and p['statistics']==bs['statistics']['b14_gt32']and file_hash(p['statistics']['path'])==p['statistics']['sha256']
 assert p['primary']==['source_gt32','source_gt']and p['secondary']==[['native_sam','source_gt'],['lambda0','source_gt'],['B14_raw','source_gt32'],['B14_z','B14_raw'],['B14_z','source_gt32']]
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert not(ROOT/'queries').exists()and not(ROOT/'features').exists()and not(ROOT/'logits').exists()
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'original_reserve40_identity_and_no_old_model_cache_verified':True,'all320_original_JPEG_pixel_dhash_dimensions_verified':True,'complete_new_reference_pairs_recomputed':len(pairs),'reference_counts':materials['reference_counts'],'all_actual_review_materials_and_eligibility_frozen':True,'all18_old_heads_offsets_backbones_training_statistics_unchanged':True,'no_new_network_GT_masks_features_or_model_evaluation':True});print('DONE independent original reserve RGB-only qualification refresh',len(pairs),len(eligible['accepted_groups']),flush=True)
if __name__=='__main__':main()
