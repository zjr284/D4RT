"""Freeze COCO RGB eligibility only after actual complete blind pair review."""
import json
from pathlib import Path
from semantic.coco_cohort_segments import ROOT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

def main():
 if(ROOT/'summary.json').exists():raise RuntimeError('Completed COCO cohort exists')
 assert not Path('semantic_artifacts/coco_semantic_confirmation').exists()
 p=json.loads((ROOT/'protocol.json').read_text());m=json.loads((ROOT/'manifest.json').read_text());materials=json.loads((ROOT/'materials.json').read_text());pairs=json.loads((ROOT/'pairs.json').read_text())
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert materials['images']==160 and materials['roles']=={'candidate':80,'reserve':80}and len(pairs)==materials['pair_count']and file_hash(ROOT/'pairs.json')==materials['pairs_sha256']
 dp=ROOT/'review_decisions.json'
 if not pairs:atomic_json(dp,{'review_required':False,'reason':'No exact or near pairs in complete frozen screen','decisions':[]})
 elif not dp.exists():raise RuntimeError('All RGB sheets must actually be viewed and all pairs judged')
 decisions=json.loads(dp.read_text());assert [d['pair_id']for d in decisions['decisions']]==list(range(len(pairs)));excluded=set()
 for pair,d in zip(pairs,decisions['decisions']):
  assert d['judgment']in('distinct','same','uncertain')and d['reason']
  if pair['exact_file']or pair['exact_pixels']:assert d['judgment']!='distinct'
  if d['judgment']!='distinct':
   excluded.add(pair['image_id'])
   if pair['reference_dataset']=='new_coco':excluded.add(pair['reference_image_id'])
 eligible={role:[im['id']for im in m[role]if im['id']not in excluded]for role in ('candidate','reserve')};frozen={'manifest_sha256':file_hash(ROOT/'manifest.json'),**eligible,'excluded_image_ids':sorted(excluded),'decisions_sha256':file_hash(dp),'no_replacement':True,'frozen_before_features_or_GT_masks':True};atomic_json(ROOT/'frozen_eligibility.json',frozen)
 atomic_json(ROOT/'summary.json',{'protocol':p,'manifest_sha256':file_hash(ROOT/'manifest.json'),'materials_sha256':file_hash(ROOT/'materials.json'),'candidate_images':len(eligible['candidate']),'reserve_images':len(eligible['reserve']),'metadata_role_class_image_counts':m['role_class_image_counts'],'images_acquired':160,'actual_GET_requests':len(json.loads((ROOT/'requests.json').read_text())),'prior_GET_requests':72,'prior_received_RGB_bytes':9446274,'received_RGB_bytes':sum(r['received_bytes']for r in json.loads((ROOT/'requests.json').read_text())),'near_pairs':len(pairs),'exact_pairs':materials['exact_pairs'],'sheets':len(materials['sheets']),'excluded_image_ids':sorted(excluded),'decisions_sha256':file_hash(dp),'eligibility_sha256':file_hash(ROOT/'frozen_eligibility.json'),'finalizer_sha256':file_hash('semantic/finalize_coco_segments.py'),'no_GT_masks_or_model_outputs':True,'LVVIS40_reserve_unchanged':True,'deployment_unchanged':'native_gt'});print('DONE COCO cohort eligibility',len(eligible['candidate']),len(eligible['reserve']),flush=True)
if __name__=='__main__':main()
