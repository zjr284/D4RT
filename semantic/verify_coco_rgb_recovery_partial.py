"""Offline boundary check for the one-GET limited COCO recovery."""
import json
from pathlib import Path
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/coco_cohort_recovery');OLD=Path('semantic_artifacts/coco_cohort')
p=json.loads((ROOT/'protocol.json').read_text());health=json.loads((ROOT/'health.json').read_text());rows=json.loads((ROOT/'requests.json').read_text());failure=json.loads((ROOT/'failure.json').read_text())
for path,h in p['input_sha256'].items():assert file_hash(path)==h
assert health['status']==200 and len(rows)==1 and rows[0]['image_id']==460379 and rows[0]['status']==200
r=rows[0];assert r['range_start']==0 and r['received_bytes']==225852 and r['total_bytes']==262954 and Path(r['path']).stat().st_size==225852 and file_hash(r['path'])==r['sha256']=='6bbc477b0246f8d69e4046e54e35f778ce5b5189bc22556c5ec36850fd9083b7'
assert r['version']=={k:health['headers'][k]for k in ('ETag','Last-Modified')}and int(health['headers']['Content-Length'])==262954
assert json.loads((ROOT/'image_records.json').read_text())==json.loads((OLD/'image_records.json').read_text())and not(ROOT/'materials.json').exists()
assert failure['message']=='Image 460379 not complete after at most two requests; no replacement'
s={'protocol':p,'scope':'one-GET recovery short read, no complete cohort','health_HEAD_requests':1,'health_HTTP_status':200,'new_GET_requests':1,'new_received_bytes':225852,'known_image_total_bytes':262954,'remaining_suffix_bytes':37102,'prefix_path':r['path'],'prefix_sha256':r['sha256'],'prefix_version':r['version'],'old_completed_images_preserved':51,'no_GT_masks_or_model_evaluation':True,'failure':failure,'failure_message_mentions_two_but_actual_special_image_budget_was_one':True,'deployment_unchanged':'native_gt'}
atomic_json(ROOT/'summary.json',s);atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'health_and_short_prefix_recomputed':True,'one_new_GET_limit_observed':True,'all_old_inputs_and51_RGB_records_unchanged':True,'no_false_complete_image_or_RGB_screening':True})
print('VERIFIED COCO recovery prefix',225852,'missing',37102,flush=True)
