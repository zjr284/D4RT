"""Offline verification of the two frozen HTTPS HEAD observations."""
import json
from semantic.coco_endpoint_probe import ROOT,URLS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
p=json.loads((ROOT/'protocol.json').read_text());s=json.loads((ROOT/'summary.json').read_text());rows=json.loads((ROOT/'responses.json').read_text());assert p==s['protocol']and rows==s['responses']and file_hash(ROOT/'responses.json')==s['responses_sha256']
for path,h in p['input_sha256'].items():assert file_hash(path)==h
assert [r['url']for r in rows]==URLS and len(rows)==2 and all(r['method']=='HEAD'and r['received_body_bytes']==0 for r in rows)
assert rows[0]['status']is None and rows[0]['error_type']=='URLError'and 'CERTIFICATE_VERIFY_FAILED'in rows[0]['error']and 'Hostname mismatch'in rows[0]['error']
assert rows[1]['status']==200 and rows[1]['final_url']==URLS[1]and rows[1]['headers']['Server']=='AmazonS3'and int(rows[1]['headers']['Content-Length'])==252907541
assert s['GET_requests']==s['body_bytes_received']==0 and s['candidate_bucket_identity_not_independently_proven']
atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'two_HEAD_records_recomputed':True,'canonical_TLS_failure_preserved':True,'S3_candidate_200_only_not_proven_official_identity':True,'no_body_download_or_model_evaluation':True,'frozen_inputs_unchanged':True})
print('VERIFIED COCO HEAD diagnostics; provenance still pending',flush=True)
