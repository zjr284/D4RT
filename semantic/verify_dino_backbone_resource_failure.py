"""Offline independent evidence of failed bounded weight acquisition."""
import hashlib,json,zipfile
from pathlib import Path
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/dino_backbone_capacity')
WEIGHT=Path('semantic_artifacts/weights/dinov2_vitb14_pretrain.partial.pth')
def main():
 protocol=json.loads((ROOT/'resource_protocol.json').read_text());failure=json.loads((ROOT/'resource_failure.json').read_text())
 assert file_hash('semantic/OVIS_DINO_BACKBONE_CAPACITY_PROTOCOL.md')==protocol['protocol_sha256']and file_hash('semantic/dino_backbone_resource.py')==protocol['resource_code_sha256']
 h=hashlib.sha256();size=0
 with WEIGHT.open('rb')as f:
  first=f.read(4);assert first==b'PK\x03\x04';f.seek(0)
  for chunk in iter(lambda:f.read(1024*1024),b''):size+=len(chunk);h.update(chunk)
 assert size==346335620 and h.hexdigest()=='77f43daba3dd5b196d71ae576c7881447e8b87aed2d3c043e605b35f97aff9b2'
 assert not zipfile.is_zipfile(WEIGHT)and size<protocol['maximum_bytes']
 assert failure['type']=='AssertionError'and not(ROOT/'resource_preflight.json').exists()and not ROOT.joinpath('protocol.json').exists()
 assert not list(ROOT.glob('*/runs/*.pt'))and not ROOT.joinpath('features').exists()
 log=(ROOT.parent/'dino_backbone_capacity_resource.log').read_text();assert "n==int(record['content_length'])"in log
 s={'scope':'resource failure only; no feature or head result','resource_protocol_sha256':file_hash(ROOT/'resource_protocol.json'),'partial_bytes':size,'partial_sha256':h.hexdigest(),'zip_complete':False,'failure_at_content_length_check':True,'response_content_length_not_persisted':True,'max_full_requests':1,'new_heads_trained':0,'new_features':0,'new_confirmation_groups':0,'deployment_unchanged':'native_gt'}
 atomic_json(ROOT/'summary.json',s);atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'offline_partial_SHA_size_ZIP_rechecked':True,'protocol_resource_code_unchanged':True,'no_training_or_preflight_outputs':True})
 print('VERIFIED bounded B14 acquisition failure',size,flush=True)
if __name__=='__main__':main()
