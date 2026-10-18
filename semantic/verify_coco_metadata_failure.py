"""Offline audit of failed bounded COCO HEAD; no retry or model evaluation."""
import json
from pathlib import Path
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/coco_metadata_acquisition')
def main():
 p=json.loads((ROOT/'protocol.json').read_text());failure=json.loads((ROOT/'failure.json').read_text());log=(ROOT.parent/'coco_metadata_acquisition.log').read_text()
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert failure=={'type':'HTTPError','message':'HTTP Error 404: Not Found'}and "method='HEAD'"in log
 assert not(ROOT/'head.json').exists()and not(ROOT/'ranges.json').exists()and not(ROOT/'instances_val2017.json').exists()and not(ROOT/'directory.json').exists()
 s={'protocol':p,'failure':failure,'stage':'official GCP bucket HEAD returned404','HEAD_attempts':1,'Range_GET_attempts':0,'annotation_bytes_received':0,'no_annotation_or_RGB_or_model_evaluation':True,'HTTP_error_headers_not_persisted':True,'deployment_unchanged':'native_gt'};atomic_json(ROOT/'summary.json',s)
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'offline_traceback_and_missing_output_boundary_rechecked':True,'all_frozen_inputs_and_model_selections_unchanged':True,'no_retry_performed':True})
 print('VERIFIED COCO HEAD404 failure; no member acquired',flush=True)
if __name__=='__main__':main()
