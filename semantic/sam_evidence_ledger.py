"""Nine prespecified historical SAM contrasts; never pool different cohorts."""
import csv,json
from pathlib import Path
import numpy as np
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/sam_evidence_ledger')
# id, folder, policy, left, right, role, data scope, contrast key
SPECS=[
 ('R20','dino_resolution_sam_control',None,'448_sam','448_gt','primary supervision contrast','OVIS40 / old RGB448','SAM_at448'),
 ('R24','dino_final_confirmation',None,'native_sam','native_gt','secondary supervision contrast','OVIS40 / native JPEG448','SAM_native'),
 ('R33','dino_low_batch_confirmation',None,'native_sam','source_gt','secondary unadjusted','OVIS27 / native JPEG448','native_sam_minus_source_gt'),
 ('R51','external_semantic_confirmation',None,'native_sam','source_gt','secondary unadjusted','LVVIS40 candidates / Person-only','native_sam_minus_source_gt'),
 ('R53','external_nonperson_confirmation',None,'native_sam','source_gt','secondary unadjusted','LVVIS10 class-directed groups','native_sam_minus_source_gt'),
 ('R72_main','coco_sam_auxiliary','strict','B14_aux_sam','B14_extra_gt','primary','COCO80 / strict nonPerson / shared cohort','primary'),
 ('R72_secondary','coco_sam_auxiliary','strict','B14_aux_sam','B14_gt32','secondary unadjusted','same COCO80 / strict nonPerson','B14_aux_sam_minus_B14_gt32'),
 ('R93','external_reserve_confirmation','strict','native_sam','source_gt','secondary unadjusted','original LVVIS40 reserve / Person-only','native_sam_minus_source_gt'),
 ('R93_sensitivity','external_reserve_confirmation','consensus','native_sam','source_gt','same-cohort sensitivity; secondary unadjusted','same original LVVIS40 / Person plus single-group Bird','native_sam_minus_source_gt')]

def extract(folder,s,policy,key):
 if folder=='coco_sam_auxiliary':
  data=s['policies'][policy];arms=data['results'];c=data['primary_contrast']if key=='primary'else data['secondary_contrasts_unadjusted'][key];return arms,'calibrated',c['conditional_image_bootstrap'],'image'
 if folder=='external_reserve_confirmation':
  data=s['results'][policy];arms={a:{'runs':[r['metrics']['calibrated']['pooled']for r in v['runs']]}for a,v in data['arms'].items()};return arms,None,data['secondary_contrasts_unadjusted'][key]['conditional_video_bootstrap'],'directory group'
 arms=s['results']
 if folder in ('dino_resolution_sam_control','dino_final_confirmation'):return arms,'calibrated_confirmation',s['contrasts'][key]['video_bootstrap'],'video'
 if folder=='dino_low_batch_confirmation':return arms,'calibrated_confirmation',s['secondary_descriptive_contrasts'][key]['video_bootstrap'],'video'
 return arms,'calibrated',s['secondary_contrasts_unadjusted'][key]['conditional_video_bootstrap'],'directory group'

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 if(ROOT/'summary.json').exists():raise RuntimeError('Completed evidence ledger exists')
 paths=[Path('semantic/sam_evidence_ledger.py'),Path('semantic/OVIS_SAM_EVIDENCE_LEDGER_PROTOCOL.md')];sources={}
 for folder in sorted({row[1]for row in SPECS}):
  root=Path('semantic_artifacts')/folder;s=json.loads((root/'summary.json').read_text());v=json.loads((root/'verification.json').read_text());assert file_hash(root/'summary.json')==v['summary_sha256'];sources[folder]=s;paths.extend([root/'summary.json',root/'verification.json'])
 p={'round':96,'scope':'historical evidence compilation only','fixed_specs':SPECS,'input_sha256':{str(x):file_hash(x)for x in paths},'no_cross_cohort_aggregation':True};atomic_json(ROOT/'protocol.json',p);rows=[]
 for rid,folder,policy,left,right,role,scope,key in SPECS:
  source=sources[folder];arms,metric_key,boot,unit=extract(folder,source,policy,key);scores={};support=None
  for alias,arm in [('SAM',left),('reference',right)]:
   m=[r[metric_key]if metric_key else r for r in arms[arm]['runs']];scores[alias]=[r['macro_f1_present']for r in m]
   for r in m:
    if support is None:support=r['support']
    assert r['support']==support
  diffs=[a-b for a,b in zip(scores['SAM'],scores['reference'])];assert len(diffs)==3
  mean_key='mean_per_image_difference'if unit=='image'else'mean_per_video_difference';count_key='image_count'if unit=='image'else'video_count';lo,hi=boot['percentile_95_interval'];status='positive interval'if lo>0 else'negative interval'if hi<0 else'interval includes zero'
  rows.append({'id':rid,'scope':scope,'role':role,'policy':policy or'original fixed','left_arm':left,'reference_arm':right,'SAM_F1_mean':float(np.mean(scores['SAM'])),'reference_F1_mean':float(np.mean(scores['reference'])),'pooled_F1_difference_mean':float(np.mean(diffs)),'seed0_difference':diffs[0],'seed1_difference':diffs[1],'seed2_difference':diffs[2],'unit':unit,'units_with_queries':boot[count_key],'actual_GT_classes':','.join(sorted(support,key=int)),'actual_GT_class_count':len(support),'query_count':sum(support.values()),'unit_equal_mean_difference':boot[mean_key],'unit_equal_interval_low':lo,'unit_equal_interval_high':hi,'interval_description':status,'source_summary':str(Path('semantic_artifacts')/folder/'summary.json'),'source_summary_sha256':p['input_sha256'][str(Path('semantic_artifacts')/folder/'summary.json')]})
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'summary.json',{'protocol':p,'rows':rows,'limits':['same cohort policies and comparisons are not independent sets','no pooled cross-cohort effect','secondary intervals unadjusted','different recipes and class supports','no training inference calibration or deployment change']})
 with(ROOT/'evidence.csv').open('w',encoding='utf-8-sig',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 print('DONE fixed SAM evidence ledger',len(rows),flush=True)
if __name__=='__main__':main()
