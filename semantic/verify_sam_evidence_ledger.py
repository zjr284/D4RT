"""Independently check source class-F1 means, original intervals and CSV values."""
import csv,json,math
from pathlib import Path
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/sam_evidence_ledger')

def close(a,b):assert abs(a-b)<1e-12,(a,b)
def main():
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==96 and len(s['rows'])==9 and p['no_cross_cohort_aggregation']
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 expected=['R20','R24','R33','R51','R53','R72_main','R72_secondary','R93','R93_sensitivity'];assert[r['id']for r in s['rows']]==expected
 for row in s['rows']:
  path=Path(row['source_summary']);src=json.loads(path.read_text());v=json.loads(path.with_name('verification.json').read_text());assert file_hash(path)==v['summary_sha256']==row['source_summary_sha256'];rid=row['id'];left=row['left_arm'];right=row['reference_arm']
  if rid in ('R20','R24','R33'):
   arms=src['results'];field='calibrated_confirmation';contrast=src['contrasts']['SAM_at448'if rid=='R20'else'SAM_native']if rid!='R33'else src['secondary_descriptive_contrasts']['native_sam_minus_source_gt'];boot=contrast['video_bootstrap'];unit='video'
  elif rid in ('R51','R53'):
   arms=src['results'];field='calibrated';boot=src['secondary_contrasts_unadjusted']['native_sam_minus_source_gt']['conditional_video_bootstrap'];unit='directory group'
  elif rid.startswith('R72'):
   node=src['policies']['strict'];arms=node['results'];field='calibrated';contrast=node['primary_contrast']if rid=='R72_main'else node['secondary_contrasts_unadjusted']['B14_aux_sam_minus_B14_gt32'];boot=contrast['conditional_image_bootstrap'];unit='image'
  else:
   policy='strict'if rid=='R93'else'consensus';node=src['results'][policy];arms={a:{'runs':[r['metrics']['calibrated']['pooled']for r in x['runs']]}for a,x in node['arms'].items()};field=None;boot=node['secondary_contrasts_unadjusted']['native_sam_minus_source_gt']['conditional_video_bootstrap'];unit='directory group'
  arrays={}
  for label,arm in [('SAM',left),('reference',right)]:
   m=[r[field]if field else r for r in arms[arm]['runs']];values=[]
   for r in m:
    values.append(math.fsum(r['per_class_f1'].values())/len(r['per_class_f1']));close(values[-1],r['macro_f1_present']);assert len(r['support'])==row['actual_GT_class_count']and sum(r['support'].values())==row['query_count'];assert ','.join(sorted(r['support'],key=int))==row['actual_GT_classes']
   arrays[label]=values;close(math.fsum(values)/3,row[label+'_F1_mean'])
  for i in range(3):close(arrays['SAM'][i]-arrays['reference'][i],row[f'seed{i}_difference'])
  close(math.fsum(arrays['SAM'][i]-arrays['reference'][i]for i in range(3))/3,row['pooled_F1_difference_mean']);assert row['unit']==unit
  close(boot['percentile_95_interval'][0],row['unit_equal_interval_low']);close(boot['percentile_95_interval'][1],row['unit_equal_interval_high']);close(boot['mean_per_image_difference'if unit=='image'else'mean_per_video_difference'],row['unit_equal_mean_difference']);assert boot['image_count'if unit=='image'else'video_count']==row['units_with_queries']
  lo,hi=boot['percentile_95_interval'];assert row['interval_description']==('positive interval'if lo>0 else'negative interval'if hi<0 else'interval includes zero')
  if rid in ('R72_secondary','R33','R51','R53','R93'):assert row['role']=='secondary unadjusted'
  if rid=='R72_main':assert row['role']=='primary'
  if rid=='R93_sensitivity':assert row['role'].startswith('same-cohort sensitivity')
 rows=list(csv.DictReader((ROOT/'evidence.csv').open(encoding='utf-8-sig')));assert len(rows)==9
 for a,b in zip(rows,s['rows']):
  assert set(a)==set(b)
  for k,v in b.items():
   if isinstance(v,(float,int)):close(float(a[k]),v)
   else:assert a[k]==v
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'CSV_sha256':file_hash(ROOT/'evidence.csv'),'all9_rows_and_7_source_verification_bindings_checked':True,'per_class_F1_reaggregation_and_seed_differences_checked':True,'all_original_unit_intervals_and_support_counts_checked':True,'same_cohort_and_primary_secondary_roles_preserved':True,'no_cross_cohort_effect_estimated':True,'no_model_data_or_source_result_changes':True});print('DONE independent SAM evidence ledger',flush=True)
if __name__=='__main__':main()
