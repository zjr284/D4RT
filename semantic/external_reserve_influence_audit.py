"""Post-score full-group influence and metric-scope diagnostics, no model changes."""
from collections import Counter,defaultdict
from itertools import combinations
import csv,fcntl,json
from pathlib import Path
import numpy as np
from semantic import external_reserve_confirmation as source
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics
ROOT=Path('semantic_artifacts/external_reserve_influence_audit')

def influence(per):
 values={k:v for k,v in per.items()if v is not None};full=float(np.mean(list(values.values())));out={}
 for n in (1,2):
  rows=[{'omitted_groups':list(keys),'remaining_mean':float(np.mean([v for k,v in values.items()if k not in keys]))}for keys in combinations(values,n)]
  means=np.array([r['remaining_mean']for r in rows]);out[str(n)]={'all_omission_subsets':rows,'min':float(means.min()),'max':float(means.max()),'positive':int((means>0).sum()),'negative':int((means<0).sum()),'tied':int((means==0).sum())}
 positive=sorted([{'group':k,'difference':v}for k,v in values.items()if v>0],key=lambda r:(-r['difference'],r['group']));gain=sum(r['difference']for r in positive)
 return {'full_video_equal_mean':full,'valid_groups':len(values),'leave_out':out,'positive_groups_by_contribution':positive,'top2_fraction_of_positive_contributions':sum(r['difference']for r in positive[:2])/gain if gain else None,'note':'post-score leverage diagnostic; subset means are not confidence intervals or replacement tests'}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed influence audit exists')
  s=json.loads((source.ROOT/'summary.json').read_text());verification=json.loads((source.ROOT/'verification.json').read_text());assert file_hash(source.ROOT/'summary.json')==verification['summary_sha256']
  clips=json.loads((source.ROOT/'manifest.json').read_text())['confirmation'];paths=[source.ROOT/name for name in ('summary.json','verification.json','manifest.json','coverage.json')]+[Path(x)for x in s['output_hashes']]+[source.ANN,source.old.MAPPING_SUMMARY,Path('semantic/external_reserve_influence_audit.py'),Path('semantic/OVIS_EXTERNAL_RESERVE_INFLUENCE_PROTOCOL.md')]
  p={'round':95,'scope':'post-score descriptive','source_summary_sha256':verification['summary_sha256'],'comparisons':[s['protocol']['primary'],*s['protocol']['secondary']],'input_sha256':{str(x):file_hash(x)for x in paths}};atomic_json(ROOT/'protocol.json',p)
  payload=json.loads(source.ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(source.old.MAPPING_SUMMARY.read_text())['shared_category_mapping'].items()};category_names={c['id']:c['name']for c in payload['categories']};annotations=defaultdict(list)
  for a in payload['annotations']:annotations[a['image_id']].append(a)
  group_coverage={}
  for clip in clips:
   ann=[a for im in clip['frames']for a in annotations[im['id']]];counts=Counter(category_names[a['category_id']]for a in ann);eligible=sum(a['category_id']in mapping and not a.get('iscrowd',0)for a in ann);row={'annotation_category_counts':dict(sorted(counts.items())),'all_annotations':len(ann),'mapped_noncrowd_annotations':eligible}
   for policy in source.POLICIES:
    with np.load(source.qp(policy,clip['group']))as a:row[policy+'_queries']=len(a['labels'])
   row['strict_empty_reason']=None if row['strict_queries']else'no_mapped_noncrowd_annotations'if not eligible else'mapped_annotations_filtered_by_fixed_mask_rules';group_coverage[clip['group']]=row
  results={};csv_rows=[]
  for policy in source.POLICIES:
   old=s['results'][policy];labels=np.concatenate([np.load(source.qp(policy,c['group']))['labels']for c in clips]);present=sorted(map(int,np.unique(labels)));diagnostics=[]
   for c in range(26):
    pred=np.full(len(labels),c);m=classification_metrics(pred,labels,26);diagnostics.append({'constant_prediction_class':c,'accuracy':m['accuracy'],'macro_f1_present':m['macro_f1_present']})
   comparisons={}
   for i,(left,right)in enumerate(p['comparisons']):
    stored=old['primary_contrast']if i==0 else old['secondary_contrasts_unadjusted'][left+'_minus_'+right];key=left+'_minus_'+right;per=stored['per_video_three_seed_mean_differences'];contribution={}
    for c in present:
     diffs=[old['arms'][left]['runs'][j]['metrics']['calibrated']['pooled']['per_class_f1'][str(c)]-old['arms'][right]['runs'][j]['metrics']['calibrated']['pooled']['per_class_f1'][str(c)]for j in range(3)];contribution[str(c)]={'per_seed_f1_differences':diffs,'per_seed_macro_contributions':[x/len(present)for x in diffs],'mean_macro_contribution':float(np.mean(diffs)/len(present))}
    mean_sum=sum(r['mean_macro_contribution']for r in contribution.values());assert abs(mean_sum-stored['mean_pooled_macro_difference'])<1e-12
    directions={}
    for c in clips:
     group=c['group'];diffs=[old['arms'][left]['runs'][j]['metrics']['calibrated']['per_video'][group]['macro_f1_present']-old['arms'][right]['runs'][j]['metrics']['calibrated']['per_video'][group]['macro_f1_present']for j in range(3)]if per[group]is not None else None;directions[group]=diffs
     csv_rows.append({'policy':policy,'comparison':key,'group':group,'query_count':group_coverage[group][policy+'_queries'],'mean_f1_difference':per[group],'seed0_difference':diffs[0]if diffs else None,'seed1_difference':diffs[1]if diffs else None,'seed2_difference':diffs[2]if diffs else None})
    comparisons[key]={'primary':i==0,'per_group_three_seed_differences':directions,'influence':influence(per),'pooled_class_contributions':contribution,'pooled_macro_difference':mean_sum,'original_video_bootstrap':stored['conditional_video_bootstrap']}
   model_identity={}
   if present==[1]:
    for arm,r in old['arms'].items():
     model_identity[arm]=[{'precision':1.,'recall':run['metrics']['calibrated']['pooled']['accuracy'],'f1':run['metrics']['calibrated']['pooled']['macro_f1_present'],'f1_from_recall':2*run['metrics']['calibrated']['pooled']['accuracy']/(1+run['metrics']['calibrated']['pooled']['accuracy'])}for run in r['runs']]
     assert all(abs(v['f1']-v['f1_from_recall'])<1e-12 for v in model_identity[arm])
   results[policy]={'present_GT_classes':present,'constant_rule_diagnostics':diagnostics,'comparisons':comparisons,'person_only_metric_identity':model_identity}
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'group_annotation_and_query_coverage':group_coverage,'results':results,'source_models_predictions_and_selection_unchanged':True})
  with(ROOT/'per_video.csv').open('w',newline='',encoding='utf-8-sig')as f:
   writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csv_rows)
  print('DONE influence and scoring-scope audit',len(csv_rows),flush=True)
if __name__=='__main__':main()
