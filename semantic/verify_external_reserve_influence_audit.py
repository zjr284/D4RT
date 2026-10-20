"""Independent matrix metrics, analytic constant rules and sum-based omissions."""
from collections import defaultdict,Counter
from itertools import combinations
import csv,json,math
import numpy as np
from semantic import external_reserve_influence_audit as work
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT;source=work.source

def metrics(y,pred):
 m=np.zeros((26,26),np.int64);np.add.at(m,(y,pred),1);n=m.sum(1);pp=m.sum(0);present=np.flatnonzero(n);f1={str(int(c)):float(2*m[c,c]/(n[c]+pp[c]))for c in present};return {'macro':sum(f1.values())/len(f1),'class':f1,'accuracy':float(np.trace(m)/len(y))}

def near(a,b):assert abs(a-b)<1e-12,(a,b)
def main():
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==95
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 old=json.loads((source.ROOT/'summary.json').read_text());clips=json.loads((source.ROOT/'manifest.json').read_text())['confirmation'];payload=json.loads(source.ANN.read_bytes());names={c['id']:c['name']for c in payload['categories']};mapping={int(k):int(v)for k,v in json.loads(source.old.MAPPING_SUMMARY.read_text())['shared_category_mapping'].items()};by=defaultdict(list)
 for a in payload['annotations']:by[a['image_id']].append(a)
 for clip in clips:
  ann=[a for im in clip['frames']for a in by[im['id']]];r=s['group_annotation_and_query_coverage'][clip['group']];assert r['all_annotations']==len(ann)and r['annotation_category_counts']==dict(Counter(names[a['category_id']]for a in ann));n=sum(a['category_id']in mapping and not a.get('iscrowd',0)for a in ann);assert r['mapped_noncrowd_annotations']==n
  for pol in source.POLICIES:
   with np.load(source.qp(pol,clip['group']))as a:assert len(a['labels'])==r[pol+'_queries']
  assert r['strict_empty_reason']==(None if r['strict_queries']else'no_mapped_noncrowd_annotations'if n==0 else'mapped_annotations_filtered_by_fixed_mask_rules')
 for pol in source.POLICIES:
  q=s['results'][pol];replay={};targets=None
  for arm in source.ARMS:
   replay[arm]=[]
   for seed in range(3):
    with np.load(old['results'][pol]['arms'][arm]['runs'][seed]['output_path'])as a:z=a['logits'].copy();y=a['labels'].copy();ids=a['group_ids'].copy()
    z[:,0]+=old['protocol']['models'][arm]['offset'];pred=z.argmax(1);targets=y if targets is None else targets;assert np.array_equal(y,targets);replay[arm].append({'pooled':metrics(y,pred),'per':{c['group']:metrics(y[ids==int(c['group'])],pred[ids==int(c['group'])])if np.any(ids==int(c['group']))else None for c in clips}})
  present=sorted(map(int,np.unique(targets)));assert q['present_GT_classes']==present;hist=np.bincount(targets,minlength=26)
  for c,r in enumerate(q['constant_rule_diagnostics']):
   assert r['constant_prediction_class']==c;near(r['accuracy'],hist[c]/len(targets));near(r['macro_f1_present'],(2*hist[c]/(len(targets)+hist[c]))/len(present)if hist[c]else 0.)
  for left,right in p['comparisons']:
   item=q['comparisons'][left+'_minus_'+right];per={}
   for c in clips:
    group=c['group'];a=replay[left][0]['per'][group]
    if a is None:assert item['per_group_three_seed_differences'][group]is None;continue
    diffs=[replay[left][j]['per'][group]['macro']-replay[right][j]['per'][group]['macro']for j in range(3)];assert np.allclose(diffs,item['per_group_three_seed_differences'][group],rtol=0,atol=1e-12);per[group]=sum(diffs)/3
   influence=item['influence'];near(influence['full_video_equal_mean'],sum(per.values())/len(per));assert influence['valid_groups']==len(per)
   for n in (1,2):
    rows=influence['leave_out'][str(n)]['all_omission_subsets'];assert len(rows)==math.comb(len(per),n);expected=list(combinations(per,n));assert[tuple(r['omitted_groups'])for r in rows]==expected;means=[]
    for row,omitted in zip(rows,expected):
     value=(sum(per.values())-sum(per[k]for k in omitted))/(len(per)-n);near(row['remaining_mean'],value);means.append(value)
    r=influence['leave_out'][str(n)];near(r['min'],min(means));near(r['max'],max(means));assert(r['positive'],r['negative'],r['tied'])==(sum(v>0 for v in means),sum(v<0 for v in means),sum(v==0 for v in means))
   positive=sorted([(k,v)for k,v in per.items()if v>0],key=lambda a:(-a[1],a[0]));assert[r['group']for r in influence['positive_groups_by_contribution']]==[k for k,v in positive];gain=sum(v for k,v in positive)
   if gain:near(influence['top2_fraction_of_positive_contributions'],sum(v for k,v in positive[:2])/gain)
   total=0.
   for c in present:
    r=item['pooled_class_contributions'][str(c)];diffs=[replay[left][j]['pooled']['class'][str(c)]-replay[right][j]['pooled']['class'][str(c)]for j in range(3)];assert np.allclose(diffs,r['per_seed_f1_differences'],rtol=0,atol=1e-12);assert np.allclose(np.array(diffs)/len(present),r['per_seed_macro_contributions'],rtol=0,atol=1e-12);value=sum(diffs)/3/len(present);near(value,r['mean_macro_contribution']);total+=value
   near(total,item['pooled_macro_difference']);near(total,sum(replay[left][j]['pooled']['macro']-replay[right][j]['pooled']['macro']for j in range(3))/3)
  if present==[1]:
   for arm in source.ARMS:
    for j,r in enumerate(q['person_only_metric_identity'][arm]):near(r['recall'],replay[arm][j]['pooled']['accuracy']);near(r['f1'],2*r['recall']/(1+r['recall']));near(r['precision'],1.)
 rows=list(csv.DictReader((ROOT/'per_video.csv').open(encoding='utf-8-sig')));assert len(rows)==480
 for row in rows:
  diffs=s['results'][row['policy']]['comparisons'][row['comparison']]['per_group_three_seed_differences'][row['group']];expected=s['group_annotation_and_query_coverage'][row['group']][row['policy']+'_queries'];assert int(row['query_count'])==expected
  if diffs is None:assert row['mean_f1_difference']==''and all(row[f'seed{j}_difference']==''for j in range(3))
  else:
   near(float(row['mean_f1_difference']),sum(diffs)/3)
   for j in range(3):near(float(row[f'seed{j}_difference']),diffs[j])
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'CSV_sha256':file_hash(ROOT/'per_video.csv'),'all36_frozen_outputs_metrics_rebuilt':True,'all12_comparisons_full_leave_one_and_two_recomputed':True,'all_class_contributions_sum_to_pooled_difference':True,'all52_constant_rules_analytically_verified':True,'all40_annotation_and_empty_group_records_verified':True,'all480_CSV_rows_verified':True,'all_source_fingerprints_unchanged':True});print('DONE independent influence and scope audit',flush=True)
if __name__=='__main__':main()
