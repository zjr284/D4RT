"""Boolean per-class counts and explicit image/class weights, independent of CM audit."""
import json
import numpy as np
from semantic import coco_fusion_confirmation as confirm
from semantic import coco_fusion_metric_audit as work
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from semantic.verify_external_semantic_confirmation import equal
ROOT=work.ROOT

def count_one(pred,y,c):
 tp=int(np.sum((pred==c)&(y==c)));fp=int(np.sum((pred==c)&(y!=c)));fn=int(np.sum((pred!=c)&(y==c)))
 return {'class':c,'support':int(np.sum(y==c)),'TP':tp,'FP':fp,'FN':fn,'F1':2*tp/max(2*tp+fp+fn,1)}

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text()) and p['round']==87 and p['descriptive_only']
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 amendment=json.loads((ROOT/'amendment.json').read_text());assert file_hash(ROOT/'amendment.json')==s['amendment_sha256'] and file_hash(ROOT/'protocol.json')==amendment['original_protocol_sha256'] and file_hash('semantic/coco_fusion_metric_audit.py')==amendment['original_script_sha256'] and file_hash('semantic/coco_fusion_metric_audit_resume.py')==amendment['resume_script_sha256'] and file_hash(amendment['preserved_first_failure_log'])==amendment['preserved_first_failure_log_sha256']
 assert 'TypeError'in open(amendment['preserved_first_failure_log']).read() and not amendment['changes_to_queries_predictions_scores_or_model_parameters']
 source=json.loads((confirm.ROOT/'summary.json').read_text());source_val=json.loads((confirm.models.ROOT/'summary.json').read_text());images=json.loads((confirm.ROOT/'manifest.json').read_text())['candidate'];ids=np.repeat([im['id']for im in images],1024)
 targets={policy:[]for policy in confirm.POLICIES}
 for im in images:
  with np.load(confirm.qp(im['id']))as a:
   for policy in confirm.POLICIES:targets[policy].append(a[policy].copy())
 targets={policy:np.concatenate(v)for policy,v in targets.items()};preds={};tables={}
 for arm,r in source['results'].items():
  preds[arm]={};tables[arm]={policy:{}for policy in confirm.POLICIES}
  for profile,offset in r['offsets'].items():
   values=[]
   for run in r['runs']:
    with np.load(run['output_path'])as a:logits=a['logits'].copy();assert np.array_equal(ids,a['image_ids'])
    logits[:,0]+=offset;values.append(logits.argmax(-1))
   preds[arm][profile]=values
   for policy,target in targets.items():
    valid=target>=0;tables[arm][policy][profile]=[[count_one(pred[valid],target[valid],c)for c in range(26)]for pred in values]
 equal(tables,s['all_five_group_class_tables']);audit_count=0;missing_source_classes=set()
 assert s['protocol']['pairs']==[list(pair)for pair in work.PAIRS]
 for left,right in work.PAIRS:
  key=left+'_minus_'+right
  for policy,target in targets.items():
   valid=target>=0;classes=list(map(int,np.unique(target[valid])));n=sum(np.any(valid&(ids==im['id']))for im in images)
   for profile in confirm.PROFILES:
    a=s['audits'][key][policy][profile];assert a['present_classes']==classes and a['valid_images']==n
    cells=[];contributions={c:0. for c in classes}
    for im in images:
     mask=valid&(ids==im['id']);y=target[mask];present=list(map(int,np.unique(y)))
     if not present:continue
     for c in present:
      diff=[]
      for seed in range(3):diff.append(count_one(preds[left][profile][seed][mask],y,c)['F1']-count_one(preds[right][profile][seed][mask],y,c)['F1'])
      value=float(np.mean(diff)/(int(n)*len(present)));contributions[c]+=value;cells.append({'image_id':im['id'],'class':c,'support':int(np.sum(y==c)),'image_present_classes':len(present),'three_seed_mean_weighted_contribution':value})
    equal(cells,a['image_class_cells']);rows=[]
    for c in classes:
     delta=[tables[left][policy][profile][seed][c]['F1']-tables[right][policy][profile][seed][c]['F1']for seed in range(3)];src=[]
     for seed in range(3):
      l=source_val['results'][left]['runs'][seed]['spatial_val'][profile];r=source_val['results'][right]['runs'][seed]['spatial_val'][profile]
      if str(c)not in l['support']or str(c)not in r['support']:
       assert str(c)not in l['per_class_f1']and str(c)not in r['per_class_f1'];src.append(None);missing_source_classes.add(c)
      else:src.append(l['per_class_f1'][str(c)]-r['per_class_f1'][str(c)])
     rows.append({'class':c,'support':int(np.sum(target==c)),'images_with_target_class':int(sum(np.any((target==c)&(ids==im['id']))for im in images)),'per_seed_F1_difference':delta,'mean_pooled_macro_contribution':float(np.mean(delta)/len(classes)),'mean_image_equal_macro_contribution':contributions[c],'source_spatial_val_same_class_F1_difference':src})
    equal(rows,a['per_class']);pooled=float(sum(r['mean_pooled_macro_contribution']for r in rows));image_mean=float(sum(contributions.values()));equal(pooled,a['pooled_macro_difference_reconstructed']);equal(image_mean,a['image_equal_macro_difference_reconstructed'])
    # Reproduce existing confirmation scalar contrasts as well as additive cells.
    expected_pooled=float(np.mean([source['results'][left]['runs'][seed]['policies'][policy][profile]['macro_f1_present']-source['results'][right]['runs'][seed]['policies'][policy][profile]['macro_f1_present']for seed in range(3)]));equal(pooled,expected_pooled)
    expected_image=float(np.mean([np.mean([source['results'][left]['runs'][seed]['policies'][policy]['per_image'][str(im['id'])][profile]['macro_f1_present']-source['results'][right]['runs'][seed]['policies'][policy]['per_image'][str(im['id'])][profile]['macro_f1_present']for seed in range(3)])for im in images]));equal(image_mean,expected_image)
    for group,mask in [('background',target==0),('person',target==1),('nonperson_FG',target>1)]:
     matrices=[]
     for seed in range(3):
      y=target[mask];lp=preds[left][profile][seed][mask];rp=preds[right][profile][seed][mask]
      if group=='background':ls=np.asarray(lp!=0,int);rs=np.asarray(rp!=0,int);size=2
      else:ls=np.select([lp==y,lp==0],[0,1],default=2);rs=np.select([rp==y,rp==0],[0,1],default=2);size=3
      matrix=[[int(np.sum((rs==i)&(ls==j)))for j in range(size)]for i in range(size)];assert sum(map(sum,matrix))==int(mask.sum());matrices.append(matrix)
     equal(matrices,a['same_point_error_transitions'][group]['rows_right_columns_left'])
    audit_count+=1
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['no_training_recalibration_new_confirmation_or_deployment_change']
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all_five_groups_three_seeds_two_policies_three_profiles_boolean_class_counts_recomputed':True,'all18_class_weight_decompositions_and_both_confirmation_scalar_sums_recomputed':audit_count==18,'all_image_class_cells_and_same_point_error_cartesian_counts_recomputed':True,'source_missing_classes_preserved_as_null':sorted(missing_source_classes),'first_failure_and_explicit_amendment_provenance_verified':True,'all_frozen_inputs_unchanged':True,'no_model_training_calibration_new_confirmation_or_deployment_change':True});print('DONE independent exact metric weighting decomposition',audit_count,sorted(missing_source_classes),flush=True)
if __name__=='__main__':main()
