"""Exact decomposition of two different macro aggregations; descriptive only."""
import fcntl,json
from pathlib import Path
import numpy as np
from semantic import coco_fusion_confirmation as prior
from semantic.run import atomic_json
from semantic.oracle_probe import file_hash
ROOT=Path('semantic_artifacts/coco_fusion_metric_audit')
PAIRS=(('b14_ade_decoder','b14_zero'),('b14_ade_decoder','z_b14_gt32'),('b14_ade_encoder','b14_mit'))

def counts(pred,target):
 cm=np.bincount((target*26+pred).astype(np.int64),minlength=26*26).reshape(26,26)
 tp=cm.diagonal();support=cm.sum(1);fp=cm.sum(0)-tp;fn=support-tp
 return [{'class':c,'support':int(support[c]),'TP':int(tp[c]),'FP':int(fp[c]),'FN':int(fn[c]),'F1':float(2*tp[c]/max(2*tp[c]+fp[c]+fn[c],1))}for c in range(26)]

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists():raise RuntimeError('Completed metric decomposition exists')
  source=json.loads((prior.ROOT/'summary.json').read_text());assert file_hash(prior.ROOT/'summary.json')==json.loads((prior.ROOT/'verification.json').read_text())['summary_sha256']
  manifest=json.loads((prior.ROOT/'manifest.json').read_text());images=manifest['candidate'];ids=np.repeat([im['id']for im in images],1024);targets={policy:[]for policy in prior.POLICIES}
  paths=[prior.ROOT/name for name in ('summary.json','verification.json','manifest.json','protocol.json')]+[Path('semantic/coco_fusion_metric_audit.py'),Path('semantic/OVIS_COCO_FUSION_METRIC_AUDIT_PROTOCOL.md'),prior.models.ROOT/'summary.json']
  paths.extend(Path(x)for x in source['query_hashes']);paths.extend(Path(r['output_path'])for v in source['results'].values()for r in v['runs'])
  p={'round':87,'descriptive_only':True,'pairs':[list(pair)for pair in PAIRS],'policies':list(prior.POLICIES),'profiles':list(prior.PROFILES),'new_model_inference':0,'new_confirmations':0,'input_sha256':{str(path):file_hash(path)for path in paths}};assert p==json.loads((ROOT/'protocol.json').read_text())
  for im in images:
   with np.load(prior.qp(im['id']))as a:
    for policy in prior.POLICIES:targets[policy].append(a[policy].copy())
  targets={key:np.concatenate(v)for key,v in targets.items()};preds={};tables={}
  for arm,r in source['results'].items():
   preds[arm]={profile:[]for profile in prior.PROFILES};tables[arm]={policy:{profile:[]for profile in prior.PROFILES}for policy in prior.POLICIES}
   for run in r['runs']:
    with np.load(run['output_path'])as a:values=a['logits'].copy();assert np.array_equal(ids,a['image_ids'])
    for profile,offset in r['offsets'].items():
     shifted=values.copy();shifted[:,0]+=offset;pred=shifted.argmax(-1);preds[arm][profile].append(pred)
     for policy,target in targets.items():valid=target>=0;tables[arm][policy][profile].append(counts(pred[valid],target[valid]))
  source_val=json.loads((prior.models.ROOT/'summary.json').read_text());audits={}
  for left,right in PAIRS:
   key=left+'_minus_'+right;audits[key]={}
   for policy,target in targets.items():
    valid=target>=0;classes=sorted(map(int,np.unique(target[valid])));n=sum(bool(np.any(valid&(ids==im['id'])))for im in images);audits[key][policy]={}
    for profile in prior.PROFILES:
     rows=[];image_contributions={c:0. for c in classes};image_cells=[]
     for im in images:
      mask=valid&(ids==im['id']);y=target[mask]
      if not len(y):continue
      present=sorted(map(int,np.unique(y)));perseed=[]
      for seed in range(3):
       lc=counts(preds[left][profile][seed][mask],y);rc=counts(preds[right][profile][seed][mask],y)
       perseed.append({c:(lc[c]['F1']-rc[c]['F1'])/(n*len(present))for c in present})
      for c in present:
       value=float(np.mean([a[c]for a in perseed]));image_contributions[c]+=value;image_cells.append({'image_id':im['id'],'class':c,'support':int(np.sum(y==c)),'image_present_classes':len(present),'three_seed_mean_weighted_contribution':value})
     for c in classes:
      delta=[tables[left][policy][profile][seed][c]['F1']-tables[right][policy][profile][seed][c]['F1']for seed in range(3)]
      src=[]
      for seed in range(3):
       lf=source_val['results'][left]['runs'][seed]['spatial_val'][profile]['per_class_f1'].get(str(c));rf=source_val['results'][right]['runs'][seed]['spatial_val'][profile]['per_class_f1'].get(str(c));src.append(None if lf is None or rf is None else lf-rf)
      rows.append({'class':c,'support':int(np.sum(target==c)),'images_with_target_class':sum(bool(np.any((target==c)&(ids==im['id'])))for im in images),'per_seed_F1_difference':delta,'mean_pooled_macro_contribution':float(np.mean(delta)/len(classes)),'mean_image_equal_macro_contribution':image_contributions[c],'source_spatial_val_same_class_F1_difference':src})
     transitions={}
     for group,gmask in [('background',target==0),('person',target==1),('nonperson_FG',target>1)]:
      matrices=[]
      for seed in range(3):
       y=target[gmask];lp=preds[left][profile][seed][gmask];rp=preds[right][profile][seed][gmask]
       if group=='background':ls=(lp!=0).astype(int);rs=(rp!=0).astype(int);size=2
       else:ls=np.where(lp==y,0,np.where(lp==0,1,2));rs=np.where(rp==y,0,np.where(rp==0,1,2));size=3
       matrices.append(np.bincount(rs*size+ls,minlength=size*size).reshape(size,size).tolist())
      transitions[group]={'rows_right_columns_left':matrices,'states':['correct_BG','wrong_FG']if group=='background'else['correct_class','miss_BG','wrong_other_FG']}
     pooled=float(sum(r['mean_pooled_macro_contribution']for r in rows));image_equal=float(sum(image_contributions.values()))
     pooled_ref=float(np.mean([source['results'][left]['runs'][seed]['policies'][policy][profile]['macro_f1_present']-source['results'][right]['runs'][seed]['policies'][policy][profile]['macro_f1_present']for seed in range(3)]))
     image_ref=float(np.mean([np.mean([source['results'][left]['runs'][seed]['policies'][policy]['per_image'][str(im['id'])][profile]['macro_f1_present']-source['results'][right]['runs'][seed]['policies'][policy]['per_image'][str(im['id'])][profile]['macro_f1_present']for seed in range(3)])for im in images]))
     assert abs(pooled-pooled_ref)<1e-12 and abs(image_equal-image_ref)<1e-12
     audits[key][policy][profile]={'present_classes':classes,'valid_images':n,'per_class':rows,'image_class_cells':image_cells,'pooled_macro_difference_reconstructed':pooled,'image_equal_macro_difference_reconstructed':image_equal,'same_point_error_transitions':transitions}
  for path,h in p['input_sha256'].items():assert file_hash(path)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,'all_five_group_class_tables':tables,'audits':audits,'amendment_sha256':file_hash(ROOT/'amendment.json'),'no_training_recalibration_new_confirmation_or_deployment_change':True});print('DONE frozen metric weight decomposition',flush=True)
if __name__=='__main__':main()
