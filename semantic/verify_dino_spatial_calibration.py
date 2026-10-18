"""Independent point-RLE GT, source features, 21 logits and calibration curves."""
from collections import defaultdict
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco
import torch
from semantic import dino_spatial_calibration as work
from semantic.verify_external_semantic_confirmation import independent_features,metric,equal,linear
from semantic.verify_dino_backbone_training import independent_b14,independent_eval
from semantic.verify_coco_uniform_queries import independent_metrics
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def diagnose(values,target,times,offset):
 r=independent_eval(values,target,times,offset);m=np.asarray(r['confusion_matrix']);bg=m[0].sum();fg=m[1:].sum();r['background_to_foreground_rate']=float(m[0,1:].sum()/bg)if bg else None;r['foreground_class_confusion_rate']=float((m[1:,1:].sum()-np.trace(m[1:,1:]))/fg)if fg else None;return r

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==78 and p['primary_model_fixed']=='B14_aux_sam'and not p['model_selection']
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 clips=json.loads((ROOT/'manifest.json').read_text())['val'];assert [c['video_id']for c in clips]==p['video_ids'];payload=json.loads(work.ANNOTATIONS.read_text());videos={a['id']:a for a in payload['videos']};annotations=defaultdict(list)
 for a in payload['annotations']:annotations[a['video_id']].append(a)
 coverage=[];queries=[];packages={bone:[]for bone in ('S14','B14')}
 for clip in clips:
  vid=clip['video_id'];info=videos[vid];points=[];targets=[]
  for t in range(8):
   selected=np.random.default_rng(20261078+vid*16+t).choice(16384,256,replace=False);xy=np.column_stack((selected%128,selected//128,np.full(256,t))).astype(np.int16);assert len(np.unique(xy[:,:2],axis=0))==256;global_t=clip['start']+t;assert str(clip['frames'][t]).endswith(info['file_names'][global_t]);hit_count=np.zeros(256,np.int64);category=np.zeros(256,np.int16)
   for ann in annotations[vid]:
    if global_t>=len(ann['segmentations'])or ann['segmentations'][global_t]is None:continue
    array=coco.decode(ann['segmentations'][global_t]);array=array.any(axis=2)if array.ndim==3 else array;assert array.shape==(info['height'],info['width']);raster=np.asarray(Image.fromarray(array.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST),bool);hit=raster[xy[:,1],xy[:,0]];hit_count+=hit;category[hit]=ann['category_id']
   labels=np.where(hit_count==0,0,np.where(hit_count==1,category,-1)).astype(np.int16);points.append(xy);targets.append(labels);coverage.append({'video_id':vid,'source_time':t,'valid_queries':int((labels>=0).sum()),'ignored_queries':int((labels<0).sum()),'support':{str(int(k)):int(v)for k,v in zip(*np.unique(labels,return_counts=True))}})
  xy=np.concatenate(points);labels=np.concatenate(targets)
  with np.load(work.qp(vid))as a:assert np.array_equal(xy,a['xy'])and np.array_equal(labels,a['labels'])and int(a['video_id'])==vid
  queries.append({'xy':xy,'labels':labels})
  for bone,dim in [('S14',384),('B14',768)]:
   with np.load(work.fp(bone,vid))as a:assert np.array_equal(xy,a['xy'])and int(a['video_id'])==vid and a['decoded'].shape==(2048,dim)and a['decoded'].dtype==np.float16 and np.isfinite(a['decoded']).all();packages[bone].append(a['decoded'].copy())
  print('verified source uniform RLE point labels',vid,flush=True)
 assert coverage==json.loads((ROOT/'coverage.json').read_text())and file_hash(ROOT/'coverage.json')==s['coverage_sha256']
 for key,folder in [('query_hashes','queries'),('feature_hashes','features')]:
  assert set(s[key])=={str(path)for path in (ROOT/folder).rglob('*.npz')}
  for path,h in s[key].items():assert file_hash(path)==h
 for bone in ('S14','B14'):
  model=work.prior.prior.rgb.prior.resolution.load_dino()if bone=='S14'else work.prior.prior.load_b14();actual=independent_features(model,clips[0]['frames'],queries[0]['xy'])if bone=='S14'else independent_b14(model,clips[0]['frames'],queries[0]['xy']);assert np.array_equal(actual,packages[bone][0]);del model;torch.cuda.empty_cache()
 labels=np.concatenate([a['labels']for a in queries]);times=np.concatenate([a['xy'][:,2]for a in queries]);ids=np.repeat([c['video_id']for c in clips],2048).astype(np.int32);valid=labels>=0;target=labels[valid];ts=times[valid];frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert file_hash(ROOT/'frozen_selection.json')==s['frozen_selection_sha256']and frozen['models']==p['models']and frozen['primary_model_fixed']==p['primary_model_fixed']and not frozen['model_selection'];offsets={};per_video={}
 for arm in work.ARMS:
  values=[]
  for seed in range(3):
   path=work.prior.prior.ckpt(arm,seed);assert file_hash(path)==p['models'][arm]['checkpoint_sha256'][str(seed)];state=torch.load(path,map_location='cpu',weights_only=True)['model'];logits=np.concatenate([linear(a,state)for a in packages[arm.split('_')[0]]]);r=s['results'][arm]['runs'][seed];assert file_hash(r['output_path'])==r['output_sha256']
   with np.load(r['output_path'])as a:assert np.array_equal(a['logits'],logits)and np.array_equal(a['labels'],labels)and np.array_equal(a['times'],times)and np.array_equal(a['video_ids'],ids)and np.array_equal(a['valid_mask'],valid)and str(a['checkpoint_sha256'])==file_hash(path)
   values.append(logits[valid]);print('verified spatial calibration raw logits',arm,seed,flush=True)
  curve=[];best=-1;choice=None
  for offset in sorted(p['calibration_grid'],key=lambda x:(abs(x),x)):
   vals=[independent_eval(v,target,ts,offset)['macro_f1_present']for v in values];mean=float(np.mean(vals));curve.append({'offset':float(offset),'macro_f1_by_seed':vals,'mean_macro_f1':mean})
   if mean>best:best=mean;choice=float(offset)
  equal(curve,frozen['curves'][arm]);offsets[arm]=choice;assert choice==s['results'][arm]['new_spatial_offset']and p['models'][arm]['offset']==s['results'][arm]['old_offset']==frozen['old_offsets'][arm];per_video[arm]={}
  for seed,v in enumerate(values):
   expected={};per_video[arm][str(seed)]={}
   for name,offset in [('raw',0),('old_calibration',p['models'][arm]['offset']),('spatial_calibration',choice)]:
    expected[name]=diagnose(v,target,ts,offset);shifted=v.copy();shifted[:,0]+=offset;pred=shifted.argmax(-1);per_video[arm][str(seed)][name]={str(c['video_id']):independent_metrics(pred[ids[valid]==c['video_id']],target[ids[valid]==c['video_id']])for c in clips}
   equal(expected,s['results'][arm]['runs'][seed]['metrics'])
 assert offsets==frozen['spatial_offsets']and s['valid_queries']==int(valid.sum())and s['ignored_queries']==int((~valid).sum())and s['total_coordinates']==40960;atomic_json(ROOT/'per_video_metrics.json',per_video)
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 assert s['new_confirmation_groups']==0 and s['LVVIS40_not_evaluated']and s['deployment_unchanged']=='native_gt';atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all160_original_RLE_point_hit_GT_and_independent_uniform_XY_recomputed':True,'both_first258_T8_feature_replays_bitwise_equal':True,'all21_full_logits_valid_masks_and_raw_old_new_metrics_recomputed':True,'all7_shared_calibration_grids_and_tie_rules_recomputed':True,'per_video_metrics_written_from_independent_predictions_sha256':file_hash(ROOT/'per_video_metrics.json'),'all_frozen_inputs_unchanged':True,'no_head_training_model_selection_or_confirmation':True,'deployment_unchanged':'native_gt'});print('DONE independent source-only spatial calibration audit',offsets,flush=True)
if __name__=='__main__':main()
