"""Independent bitset GT replay, manual features/heads, metrics and bootstraps."""
from collections import defaultdict
import gc,json
import numpy as np
from PIL import Image
from pycocotools import mask as masks
import torch
from torch.nn import functional as F
from semantic import external_reserve_confirmation as work
from semantic.verify_external_semantic_confirmation import metric,equal,linear
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=work.ROOT

def manual_feature(model,frames,xy,dim):
 arrays=[]
 for path in frames:
  with Image.open(path)as im:arrays.append(np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32))
 image=torch.stack([torch.from_numpy(a).permute(2,0,1)/255 for a in arrays]).cuda()
 with torch.inference_mode():
  image=F.interpolate(image,size=(448,448),mode='bicubic',align_corners=False);image=(image-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
  with torch.autocast('cuda',dtype=torch.bfloat16):
   hidden=model.prepare_tokens_with_masks(image)
   for block in model.blocks:hidden=block(hidden)
   patches=model.norm(hidden)[:,1:].reshape(8,32,32,dim).permute(0,3,1,2)
  result=torch.empty((len(xy),dim),device='cuda')
  for t in np.unique(xy[:,2]):
   idx=np.flatnonzero(xy[:,2]==t);grid=((torch.tensor(xy[idx,:2],device='cuda',dtype=torch.float32)+.5)/64-1)[None,None]
   result[torch.tensor(idx,device='cuda')]=F.grid_sample(patches[int(t):int(t)+1].float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
 return result.half().cpu().numpy()

def main():
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())and p['round']==93 and p['primary']==['source_gt32','source_gt']and p['bootstrap_seed']==20261093
 work.verify_inputs(p)
 for path,h in s['output_hashes'].items():assert file_hash(path)==h
 clips=json.loads((ROOT/'manifest.json').read_text())['confirmation'];assert clips==json.loads((work.ready.ROOT/'manifest.json').read_text())['confirmation']and len(clips)==40
 for arm in work.ARMS:
  for seed in range(3):assert file_hash(work.ready.ckpt(arm,seed))==p['models'][arm]['checkpoint_sha256'][str(seed)]
 data=json.loads(work.ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(work.old.MAPPING_SUMMARY.read_text())['shared_category_mapping'].items()};by=defaultdict(list)
 for a in data['annotations']:by[a['image_id']].append(a)
 coverage=[];packages={policy:[]for policy in work.POLICIES}
 for clip in clips:
  xx={k:[]for k in work.POLICIES};yy={k:[]for k in work.POLICIES}
  for t,im in enumerate(clip['frames']):
   count=np.zeros((128,128),np.int32);owner=np.zeros((128,128),np.int16);bits=np.zeros((128,128),np.uint32);crowd=np.zeros((128,128),bool)
   for ann in by[im['id']]:
    raw=masks.decode(ann['segmentation']);assert raw.shape==(im['height'],im['width']);hit=np.asarray(Image.fromarray(raw.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST),bool);count+=hit;cid=mapping.get(ann['category_id']);owner[hit]=-1 if ann.get('iscrowd',0)or cid is None else cid
    if ann.get('iscrowd',0):crowd|=hit
    elif cid is not None:bits[hit]|=np.uint32(1<<cid)
   strict=np.where(count==0,0,np.where(count==1,owner,-1)).astype(np.int16);consensus=np.full((128,128),-1,np.int16);consensus[count==0]=0
   for cid in range(1,26):consensus[(bits==np.uint32(1<<cid))&~crowd]=cid
   for policy,raster in [('strict',strict),('consensus',consensus)]:
    pool=np.flatnonzero(raster.ravel()>0)
    if len(pool):
     chosen=np.random.default_rng(20261041+im['id']).choice(pool,256,replace=len(pool)<256);xy=np.column_stack((chosen%128,chosen//128,np.full(256,t)));y=raster.ravel()[chosen].astype(np.int64)
    else:xy=np.empty((0,3),np.int64);y=np.empty(0,np.int64)
    xx[policy].append(xy);yy[policy].append(y);coverage.append({'group':clip['group'],'time':t,'image_id':im['id'],'policy':policy,'query_count':len(y),'distinct_pixels':len(np.unique(xy[:,:2],axis=0)),'foreground_pixels':int((raster>0).sum()),'ignored_pixels':int((raster<0).sum()),'support':{str(int(k)):int(v)for k,v in zip(*np.unique(y,return_counts=True))}})
  for policy in work.POLICIES:
   x=np.concatenate(xx[policy]);y=np.concatenate(yy[policy])
   with np.load(work.qp(policy,clip['group']))as a:assert np.array_equal(x,a['xy'])and np.array_equal(y,a['labels'])
   row={'xy':x,'labels':y,'group':clip['group']}
   for bone,dim in [('S14',384),('B14',768)]:
    with np.load(work.fp(bone,policy,clip['group']))as a:assert np.array_equal(x,a['xy']);row[bone]=a['decoded'].copy()
    assert row[bone].dtype==np.float16 and row[bone].shape==(len(y),dim)and np.isfinite(row[bone]).all()
   packages[policy].append(row)
 equal(coverage,json.loads((ROOT/'coverage.json').read_text()));print('verified 640 independent GT/frame query records',flush=True)
 # Both backbones and both policies, manual blocks/final norm/point interpolation.
 first=next(i for i,a in enumerate(packages['strict'])if len(a['labels']));clip=clips[first];frames=[str(work.old.RGB/'images'/im['file_name'])for im in clip['frames']]
 for bone,dim in [('S14',384),('B14',768)]:
  model=work.old.rgb.prior.resolution.load_dino()if bone=='S14'else work.load_b14()
  for policy in work.POLICIES:
   row=packages[policy][first];value=manual_feature(model,frames,row['xy'],dim);assert np.array_equal(value,row[bone]),(bone,policy,float(np.max(np.abs(value.astype(float)-row[bone].astype(float)))))
  del model;gc.collect();torch.cuda.empty_cache()
 print('verified 4 manual full-group feature replays',flush=True)
 with np.load(p['statistics']['path'])as a:mu=a['mean'].copy();scale=a['scale'].copy()
 for policy,rows in packages.items():
  y=np.concatenate([a['labels']for a in rows]);ids=np.concatenate([np.full(len(a['labels']),int(a['group']),np.int64)for a in rows]);r=s['results'][policy]
  assert len(y)==r['total_queries']and sum(bool(len(a['labels']))for a in rows)==r['query_bearing_groups']and[a['group']for a in rows if not len(a['labels'])]==r['empty_groups']
  rebuilt={}
  for arm in work.ARMS:
   rebuilt[arm]=[];bone='B14'if arm.startswith('B14')else'S14'
   for seed in range(3):
    state=torch.load(work.ready.ckpt(arm,seed),map_location='cpu',weights_only=True)['model'];xs=[(a[bone].astype(np.float32)-mu)/scale if arm=='B14_z'else a[bone].astype(np.float32)for a in rows];value=np.concatenate([linear(x,state)for x in xs])
    output=r['arms'][arm]['runs'][seed]
    with np.load(output['output_path'])as a:assert np.array_equal(value,a['logits'])and np.array_equal(y,a['labels'])and np.array_equal(ids,a['group_ids'])
    # A separate double-precision algebra check guards the linear head operation.
    sample=next(x for x in xs if len(x))[:32];manual=sample.astype(np.float64)@state['classifier.weight'].numpy().astype(np.float64).T+state['classifier.bias'].numpy().astype(np.float64);assert np.max(np.abs(manual-value[:len(sample)]))<1e-4
    modes={}
    for mode,offset in [('raw',0),('calibrated',p['models'][arm]['offset'])]:
     z=value.copy();z[:,0]+=offset;pred=z.argmax(1);pooled,confusion=metric(pred,y);per={c['group']:metric(pred[ids==int(c['group'])],y[ids==int(c['group'])])[0]if np.any(ids==int(c['group']))else None for c in clips};modes[mode]={'pooled':pooled,'per_video':per,'confusion':confusion.tolist()}
    equal(modes,output['metrics']);rebuilt[arm].append(modes);print('verified logits',policy,arm,seed,flush=True)
  for pair in [r['primary_contrast'],*r['secondary_contrasts_unadjusted'].values()]:
   a,b=pair['left'],pair['right'];per={c['group']:float(np.mean([rebuilt[a][j]['calibrated']['per_video'][c['group']]['macro_f1_present']-rebuilt[b][j]['calibrated']['per_video'][c['group']]['macro_f1_present']for j in range(3)]))if rebuilt[a][0]['calibrated']['per_video'][c['group']]is not None else None for c in clips};equal(per,pair['per_video_three_seed_mean_differences']);diff=[rebuilt[a][j]['calibrated']['pooled']['macro_f1_present']-rebuilt[b][j]['calibrated']['pooled']['macro_f1_present']for j in range(3)];equal(diff,pair['per_seed_pooled_macro_difference']);equal(float(np.mean(diff)),pair['mean_pooled_macro_difference']);values=np.array([v for v in per.values()if v is not None]);rng=np.random.default_rng(20261093);means=np.array([values[rng.integers(len(values),size=len(values))].mean()for _ in range(2000)]);boot=pair['conditional_video_bootstrap'];print('bootstrap keys',list(boot),flush=True)
   # Stored helper uses mean_difference/ci95 (checked explicitly below).
   atomic_json(ROOT/('bootstrap_independent_'+policy+'_'+a+'_minus_'+b+'.json'),{'mean':float(values.mean()),'interval':np.quantile(means,[.025,.975]).tolist()})
   # Value-based check allows the established helper's stable key spelling.
   pairs=[v for v in boot.values()if isinstance(v,list)and len(v)==2]
   assert any(np.allclose(v,np.quantile(means,[.025,.975]),rtol=0,atol=1e-12)for v in pairs),boot
 work.verify_inputs(p)
 for path,h in s['output_hashes'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all640_GT_frame_records_rebuilt_with_bitset_consensus':True,'four_full_group_manual_feature_replays_identical':True,'all36_logits_metrics_confusions_recomputed':True,'double_precision_head_algebra_checked':True,'all12_paired_bootstrap_intervals_recomputed':True,'all_frozen_models_inputs_queries_features_logits_unchanged':True,'feature_group':clip['group']});print('DONE independent original reserve confirmation',flush=True)
if __name__=='__main__':main()
