"""Independent GT/empty-query replay, model features, all logits and paired inference."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import torch
import torch.nn.functional as F
from semantic.external_semantic_confirmation import ROOT,RGB,REVIEW,ANN,OVIS,ARMS,feature_path
from semantic import dino_native_rgb_control as rgb
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def metric(pred,target):
    matrix=np.bincount(target.astype(np.int64)*26+pred.astype(np.int64),minlength=26*26).reshape(26,26)
    support=matrix.sum(1);count=matrix.sum(0);present=np.flatnonzero(support)
    f1={str(int(c)):float(2*matrix[c,c]/(support[c]+count[c]))for c in present}
    return {'accuracy':float(np.trace(matrix)/matrix.sum()),'macro_f1_present':float(np.mean(list(f1.values()))),
        'foreground_macro_f1_present':float(np.mean([f1[str(int(c))]for c in present if c>0])),
        'support':{str(int(c)):int(support[c])for c in present},'per_class_f1':f1},matrix


def equal(a,b):
    if isinstance(a,dict):assert set(a)==set(b);[equal(a[k],b[k])for k in a]
    elif isinstance(a,(list,tuple)):assert len(a)==len(b);[equal(x,y)for x,y in zip(a,b)]
    elif isinstance(a,float):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)


def linear(features,state):
    with torch.inference_mode():return F.linear(torch.from_numpy(features).float().cuda(),state['classifier.weight'].cuda(),state['classifier.bias'].cuda()).cpu().numpy()


def independent_features(model,frames,xy):
    arrays=[]
    for path in frames:
        with Image.open(path)as im:arrays.append(np.asarray(im.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),dtype=np.float32))
    video=torch.stack([torch.from_numpy(a).permute(2,0,1)/255 for a in arrays]).cuda()
    with torch.inference_mode():
        mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        normalized=(F.interpolate(video.float(),size=(448,448),mode='bicubic',align_corners=False)-mean)/std
        with torch.autocast('cuda',dtype=torch.bfloat16):tokens=model.forward_features(normalized)['x_norm_patchtokens'].reshape(8,32,32,384).permute(0,3,1,2)
        out=torch.empty((len(xy),384),dtype=torch.float32,device='cuda')
        for time in sorted(set(map(int,xy[:,2]))):
            rows=np.flatnonzero(xy[:,2]==time);coordinates=torch.from_numpy(xy[rows,:2]).float().cuda()
            grid=(((coordinates+.5)/128)*2-1)[None,None]
            values=F.grid_sample(tokens[time:time+1].float(),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
            out[torch.as_tensor(rows,device='cuda')]=values
    return out.half().cpu().numpy()


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert frozen['selected']==p['selected_by_OVIS_validation']=='source_gt32'and frozen['selection']==p['selection']
    manifest=json.loads((ROOT/'manifest.json').read_text())['confirmation'];assert [c['group']for c in manifest]==p['candidate_groups']
    assert {path.stem for path in (ROOT/'features').glob('*.npz')}==set(p['candidate_groups'])and not set(p['reserve_groups'])&set(p['candidate_groups'])
    payload=json.loads(ANN.read_bytes());categories=json.loads(OVIS.read_bytes())['categories'];normalize=lambda x:' '.join(x.lower().replace('_',' ').split())
    targets={normalize(c['name']):c['id']for c in categories};mapping={c['id']:targets[normalize(c['name'])]for c in payload['categories']if normalize(c['name'])in targets}
    by_image=defaultdict(list)
    for ann in payload['annotations']:by_image[ann['image_id']].append(ann)
    packages=[];coverage={};positive_groups=0;empty_frames=0
    for clip in manifest:
        xx=[];yy=[]
        for slot,im in enumerate(clip['frames']):
            cover=np.zeros((128,128),np.int32);single=np.zeros((128,128),np.int16)
            for ann in by_image[im['id']]:
                mask=coco_mask.decode(ann['segmentation']);assert mask.shape==(im['height'],im['width'])
                raster=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST),dtype=bool)
                cover+=raster.astype(np.int32);single[raster]=-1 if ann.get('iscrowd',0)else mapping.get(ann['category_id'],-1)
            labels=np.where(cover==0,0,np.where(cover==1,single,-1)).astype(np.int16);pool=np.flatnonzero(labels.ravel()>0)
            if len(pool):
                selected=np.random.default_rng(p['query_seed_base']+im['id']).choice(pool,256,replace=len(pool)<256)
                xy=np.column_stack([selected%128,selected//128,np.full(256,slot)]).astype(np.int16);y=labels.ravel()[selected].astype(np.int16)
            else:xy=np.empty((0,3),np.int16);y=np.empty(0,np.int16);empty_frames+=1
            xx.append(xy);yy.append(y);coverage[(clip['group'],slot)]={'group':clip['group'],'time':slot,'image_id':im['id'],
                'known_foreground_pixels':int(np.sum(labels>0)),'ignored_pixels':int(np.sum(labels<0)),'query_count':len(y)}
        with np.load(feature_path(clip['group']))as data:package={k:data[k].copy()for k in data.files}
        assert np.array_equal(package['xy'],np.concatenate(xx))and np.array_equal(package['labels'],np.concatenate(yy))
        assert int(package['video_id'])==int(clip['group'])and package['decoded'].shape==(len(package['labels']),384)
        assert package['decoded'].dtype==np.float16 and np.isfinite(package['decoded']).all()
        positive_groups+=int(bool(len(package['labels'])));packages.append(package)
        assert file_hash(feature_path(clip['group']))==s['feature_hashes'][str(feature_path(clip['group']))]
    stored_coverage=json.loads((ROOT/'query_coverage.json').read_text());assert len(stored_coverage)==320
    assert {(r['group'],r['time']):r for r in stored_coverage}==coverage
    assert empty_frames==s['empty_frames']and positive_groups==s['query_bearing_groups']
    assert {c['group']:len(d['labels'])for c,d in zip(manifest,packages)}==s['query_counts']
    assert [c['group']for c,d in zip(manifest,packages)if not len(d['labels'])]==s['empty_groups']
    assert sum(len(d['labels'])for d in packages)==s['total_queries']
    model=rgb.prior.resolution.load_dino();first=next(i for i,c in enumerate(packages)if len(c['labels']))
    recomputed=independent_features(model,[str(RGB/'images'/im['file_name'])for im in manifest[first]['frames']],packages[first]['xy'])
    assert np.array_equal(recomputed,packages[first]['decoded'])
    val=p['compatibility_val_video']
    with np.load(ROOT/'compatibility_features.npz')as data:compat={k:data[k].copy()for k in data.files}
    with np.load(rgb.feature_path('native_gt','val',258))as data:assert np.array_equal(compat['old'],data['decoded'])and np.array_equal(compat['xy'],data['xy'])
    current=independent_features(model,val['frames'],compat['xy']);assert np.array_equal(current,compat['current']);del model
    assert float(np.max(np.abs(compat['old'].astype(np.float32)-current.astype(np.float32))))==s['compatibility']['max_abs_feature_difference']
    assert bool(np.array_equal(compat['old'],current))==s['compatibility']['features_bitwise_equal']
    labels=np.concatenate([c['labels']for c in packages]);video_ids=np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int32)for c in packages]);times=np.concatenate([c['xy'][:,2]for c in packages])
    rebuilt={};confusions={}
    for arm in ARMS:
        rebuilt[arm]=[];offset=p['selection'][arm]['offset'];assert s['results'][arm]['offset']==offset
        for seed in range(3):
            state=torch.load(checkpoint(arm,seed),map_location='cpu',weights_only=True)['model'];value=np.concatenate([linear(c['decoded'],state)for c in packages])
            path=ROOT/arm/'logits'/f'dinov2_{seed}_confirmation.npz'
            with np.load(path)as data:
                assert np.array_equal(data['logits'],value)and np.array_equal(data['labels'],labels)and np.array_equal(data['video_ids'],video_ids)and np.array_equal(data['times'],times)
                assert str(data['checkpoint_sha256'])==p['selection'][arm]['checkpoint_sha256'][str(seed)]
            shifted=value.copy();shifted[:,0]+=offset;pred=shifted.argmax(1);raw=value.argmax(1)
            aggregate,matrix=metric(pred,labels);unshifted,_=metric(raw,labels);per_video={}
            for clip in manifest:
                rows=video_ids==int(clip['group']);per_video[clip['group']]=metric(pred[rows],labels[rows])[0]if np.any(rows)else None
            run={'seed':seed,'raw':unshifted,'calibrated':aggregate,'per_video':per_video};equal(run,s['results'][arm]['runs'][seed]);rebuilt[arm].append(run);confusions[f'{arm}_{seed}']=matrix.tolist()
            a=linear(compat['old'],state);b=linear(current,state);aa=a.copy();bb=b.copy();aa[:,0]+=offset;bb[:,0]+=offset
            entry={'max_abs_logit_difference':float(np.max(np.abs(a-b))),'calibrated_class_changes':int(np.sum(aa.argmax(1)!=bb.argmax(1)))}
            equal(entry,s['compatibility']['head_differences'][f'{arm}_{seed}'])
            print('verified external logits',arm,seed,flush=True)
    def check_contrast(expected):
        left,right=expected['left'],expected['right'];diffs={}
        for clip in manifest:
            group=clip['group']
            diffs[group]=float(np.mean([rebuilt[left][seed]['per_video'][group]['macro_f1_present']-rebuilt[right][seed]['per_video'][group]['macro_f1_present']for seed in range(3)]))if rebuilt[left][0]['per_video'][group]is not None else None
        equal(diffs,expected['per_video_three_seed_mean_differences']);values=np.array([d for d in diffs.values()if d is not None])
        draws=np.random.default_rng(p['bootstrap_seed']).integers(0,len(values),(p['bootstrap_repetitions'],len(values)));means=values[draws].mean(1)
        boot=expected['conditional_video_bootstrap'];assert np.allclose(np.quantile(means,[.025,.975]),boot['percentile_95_interval'],atol=1e-12,rtol=0)
        assert abs(values.mean()-boot['mean_per_video_difference'])<1e-12 and len(values)==boot['video_count']
        assert int((values>0).sum())==boot['positive_videos']and int((values<0).sum())==boot['negative_videos']and int((values==0).sum())==boot['tied_videos']
        pooled=[rebuilt[left][seed]['calibrated']['macro_f1_present']-rebuilt[right][seed]['calibrated']['macro_f1_present']for seed in range(3)]
        equal(pooled,expected['per_seed_pooled_macro_difference']);assert abs(np.mean(pooled)-expected['mean_pooled_macro_difference'])<1e-12
    check_contrast(s['primary_contrast'])
    for c in s['secondary_contrasts_unadjusted'].values():check_contrast(c)
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'independent_confusions.json',confusions)
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'replayed_candidate_groups':40,'query_bearing_groups':positive_groups,
        'replayed_queries':len(labels),'confirmed_support':{str(int(k)):int(v)for k,v in zip(*np.unique(labels,return_counts=True))},'logit_packages_recomputed':12,
        'feature_recomputed_group':manifest[first]['group'],'independent_confusions_sha256':file_hash(ROOT/'independent_confusions.json'),
        'verification':{'all_query_masks_and_sampling_recomputed':True,'empty_frames_and_groups_retained':True,'first_nonempty_feature_array_bitwise_reproduced':True,
        'source_compatibility_features_and_twelve_head_differences_reproduced':True,'all_twelve_logits_labels_and_metrics_identical':True,
        'all_paired_bootstrap_intervals_recomputed':True,'frozen_inputs_offsets_and_models_unchanged':True,'reserve_not_model_evaluated':True}})
    print('DONE independent external semantic audit',len(labels),flush=True)


if __name__=='__main__':main()
