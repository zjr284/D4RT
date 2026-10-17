"""Independent RLE/class sampling, feature, 12-logit and panel-statistic replay."""
from collections import defaultdict
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import torch
from semantic.external_nonperson_confirmation import ROOT,PANEL,PRIOR,ANN,OVIS,ARMS,feature_path
from semantic import dino_native_rgb_control as rgb
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.verify_external_semantic_confirmation import metric,equal,linear,independent_features
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    old=json.loads((PRIOR/'protocol.json').read_text());assert all(p[k]==old[k]for k in ('selection','torch_version','cuda_version','device'))
    frozen=json.loads((ROOT/'frozen_selection.json').read_text());assert frozen['selected']==p['selected_by_OVIS_validation']=='source_gt32'and frozen['selection']==p['selection']
    clips=json.loads((ROOT/'manifest.json').read_text())['confirmation'];eligible=json.loads((PANEL/'frozen_eligibility.json').read_text())
    assert [c['group']for c in clips]==p['candidate_groups']==eligible['accepted_groups']
    assert not set(p['candidate_groups'])&set(p['excluded_original80'])
    assert {path.stem for path in (ROOT/'features').glob('*.npz')}==set(p['candidate_groups'])
    payload=json.loads(ANN.read_bytes());cats=json.loads(OVIS.read_bytes())['categories'];norm=lambda n:' '.join(n.lower().replace('_',' ').split())
    names={norm(c['name']):c['id']for c in cats};mapping={c['id']:names[norm(c['name'])]for c in payload['categories']if norm(c['name'])in names}
    by_image=defaultdict(list)
    for ann in payload['annotations']:by_image[ann['image_id']].append(ann)
    packages=[];coverage={};empty_frames=0
    for clip in clips:
        all_xy=[];all_y=[]
        for slot,im in enumerate(clip['frames']):
            cover=np.zeros((128,128),np.int32);single=np.zeros((128,128),np.int16)
            for ann in by_image[im['id']]:
                mask=coco_mask.decode(ann['segmentation']);assert mask.shape==(im['height'],im['width'])
                raster=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST),dtype=bool)
                cover+=raster.astype(np.int32);single[raster]=-1 if ann.get('iscrowd',0)else mapping.get(ann['category_id'],-1)
            labels=np.where(cover==0,0,np.where(cover==1,single,-1)).astype(np.int16);xx=[];yy=[]
            for cid in range(2,26):
                pool=np.flatnonzero(labels.ravel()==cid)
                if not len(pool):continue
                rng=np.random.default_rng(p['query_seed_base']+im['id']*32+cid);chosen=rng.choice(pool,64,replace=len(pool)<64)
                xx.append(np.column_stack((chosen%128,chosen//128,np.full(64,slot))).astype(np.int16));yy.append(np.full(64,cid,np.int16))
            xy=np.concatenate(xx)if xx else np.empty((0,3),np.int16);y=np.concatenate(yy)if yy else np.empty(0,np.int16)
            all_xy.append(xy);all_y.append(y);empty_frames+=int(not len(y))
            coverage[(clip['group'],slot)]={'group':clip['group'],'time':slot,'image_id':im['id'],'nonperson_pixels':int(np.sum(labels>1)),
                'ignored_pixels':int(np.sum(labels<0)),'query_count':len(y),'query_class_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(y,return_counts=True))}}
        with np.load(feature_path(clip['group']))as data:package={k:data[k].copy()for k in data.files}
        assert np.array_equal(package['xy'],np.concatenate(all_xy))and np.array_equal(package['labels'],np.concatenate(all_y))
        assert int(package['video_id'])==int(clip['group'])and package['decoded'].shape==(len(package['labels']),384)and package['decoded'].dtype==np.float16
        assert np.isfinite(package['decoded']).all()and np.all(package['labels']>1)
        assert file_hash(feature_path(clip['group']))==s['feature_hashes'][str(feature_path(clip['group']))]
        packages.append(package)
    stored=json.loads((ROOT/'query_coverage.json').read_text());assert len(stored)==8*len(clips)
    assert {(r['group'],r['time']):r for r in stored}==coverage and empty_frames==s['empty_frames']
    assert {c['group']:len(d['labels'])for c,d in zip(clips,packages)}==s['query_counts']
    assert [c['group']for c,d in zip(clips,packages)if not len(d['labels'])]==s['empty_groups']
    model=rgb.prior.resolution.load_dino();first=next(i for i,c in enumerate(packages)if len(c['labels']))
    feature=independent_features(model,[str(PANEL/'images'/im['file_name'])for im in clips[first]['frames']],packages[first]['xy'])
    assert np.array_equal(feature,packages[first]['decoded']);del model
    labels=np.concatenate([c['labels']for c in packages]);times=np.concatenate([c['xy'][:,2]for c in packages]);ids=np.concatenate([np.full(len(c['labels']),int(c['video_id']),np.int32)for c in packages])
    support={str(int(c)):int(n)for c,n in zip(*np.unique(labels,return_counts=True))};assert support==s['actual_support']and len(labels)==s['total_queries']
    assert sorted(set(p['metadata_selected_classes'])-set(map(int,np.unique(labels))))==s['metadata_classes_without_queries']
    rebuilt={};confusions={}
    for arm in ARMS:
        rebuilt[arm]=[];offset=p['selection'][arm]['offset'];assert s['results'][arm]['offset']==offset
        for seed in range(3):
            state=torch.load(checkpoint(arm,seed),map_location='cpu',weights_only=True)['model'];values=np.concatenate([linear(c['decoded'],state)for c in packages])
            with np.load(ROOT/arm/'logits'/f'dinov2_{seed}_confirmation.npz')as data:
                assert np.array_equal(data['logits'],values)and np.array_equal(data['labels'],labels)and np.array_equal(data['video_ids'],ids)and np.array_equal(data['times'],times)
                assert str(data['checkpoint_sha256'])==p['selection'][arm]['checkpoint_sha256'][str(seed)]
            shifted=values.copy();shifted[:,0]+=offset;pred=shifted.argmax(1);calibrated,matrix=metric(pred,labels);raw,_=metric(values.argmax(1),labels)
            per_video={}
            for c in clips:
                rows=ids==int(c['group']);per_video[c['group']]=metric(pred[rows],labels[rows])[0]if np.any(rows)else None
            result={'seed':seed,'raw':raw,'calibrated':calibrated,'per_video':per_video};equal(result,s['results'][arm]['runs'][seed]);rebuilt[arm].append(result);confusions[f'{arm}_{seed}']=matrix.tolist()
            print('verified nonPerson logits',arm,seed,flush=True)
    def verify_contrast(expected):
        left,right=expected['left'],expected['right'];values={}
        for c in clips:
            group=c['group'];values[group]=float(np.mean([rebuilt[left][seed]['per_video'][group]['macro_f1_present']-rebuilt[right][seed]['per_video'][group]['macro_f1_present']for seed in range(3)]))if rebuilt[left][0]['per_video'][group]is not None else None
        equal(values,expected['per_video_three_seed_mean_differences']);d=np.array([v for v in values.values()if v is not None])
        draws=np.random.default_rng(p['bootstrap_seed']).integers(0,len(d),(p['bootstrap_repetitions'],len(d)));means=d[draws].mean(1);b=expected['conditional_video_bootstrap']
        assert abs(d.mean()-b['mean_per_video_difference'])<1e-12 and len(d)==b['video_count']
        assert np.allclose(np.quantile(means,[.025,.975]),b['percentile_95_interval'],atol=1e-12,rtol=0)
        assert int((d>0).sum())==b['positive_videos']and int((d<0).sum())==b['negative_videos']and int((d==0).sum())==b['tied_videos']
        pooled=[rebuilt[left][seed]['calibrated']['macro_f1_present']-rebuilt[right][seed]['calibrated']['macro_f1_present']for seed in range(3)]
        equal(pooled,expected['per_seed_pooled_macro_difference']);assert abs(np.mean(pooled)-expected['mean_pooled_macro_difference'])<1e-12
    verify_contrast(s['primary_contrast'])
    for c in s['secondary_contrasts_unadjusted'].values():verify_contrast(c)
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert s['training_or_calibration_updates']==0 and not s['original80_model_evaluated_again']and s['deployment_unchanged']=='native_gt'
    atomic_json(ROOT/'independent_confusions.json',confusions)
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'replayed_groups':len(clips),'replayed_queries':len(labels),'confirmed_nonperson_support':support,
        'logit_packages_recomputed':12,'first_feature_group_recomputed':clips[first]['group'],'independent_confusions_sha256':file_hash(ROOT/'independent_confusions.json'),
        'verification':{'all_GT_maps_class_balanced_queries_and_empty_frames_recomputed':True,'first_feature_array_bitwise_reproduced':True,
            'all_twelve_logits_and_metrics_identical':True,'all_three_paired_bootstraps_recomputed':True,'models_offsets_runtime_and_all_inputs_unchanged':True,
            'original80_not_reevaluated_or_replaced':True}})
    print('DONE independent nonPerson panel audit',len(labels),support,flush=True)

if __name__=='__main__':main()
