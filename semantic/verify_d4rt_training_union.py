"""Independent frozen-cohort identity, RLE query and resource-support verification."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha
from semantic.verify_d4rt_training_fit_audit import near
ROOT=Path('semantic_artifacts/d4rt_training_union');HISTORY=Path('semantic_artifacts/video_coverage_intervention')


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text())
    assert p['round']==110 and s['protocol_sha256']==sha(ROOT/'protocol.json')and s['manifest_sha256']==sha(ROOT/'manifest.json')
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    old=json.loads((HISTORY/'manifest.json').read_text());original=json.loads(Path('semantic_artifacts/manifest.json').read_text())
    random={int(c['video_id'])for c in old['training_sets']['random80']};cover={int(c['video_id'])for c in old['training_sets']['coverage80']}
    base={int(c['video_id'])for c in original['train']};extra=random|cover;excluded={int(c['video_id'])for split in ('val','test')for c in original[split]}
    assert len(random)==len(cover)==80 and len(random&cover)==60 and len(extra)==100 and not extra&(base|excluded)
    assert p['baseline_video_ids']==sorted(base)and p['additional_video_ids']==sorted(extra)and p['union_video_ids']==sorted(base|extra)
    assert len(base)==60 and p['excluded_current_val_test']==sorted(excluded)
    assert {int(c['video_id'])for c in manifest['baseline_train']}==base and {int(c['video_id'])for c in manifest['additional_train']}==extra
    assert len(manifest['train'])==len(s['video_resources'])==160 and {int(c['video_id'])for c in manifest['train']}==base|extra
    assert manifest['val']==original['val']and manifest['test']==original['test']
    originals={int(c['video_id']):c for c in original['train']}
    for group in old['training_sets'].values():
        for c in group:
            vid=int(c['video_id'])
            if vid in originals:assert c==originals[vid]
            originals[vid]=c
    ann=json.loads(Path('semantic_artifacts/ovis/annotations_train.json').read_text());by=defaultdict(list)
    for a in ann['annotations']:by[int(a['video_id'])].append(a)
    names=('baseline60','additional100','union160');support={name:{'videos':set(),'queries':0,'per_class':{str(c):{'queries':0,'unique_video_frame_xy':set(),'videos':set()}for c in range(26)}}for name in names}
    checked=[];paired=True
    for clip in manifest['train']:
        vid=int(clip['video_id']);assert clip==originals[vid]and len(clip['frames'])==8
        record=next(r for r in s['video_resources']if r['video_id']==vid);assert sha(record['queries_path'])==record['queries_sha256']
        with np.load(record['teacher_path'])as a:maps=a['labels'].copy()
        counts=np.zeros((128,128),np.int64);cats=np.zeros_like(counts)
        for a in by[vid]:
            if clip['start']>=len(a['segmentations']):continue
            raw=a['segmentations'][clip['start']]
            if raw is None:continue
            m=mask_api.decode(raw)
            if m.ndim==3:m=m.any(2)
            hit=np.asarray(Image.fromarray(m.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST))>0
            counts+=hit;cats[hit]=a['category_id']
        truth=np.where(counts==0,0,np.where(counts==1,cats,-1));assert np.array_equal(truth,maps[0])
        with np.load(record['queries_path'])as a:xy=a['xy'].copy();y=a['labels'].copy()
        assert np.array_equal(y,maps[xy[:,2],xy[:,1],xy[:,0]])and np.all((y>=0)&(y<26))
        rng=np.random.Generator(np.random.PCG64(42+vid));expected=[]
        for t in range(8 if vid in extra else 1):
            for pool in (np.argwhere(maps[t]>0),np.argwhere(maps[t]==0)):
                if len(pool):
                    chosen=pool[rng.choice(len(pool),size=128,replace=len(pool)<128)]
                    expected.extend((int(x),int(v),t)for v,x in chosen)
        if vid in extra:assert np.array_equal(np.asarray(expected),xy)
        else:
            with np.load(Path('semantic_artifacts/source_query_factorial/features/train')/f'{vid}.npz')as a:assert np.array_equal(a['xy'],xy)and np.array_equal(a['labels'],y)
            assert np.array_equal(np.asarray(expected),xy[xy[:,2]==0])
        source=xy[:,2]==0;assert np.count_nonzero(source)==256 and np.count_nonzero(y[source]>0)==128
        assert record['source_FG_classes']==sorted(set(int(c)for c in y[source]if c>0))
        fg=(xy[:,2]>0)&(y>0);bgcounts={str(t):int(np.count_nonzero(maps[t]==0))for t in range(1,8)}
        matched=sum(bgcounts[str(int(t))]>0 for t in xy[fg,2]);valid=bool(np.any(fg)and matched==int(fg.sum()))
        assert record['later_BG_pixels_by_frame']==bgcounts and record['later_BG_pixels']==sum(bgcounts.values())
        assert record['later_SAM_FG_queries']==int(fg.sum())and record['later_SAM_FG_unique_positions']==len(set(map(tuple,xy[fg])))
        assert record['later_SAM_FG_queries_with_BG_frame']==matched and record['all_sampled_later_FG_have_same_frame_BG']==valid
        paired=paired and valid;group='baseline60'if vid in base else'additional100';assert record['group']==group
        for name in (group,'union160'):
            support[name]['videos'].add(vid);support[name]['queries']+=int(source.sum())
            for point,label in zip(xy[source],y[source]):
                c=str(int(label));item=support[name]['per_class'][c];item['queries']+=1
                item['unique_video_frame_xy'].add((vid,int(point[2]),int(point[0]),int(point[1])));item['videos'].add(vid)
        checked.append(vid)
    for group in names:
        support[group]['videos']=len(support[group]['videos'])
        for item in support[group]['per_class'].values():item['videos']=len(item['videos']);item['unique_video_frame_xy']=len(item['unique_video_frame_xy'])
    near(support,s['source_GT_support']);assert paired==s['all160_later_auxiliary_pools_same_frame_pairable']
    assert support['union160']['queries']==40960 and support['baseline60']['queries']==15360
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    atomic_json(ROOT/'verification.json',{'round':110,'summary_sha256':sha(ROOT/'summary.json'),'verifier_sha256':sha('semantic/verify_d4rt_training_union.py'),
        'historical_random_coverage_union100_and_current60_val_test_disjoint_exact':True,
        'all160_source_GT_maps_independently_rebuilt_from_RLE':sorted(checked),'all160_fixed_GT_queries_labels_and_auxiliary_pools_verified':True,
        'all3_cohort_per_class_query_unique_position_video_support_verified':True,'all_original_sources_and_teachers_unchanged':True,
        'no_new_teacher_or_model_or_scoring':True})
    print('DONE independent training-union preparation verification',flush=True)
if __name__=='__main__':main()
