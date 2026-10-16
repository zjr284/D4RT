"""Preregistered four-arm frozen semantic confirmation; reserve remains unscored."""
from collections import defaultdict
import fcntl
import gc
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from semantic import dino_native_rgb_control as rgb
from semantic.external_coco_adapter import ANN,OVIS,label_map,queries
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.semantic_feature_control import infer,save_npz,patch_at_queries
from semantic.background_calibration import predictions
from semantic.fresh_video_confirmation import bootstrap_video_differences
from semantic.model import SemanticHead
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics

ROOT=Path('semantic_artifacts/external_semantic_confirmation')
REVIEW=Path('semantic_artifacts/external_blind_review')
RGB=Path('semantic_artifacts/external_local_rgb')
COHORT=Path('semantic_artifacts/external_rgb_screen/manifest.json')
ARMS=('source_gt','source_gt32','lambda0','native_sam')
SEED=20261051
MAPPING_SUMMARY=Path('semantic_artifacts/external_coco_adapter/summary.json')


def feature_path(group):return ROOT/'features'/f'{group}.npz'


def freeze():
    previous=json.loads((REVIEW/'summary.json').read_text());v=json.loads((REVIEW/'verification.json').read_text())
    assert file_hash(REVIEW/'summary.json')==v['summary_sha256']
    eligible=json.loads((REVIEW/'frozen_eligibility.json').read_text());manifest=json.loads(COHORT.read_text())
    clips=[c for c in manifest['candidate_confirmation']if c['group']in eligible['candidate_confirmation']]
    assert [c['group']for c in clips]==eligible['candidate_confirmation']and len(clips)==40
    selection=json.loads((RGB/'summary.json').read_text())['protocol']['selection']
    selected=max(ARMS,key=lambda a:selection[a]['mean_validation_macro']);assert selected=='source_gt32'
    val=next(c for c in json.loads((rgb.ROOT/'manifest.json').read_text())['val']if c['video_id']==258)
    vendor=rgb.prior.resolution.VENDOR;weights=rgb.prior.resolution.WEIGHTS
    assert file_hash(weights)=='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()=='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=vendor,check=True)
    paths=[REVIEW/'summary.json',REVIEW/'verification.json',REVIEW/'frozen_eligibility.json',REVIEW/'review_decisions.json',COHORT,
        RGB/'summary.json',RGB/'verification.json',RGB/'image_records.json',MAPPING_SUMMARY,ANN,OVIS,weights,rgb.ROOT/'manifest.json',rgb.feature_path('native_gt','val',258),
        Path('semantic/external_semantic_confirmation.py'),Path('semantic/OVIS_EXTERNAL_SEMANTIC_CONFIRMATION_PROTOCOL.md'),
        Path('semantic/external_coco_adapter.py'),Path('semantic/dino_native_rgb_control.py'),Path('semantic/dino_resolution_control.py'),
        Path('semantic/semantic_feature_control.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py')]
    paths.extend(Path(f)for f in val['frames']);paths.extend(sorted(vendor.rglob('*.py')))
    for clip in clips:paths.extend(RGB/'images'/im['file_name']for im in clip['frames'])
    for arm in ARMS:
        for seed in range(3):
            path=checkpoint(arm,seed);assert file_hash(path)==selection[arm]['checkpoint_sha256'][str(seed)];paths.append(path)
    p={'round':51,'dataset':'LV-VIS val oracle conditional mapped foreground','candidate_groups':[c['group']for c in clips],
        'reserve_groups':eligible['reserve'],'selection':{a:selection[a]for a in ARMS},'selected_by_OVIS_validation':selected,
        'primary':['source_gt32','source_gt'],'secondary':[['native_sam','source_gt'],['lambda0','source_gt']],
        'primary_estimand':'three-seed-mean paired video-equal conditional foreground macro F1 difference on query-bearing videos',
        'bootstrap_seed':SEED,'bootstrap_repetitions':2000,'query_seed_base':20261041,'queries_per_nonempty_frame':256,
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0),
        'training_updates':0,'calibration_updates':0,'head_TF32_disabled':True,'compatibility_val_video':val,'input_sha256':{str(path):file_hash(path)for path in paths}}
    for name,data in [('protocol.json',p),('manifest.json',{'confirmation':clips,'reserve_groups':eligible['reserve']}),('frozen_selection.json',{'selected':'source_gt32','selection':p['selection'],'deployment_unchanged':'native_gt'})]:
        path=ROOT/name
        if path.exists():assert json.loads(path.read_text())==data
        else:atomic_json(path,data)
    return clips,p


def logits(features,arm,seed):
    model=SemanticHead(hidden_dim=384,num_classes=26).cuda().eval()
    model.load_state_dict(torch.load(checkpoint(arm,seed),map_location='cpu',weights_only=True)['model'])
    with torch.inference_mode():out=model(torch.from_numpy(features).float().cuda()).cpu().numpy()
    del model;return out


def compatibility(model,p):
    target=ROOT/'compatibility.json'
    if target.exists():return
    clip=p['compatibility_val_video']
    with np.load(rgb.feature_path('native_gt','val',258))as cache:
        old=cache['decoded'].copy();xy=cache['xy'].copy()
    video=rgb.load_native_video(clip).cuda();tokens=rgb.prior.resolution.extract_tokens(model,video,448)
    with torch.inference_mode():new=patch_at_queries(tokens,xy).half().cpu().numpy()
    assert new.shape==old.shape and np.isfinite(new).all();changes={}
    for arm in ARMS:
        offset=p['selection'][arm]['offset']
        for seed in range(3):
            a=logits(old,arm,seed);b=logits(new,arm,seed)
            changes[f'{arm}_{seed}']={'max_abs_logit_difference':float(np.max(np.abs(a-b))),
                'calibrated_class_changes':int(np.sum(predictions(a,offset)!=predictions(b,offset)))}
    save_npz(ROOT/'compatibility_features.npz',old=old,current=new,xy=xy)
    atomic_json(target,{'video_id':258,'query_count':len(xy),'max_abs_feature_difference':float(np.max(np.abs(old.astype(np.float32)-new.astype(np.float32)))),
        'features_bitwise_equal':bool(np.array_equal(old,new)),'head_differences':changes,
        'note':'cached historical features versus current frozen recipe, both heads evaluated in current runtime; no recalibration'})
    del video,tokens;print('compatibility checked',258,flush=True)


def cache_features(clips,p):
    tasks=[c for c in clips if not feature_path(c['group']).exists()]
    if not tasks and (ROOT/'compatibility.json').exists():return
    payload=json.loads(ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(MAPPING_SUMMARY.read_text())['shared_category_mapping'].items()}
    by_image=defaultdict(list)
    for ann in payload['annotations']:by_image[ann['image_id']].append(ann)
    model=rgb.prior.resolution.load_dino();compatibility(model,p);frame_stats=[]
    existing=ROOT/'query_coverage.json'
    if existing.exists():frame_stats=json.loads(existing.read_text())
    for clip in tasks:
        points=[];targets=[];local_stats=[]
        for slot,im in enumerate(clip['frames']):
            labels=label_map(by_image[im['id']],mapping,im['height'],im['width']);xy,y=queries(labels,im['id'],slot)
            points.append(xy);targets.append(y);local_stats.append({'group':clip['group'],'time':slot,'image_id':im['id'],
                'known_foreground_pixels':int(np.sum(labels>0)),'ignored_pixels':int(np.sum(labels<0)),'query_count':len(y)})
        xy=np.concatenate(points).astype(np.int16);y=np.concatenate(targets).astype(np.int16)
        if len(y):
            video=rgb.load_native_video({'frames':[str(RGB/'images'/im['file_name'])for im in clip['frames']]}).cuda()
            tokens=rgb.prior.resolution.extract_tokens(model,video,448)
            with torch.inference_mode():features=patch_at_queries(tokens,xy).half().cpu().numpy()
            assert np.isfinite(features).all();del video,tokens
        else:features=np.empty((0,384),np.float16)
        frame_stats=[r for r in frame_stats if r['group']!=clip['group']]+local_stats;atomic_json(existing,frame_stats)
        save_npz(feature_path(clip['group']),decoded=features,labels=y,xy=xy,video_id=np.int64(int(clip['group'])))
        print('external frozen features',clip['group'],'queries',len(y),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


def contrast(left,right,results,clips,p):
    per_video={}
    for clip in clips:
        group=clip['group'];a=[results[left]['runs'][s]['per_video'][group]for s in range(3)];b=[results[right]['runs'][s]['per_video'][group]for s in range(3)]
        per_video[group]=float(np.mean([x['macro_f1_present']-y['macro_f1_present']for x,y in zip(a,b)]))if a[0]is not None else None
    values=[v for v in per_video.values()if v is not None]
    pooled=[results[left]['runs'][s]['calibrated']['macro_f1_present']-results[right]['runs'][s]['calibrated']['macro_f1_present']for s in range(3)]
    return {'left':left,'right':right,'per_seed_pooled_macro_difference':pooled,'mean_pooled_macro_difference':float(np.mean(pooled)),
        'per_video_three_seed_mean_differences':per_video,'conditional_video_bootstrap':bootstrap_video_differences(values,seed=p['bootstrap_seed'],repetitions=p['bootstrap_repetitions'])if values else None}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'summary.json').exists():raise RuntimeError('Completed external confirmation exists')
        torch.backends.cuda.matmul.allow_tf32=False
        clips,p=freeze();cache_features(clips,p);features={str(feature_path(c['group'])):file_hash(feature_path(c['group']))for c in clips}
        fp=ROOT/'feature_hashes.json'
        if fp.exists():assert json.loads(fp.read_text())==features
        else:atomic_json(fp,features)
        packages=[]
        for clip in clips:
            with np.load(feature_path(clip['group']))as data:packages.append({k:data[k].copy()for k in data.files})
        if sum(len(c['labels'])for c in packages)==0:raise ValueError('Frozen confirmation has no evaluable queries; do not replace videos')
        results={}
        for arm in ARMS:
            runs=[];offset=p['selection'][arm]['offset']
            for seed in range(3):
                out=infer('dinov2',seed,'confirmation',packages,checkpoint_path=checkpoint(arm,seed),output_root=ROOT/arm)
                pred=predictions(out['logits'],offset);raw=predictions(out['logits'],0);per_video={}
                for clip in clips:
                    mask=out['video_ids']==int(clip['group'])
                    per_video[clip['group']]=classification_metrics(pred[mask],out['labels'][mask],26)if np.any(mask)else None
                runs.append({'seed':seed,'raw':classification_metrics(raw,out['labels'],26),'calibrated':classification_metrics(pred,out['labels'],26),'per_video':per_video})
                print('external frozen inference',arm,seed,runs[-1]['calibrated']['macro_f1_present'],flush=True)
            results[arm]={'offset':offset,'runs':runs}
        primary=contrast(*p['primary'],results,clips,p);secondary={a+'_minus_'+b:contrast(a,b,results,clips,p)for a,b in p['secondary']}
        coverage=json.loads((ROOT/'query_coverage.json').read_text());counts={c['group']:len(d['labels'])for c,d in zip(clips,packages)}
        for path,h in p['input_sha256'].items():assert file_hash(path)==h
        for path,h in features.items():assert file_hash(path)==h
        assert not set(p['reserve_groups'])&{path.stem for path in (ROOT/'features').glob('*.npz')}
        atomic_json(ROOT/'summary.json',{'protocol':p,'results':results,'primary_contrast':primary,'secondary_contrasts_unadjusted':secondary,
            'query_counts':counts,'empty_groups':[g for g,n in counts.items()if not n],'empty_frames':sum(not r['query_count']for r in coverage),
            'total_candidate_groups':len(clips),'query_bearing_groups':sum(bool(n)for n in counts.values()),'total_queries':sum(counts.values()),
            'query_coverage_sha256':file_hash(ROOT/'query_coverage.json'),'feature_hashes':features,'compatibility':json.loads((ROOT/'compatibility.json').read_text()),
            'reserve_model_evaluated':False,'training_or_calibration_updates':0,'deployment_unchanged':'native_gt'})
        print('DONE external frozen confirmation',primary['conditional_video_bootstrap'],flush=True)


if __name__=='__main__':main()
