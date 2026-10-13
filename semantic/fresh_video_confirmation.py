"""Frozen-model confirmation on preregistered previously unused OVIS videos."""
from __future__ import annotations
import gc
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from semantic.ytvis import load_dataset,load_video,sample_queries,CLIP_FRAMES
from semantic.model import extract_query_features,SemanticHead
from semantic.oracle_probe import file_hash
from semantic.background_calibration import evaluate,predictions
from semantic.run import atomic_json,classification_metrics
from tools.load_opend4rt import load_model,DEFAULT_VARIANT

ROOT=Path('semantic_artifacts/fresh_video_confirmation')
OFFSETS={'source_gt':-1.25,'positive_only':3.5,'positive_background':-2.0}
PROTOCOL={'version':1,'videos':40,'selection_seed':20261013,'query_seed':42,'seeds':[0,1,2],
          'frames':8,'resolution':128,'queries_per_frame':256,'head':'linear','budget':2000,
          'offsets':OFFSETS,'train_updates':0,'recalibration':False,
          'selection':'random shuffle of eligible videos after excluding original manifest and existing pilot/cache videos',
          'bootstrap':'2000 video resamples; mean per-video macro-F1 difference averaged over model seeds; 95% percentile interval',
          'bootstrap_seed':20261013}


def select_clips(dataset,excluded,count=40,seed=20261013):
    eligible=[]
    for vid,video in sorted(dataset.videos.items()):
        if vid in excluded:continue
        names=video['file_names']
        if len(names)<CLIP_FRAMES:continue
        start=next((i for i in range(len(names)-CLIP_FRAMES+1) if any(
            i<len(a['segmentations']) and a['segmentations'][i] is not None for a in dataset.annotations.get(vid,[]))),None)
        if start is None:continue
        try:frames=[str(dataset.frame_path(n)) for n in names[start:start+CLIP_FRAMES]]
        except FileNotFoundError:continue
        eligible.append({'video_id':vid,'start':start,'frames':frames})
    random.Random(seed).shuffle(eligible)
    if len(eligible)<count:raise ValueError('Insufficient unused eligible videos')
    return eligible[:count],len(eligible)


def checkpoint(arm,seed):
    if arm=='source_gt':return Path('semantic_artifacts/source_gt_control/runs')/f'linear_2000_seed{seed}.pt'
    return Path('semantic_artifacts/weaklabel_budget_probe')/arm/'runs'/f'linear_2000_seed{seed}.pt'


def cache_features(dataset,manifest):
    folder=ROOT/'features';folder.mkdir(exist_ok=True)
    missing=[c for c in manifest['clips'] if not (folder/f"{c['video_id']}.npz").exists()]
    if not missing:return
    model=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval()
    for clip in missing:
        vid=clip['video_id'];labels=np.stack([dataset.gt_label_map(clip,t) for t in range(8)])
        query,targets,xy=sample_queries(labels,per_frame=256,seed=42+vid)
        video=load_video(clip)[None].to('cuda',dtype=torch.bfloat16);chunks=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            memory=model.encode_video(video)
            for start in range(0,len(targets),64):
                small={k:v[:,start:start+64].to('cuda') for k,v in query.items()}
                chunks.append(extract_query_features(model,video,small,memory)['decoded'][0].to(torch.float16).cpu().numpy())
        destination=folder/f'{vid}.npz';temporary=destination.with_suffix('.partial.npz')
        np.savez_compressed(temporary,decoded=np.concatenate(chunks),labels=targets.numpy().astype(np.int16),xy=xy.numpy().astype(np.int16),
                            t_tgt=query['t_tgt'][0].numpy(),t_cam=query['t_cam'][0].numpy(),video_id=np.int64(vid));temporary.replace(destination)
        print('cached fresh video',vid,len(targets),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


def infer(arm,seed,clips):
    path=ROOT/'logits'/f'{arm}_seed{seed}.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():
        with np.load(path) as p:return {k:p[k].copy() for k in p.files}
    model=SemanticHead(num_classes=26).to('cuda').eval();model.load_state_dict(torch.load(checkpoint(arm,seed),map_location='cpu',weights_only=True)['model'])
    logits=[];labels=[];times=[];vids=[]
    with torch.inference_mode():
        for c in clips:
            logits.append(model(torch.from_numpy(c['decoded']).float().to('cuda')).cpu().numpy())
            labels.append(c['labels']);times.append(c['xy'][:,2]);vids.append(np.full(len(c['labels']),int(c['video_id']),dtype=np.int32))
    data={'logits':np.concatenate(logits),'labels':np.concatenate(labels),'times':np.concatenate(times),'video_ids':np.concatenate(vids)}
    temp=path.with_suffix('.partial.npz');np.savez_compressed(temp,**data);temp.replace(path);return data


def bootstrap_video_differences(values,seed=20261013,repetitions=2000):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Finite nonempty video differences required')
    rng=np.random.default_rng(seed);means=values[rng.integers(0,len(values),(repetitions,len(values)))].mean(axis=1)
    return {'mean_per_video_difference':float(values.mean()),'positive_videos':int((values>0).sum()),
            'negative_videos':int((values<0).sum()),'tied_videos':int((values==0).sum()),
            'video_count':len(values),'percentile_95_interval':np.quantile(means,[.025,.975]).tolist(),
            'note':'per-video present-class macro F1 estimand; not pooled-query macro F1 confidence interval'}


def class_composition(results,prior):
    old={int(c) for c in prior['comparisons']['source_gt']['runs'][0]['after_test']['support'] if c!='0'}
    new={int(c) for c in results['source_gt']['runs'][0]['calibrated']['support'] if c!='0'}
    common=sorted(old&new);additional=sorted(new-old)
    analysis={'old_foreground_classes':sorted(old),'fresh_foreground_classes':sorted(new),
              'common_foreground_classes':common,'fresh_additional_foreground_classes':additional,'groups':{}}
    for arm,v in results.items():
        values={c:float(np.mean([r['calibrated']['per_class_f1'][str(c)] for r in v['runs']])) for c in new}
        analysis['groups'][arm]={'mean_f1_common_classes':float(np.mean([values[c] for c in common])) if common else None,
              'mean_f1_additional_classes':float(np.mean([values[c] for c in additional])) if additional else None,
              'per_class_mean_f1':{str(c):values[c] for c in sorted(new)}}
    return analysis


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    old_path=Path('semantic_artifacts/manifest.json');old=json.loads(old_path.read_text())
    excluded={int(c['video_id']) for split in ('train','val','test') for c in old[split]}
    # Also exclude any pilot, teacher or feature video already processed locally.
    for folder in Path('semantic_artifacts').glob('*'):
        if folder.is_dir() and folder!=ROOT and any(word in folder.name for word in ('features','teacher')):
            for p in folder.rglob('*.npz'):
                if p.stem.isdigit():excluded.add(int(p.stem))
    dataset=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    clips,eligible_count=select_clips(dataset,excluded)
    manifest={'seed':20261013,'excluded_video_ids':sorted(excluded),'eligible_unused_video_count':eligible_count,'clips':clips}
    path=ROOT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Frozen fresh manifest differs; do not reselect')
    atomic_json(path,manifest)
    prior=Path('semantic_artifacts/source_gt_control/summary.json');prior_result=json.loads(prior.read_text())
    for arm,offset in OFFSETS.items():
        if prior_result['comparisons'][arm]['selected_offset']!=offset:raise ValueError('Offset does not match frozen prior result')
    hashes={f'{arm}_{seed}':file_hash(checkpoint(arm,seed)) for arm in OFFSETS for seed in PROTOCOL['seeds']}
    protocol=dict(PROTOCOL,manifest_sha256=file_hash(path),original_manifest_sha256=file_hash(old_path),
                  annotation_sha256=file_hash('semantic_artifacts/ovis/annotations_train.json'),
                  prior_summary_sha256=file_hash(prior),checkpoint_sha256=hashes)
    path=ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Frozen protocol/checkpoints differ')
    atomic_json(path,protocol) # Selection and models are fixed before any fresh inference.
    if {c['video_id'] for c in clips}&excluded:raise ValueError('Fresh videos overlap prior experiments')
    cache_features(dataset,manifest)
    cached=[]
    for c in clips:
        with np.load(ROOT/'features'/f"{c['video_id']}.npz") as p:cached.append({k:p[k].copy() for k in ('decoded','labels','xy','video_id')})
    if len(cached)!=40 or not all(np.isfinite(c['decoded']).all() for c in cached):raise ValueError('Incomplete or invalid features')
    results={}
    for arm,offset in OFFSETS.items():
        runs=[]
        for seed in PROTOCOL['seeds']:
            package=infer(arm,seed,cached);before=evaluate(package,0);after=evaluate(package,offset)
            pred=predictions(package['logits'],offset);per_video={}
            for c in clips:
                mask=package['video_ids']==c['video_id']
                per_video[str(c['video_id'])]=classification_metrics(pred[mask],package['labels'][mask],26)
            runs.append({'seed':seed,'uncalibrated':before,'calibrated':after,'per_video':per_video})
            print(arm,seed,'fresh calibrated macroF1',after['macro_f1_present'],flush=True)
        results[arm]={'offset':offset,'runs':runs}
    contrasts={}
    for arm,control in [('positive_only','source_gt'),('positive_background','source_gt'),('positive_background','positive_only')]:
        entry={}
        for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_accuracy'):
            d=[results[arm]['runs'][i]['calibrated'][metric]-results[control]['runs'][i]['calibrated'][metric] for i in range(3)]
            entry[metric]={'paired_seed_differences':d,'mean':float(np.mean(d)),'std':float(np.std(d,ddof=1))}
        differences={str(c['video_id']):float(np.mean([results[arm]['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present']-
                    results[control]['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present'] for i in range(3)])) for c in clips}
        entry['per_video_macro_f1_differences']=differences;entry['video_bootstrap']=bootstrap_video_differences(list(differences.values()))
        contrasts[arm+'_minus_'+control]=entry
    for arm in OFFSETS:
        for seed in PROTOCOL['seeds']:
            if file_hash(checkpoint(arm,seed))!=hashes[f'{arm}_{seed}']:raise ValueError('Checkpoint modified')
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'manifest':manifest,'results':results,'contrasts':contrasts,
                'class_composition_diagnostic':class_composition(results,prior_result),
                'verification':{'40_videos_disjoint_from_all_prior_ids':True,'all_nine_model_hashes_unchanged':True,
                                'offsets_identical_to_prior_validation_selection':True,'no_training_or_recalibration':True}})
    print('DONE fresh confirmation',flush=True)

if __name__=='__main__':main()
