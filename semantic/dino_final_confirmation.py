"""Once-only frozen evaluation of the previously reserved final40 videos."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import numpy as np
import torch
from semantic import dino_native_rgb_control as prior
from semantic.semantic_feature_control import save_npz,patch_at_queries,read_features,infer,aggregate_hash
from semantic.ytvis import load_dataset,load_video,sample_queries
from semantic.background_calibration import evaluate,predictions
from semantic.fresh_video_confirmation import bootstrap_video_differences
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics

ROOT=Path('semantic_artifacts/dino_final_confirmation')
ARMS=prior.ARMS
PROTOCOL={'version':1,'round':24,'seeds':[0,1,2],'arms':list(ARMS),'training_updates':0,'calibration_updates':0,
    'confirmation':'exact round22 final_reserve40, no reselection','selected_arm':'native_gt',
    'primary_comparison':'validation-selected native_gt minus previously-selected old_sam, complete schemes',
    'bootstrap_seed':20261024,'bootstrap_repetitions':2000,'source_metadata_previously_inspected':True,
    'pretraining_ovis_overlap':'unknown','shared_validation_repeatedly_used':True,'no_selection_from_final_scores':True}


def checkpoint(arm,seed):return prior.checkpoint(arm,seed)


def feature_path(arm,vid):
    family='native'if arm.startswith('native_')else'old'
    return ROOT/'features'/family/f'{vid}.npz'


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    original=json.loads((prior.ROOT/'summary.json').read_text());verified=json.loads((prior.ROOT/'verification.json').read_text())
    selected=json.loads((prior.ROOT/'frozen_selection.json').read_text())
    assert file_hash(prior.ROOT/'summary.json')==verified['summary_sha256']and file_hash(prior.ROOT/'frozen_selection.json')==original['frozen_selection_sha256']
    assert original['validation_selected_arm']==selected['validation_selected_arm']=='native_gt'
    old=original['manifest'];excluded=set(old['excluded_video_ids'])|{c['video_id']for c in old['confirmation']};assert len(excluded)==540
    clips=old['final_reserve'];assert len(clips)==40 and len({c['video_id']for c in clips})==40 and not excluded&{c['video_id']for c in clips}
    scores={a:float(np.mean([r['calibrated_validation']['macro_f1_present']for r in original['results'][a]['runs']]))for a in ARMS}
    assert max(ARMS,key=lambda a:scores[a])=='native_gt'
    manifest={'confirmation':clips,'excluded_video_ids':sorted(excluded),'reservation_manifest_sha256':file_hash(prior.ROOT/'manifest.json'),
              'reservation_seed':20261022,'previous_model_unevaluated':67,'source_category_metadata_previously_inspected':True}
    path=ROOT/'manifest.json'
    if path.exists()and json.loads(path.read_text())!=manifest:raise ValueError('Final reserve changed')
    atomic_json(path,manifest)
    paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'frozen_selection.json',prior.ROOT/'manifest.json',path,
           Path('semantic/dino_final_confirmation.py'),Path('semantic/OVIS_DINO_FINAL_CONFIRMATION_PROTOCOL.md'),prior.prior.resolution.WEIGHTS,
           Path('semantic_artifacts/ovis/annotations_train.json')]
    paths+=[checkpoint(a,s)for a in ARMS for s in range(3)];paths+=[Path(p)for c in clips for p in c['frames']]
    for a in ARMS:
        for s in range(3):assert file_hash(checkpoint(a,s))==verified['checkpoint_sha256'][f'{a}_{s}']
    protocol=dict(PROTOCOL,offsets=selected['offsets'],checkpoint_sha256=selected['checkpoint_sha256'],input_sha256={str(p):file_hash(p)for p in paths},
                  inherited_protocol=original['protocol'])
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=protocol:raise ValueError('Final frozen inputs changed')
    atomic_json(path,protocol);return manifest,protocol


def cache(ds,manifest):
    model=prior.prior.resolution.load_dino()
    for clip in manifest['confirmation']:
        maps=np.stack([ds.gt_label_map(clip,t)for t in range(8)]);q,y,xy=sample_queries(maps,per_frame=256,seed=42+clip['video_id'])
        meta={'labels':y.numpy().astype(np.int16),'xy':xy.numpy().astype(np.int16),'t_tgt':q['t_tgt'][0].numpy(),'t_cam':q['t_cam'][0].numpy(),'video_id':np.int64(clip['video_id'])}
        for family in ('old','native'):
            path=feature_path(family+'_sam',clip['video_id'])
            if path.exists():continue
            video=(load_video(clip)if family=='old'else prior.load_native_video(clip)).cuda()
            tokens=prior.prior.resolution.extract_tokens(model,video,448)
            with torch.inference_mode():features=patch_at_queries(tokens,meta['xy']).half().cpu().numpy()
            assert np.isfinite(features).all();save_npz(path,**meta,decoded=features)
            print('cached final',family,clip['video_id'],flush=True);del video,tokens
    del model;gc.collect();torch.cuda.empty_cache()


def read(arm,manifest):
    return read_features('dinov2','confirmation',manifest,path_resolver=lambda _,s,v:feature_path(arm,v))


def comparison(terms,results,manifest):
    entry={}
    for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_accuracy'):
        d=[sum(sign*results[a]['runs'][s]['calibrated_confirmation'][metric]for a,sign in terms)for s in range(3)]
        entry[metric]={'paired_differences':d,'mean':float(np.mean(d)),'std':float(np.std(d,ddof=1))}
    d={str(c['video_id']):float(np.mean([sum(sign*results[a]['runs'][s]['per_video'][str(c['video_id'])]['macro_f1_present']for a,sign in terms)for s in range(3)]))for c in manifest['confirmation']}
    entry['per_video_differences']=d;entry['video_bootstrap']=bootstrap_video_differences(list(d.values()),seed=PROTOCOL['bootstrap_seed'],repetitions=PROTOCOL['bootstrap_repetitions'])
    return entry


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Final40 already evaluated; audit rather than rerun')
    manifest,protocol=freeze();ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    cache(ds,manifest);results={}
    for arm in ARMS:
        clips=read(arm,manifest);runs=[]
        for s in range(3):
            p=infer('dinov2',s,'confirmation',clips,checkpoint_path=checkpoint(arm,s),output_root=ROOT/arm)
            pred=predictions(p['logits'],protocol['offsets'][arm]);metric=evaluate(p,protocol['offsets'][arm])
            per_video={str(c['video_id']):classification_metrics(pred[p['video_ids']==c['video_id']],p['labels'][p['video_ids']==c['video_id']],26)for c in manifest['confirmation']}
            runs.append({'seed':s,'raw_confirmation':evaluate(p,0),'calibrated_confirmation':metric,'per_video':per_video})
            print('final confirmed',arm,s,metric['macro_f1_present'],flush=True)
        results[arm]={'offset':protocol['offsets'][arm],'runs':runs};del clips
    terms={'selected_vs_previous':[('native_gt',1),('old_sam',-1)],'RGB_GT':[('native_gt',1),('old_gt',-1)],'RGB_SAM':[('native_sam',1),('old_sam',-1)],
           'SAM_native':[('native_sam',1),('native_gt',-1)],'SAM_old':[('old_sam',1),('old_gt',-1)],'interaction':[('native_sam',1),('native_gt',-1),('old_sam',-1),('old_gt',1)]}
    contrasts={n:comparison(t,results,manifest)for n,t in terms.items()}
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'manifest':manifest,'results':results,'contrasts':contrasts,'validation_selected_arm_unchanged':'native_gt',
        'feature_sha256':{f:aggregate_hash([feature_path(f+'_sam',c['video_id'])for c in manifest['confirmation']])for f in ('old','native')},
        'verification':{'all_heads_offsets_and_validation_choice_unchanged':True,'reserved40_exactly_used_once':True,'no_training_or_calibration':True,
                        'no_selection_from_final_scores':True,'final_excludes_previous540':True}})
    print('DONE FINAL40',contrasts['selected_vs_previous']['macro_f1_present'],contrasts['selected_vs_previous']['video_bootstrap'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
