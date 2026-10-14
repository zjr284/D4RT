"""Round22 direct-original-JPEG features vs prior128->448, two supervision arms."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from semantic import dino_resolution_sam_control as prior
from semantic.semantic_feature_control import train_dino,infer,read_features,save_npz,patch_at_queries,aggregate_hash
from semantic.ytvis import load_dataset,sample_queries
from semantic.fresh_video_confirmation import select_clips
from semantic.background_calibration import evaluate,predictions,pick_offset
from semantic.run import atomic_json,classification_metrics
from semantic.oracle_probe import file_hash

ROOT=Path('semantic_artifacts/dino_native_rgb_control')
ARMS=('old_gt','old_sam','native_gt','native_sam')
PROTOCOL={'round':22,'version':1,'arms':list(ARMS),'seeds':[0,1,2],'confirmation_seed':20261022,
    'new_training':'only six native GT/SAM heads','steps':2000,'optimizer':'AdamW lr.001 wd.01',
    'sampling':'128fg+128bg uniform','checkpoint_selection':'raw val macro F1 every50, ties earlier',
    'native_rgb':'same original JPEG direct PIL bilinear448; no128 downsample; no GT image inputs',
    'feature_recipe':'same frozen ViT-S14 grid32x32,384dim, pixel centers128,border,bilinear,FP16cache,FP32head',
    'source_prefix':'single-frame batch separate extraction','new_confirmation_videos':40,'final_reserve_videos':40,
    'calibration_grid':prior.PROTOCOL['calibration_grid'],'calibration_selection':prior.PROTOCOL['calibration_selection'],
    'shared_validation_repeatedly_used':True,'pretraining_ovis_overlap':'unknown','source_metadata_previously_inspected':True}


def checkpoint(arm,seed):
    if arm.startswith('old_'):return prior.checkpoint('448_'+arm.split('_')[1],seed)
    return ROOT/arm/'dinov2'/'runs'/f'linear_seed{seed}.pt'


def feature_path(arm,split,vid):
    if arm.startswith('old_')and split!='confirmation':return prior.feature_path('448_sam',split,vid)
    family='old'if arm.startswith('old_')else'native'
    return ROOT/'features'/family/split/f'{vid}.npz'


def freeze(ds):
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text())
    assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256']
    m=old['manifest'];excluded=set(m['excluded_video_ids'])|{c['video_id']for c in m['confirmation']};assert len(excluded)==500
    clips,count=select_clips(ds,excluded,count=80,seed=20261022);assert count==107
    manifest={'seed':20261022,'excluded_video_ids':sorted(excluded),'eligible_model_unevaluated':count,'train':m['train'],'val':m['val'],
              'confirmation':clips[:40],'final_reserve':clips[40:],'source_category_metadata_previously_inspected':True}
    path=ROOT/'manifest.json'
    if path.exists()and json.loads(path.read_text())!=manifest:raise ValueError('Manifest/reserve changed')
    atomic_json(path,manifest)
    paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',path,prior.resolution.WEIGHTS,Path('semantic_artifacts/ovis/annotations_train.json')]
    paths+=[checkpoint(a,s)for a in ('old_gt','old_sam')for s in range(3)]
    paths+=[checkpoint(a,s).with_suffix('.json')for a in ('old_gt','old_sam')for s in range(3)]
    paths+=[feature_path('old_sam',split,c['video_id'])for split in ('train','val')for c in manifest[split]]
    paths+=[Path(f)for split in ('train','val','confirmation','final_reserve')for c in manifest[split]for f in c['frames']]
    paths+=[Path('semantic/dino_native_rgb_control.py'),Path('semantic/OVIS_DINO_NATIVE_RGB_PROTOCOL.md')]
    for a,label in [('old_gt','448_gt'),('old_sam','448_sam')]:
        for s in range(3):assert file_hash(checkpoint(a,s))==v['checkpoint_sha256'][f'{label}_{s}']
    protocol=dict(PROTOCOL,input_sha256={str(p):file_hash(p)for p in paths},fixed_offsets={'old_gt':old['results']['448_gt']['offset'],'old_sam':old['results']['448_sam']['offset']},inherited_protocol=old['protocol'])
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=protocol:raise ValueError('Frozen protocol changed')
    atomic_json(path,protocol);return manifest,protocol


def load_native_video(clip):
    frames=[]
    for path in clip['frames']:
        with Image.open(path)as image:frames.append(torch.from_numpy(np.array(image.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),dtype=np.float32)).permute(2,0,1)/255)
    return torch.stack(frames)


def cache_features(family,splits,manifest,ds):
    assert not set(splits)&{'final_reserve'}
    tasks=[(s,c)for s in splits for c in manifest[s]if not feature_path(family+'_sam',s,c['video_id']).exists()]
    if not tasks:return
    if 'confirmation'in splits:assert (ROOT/'frozen_selection.json').exists()
    model=prior.resolution.load_dino()
    for split,clip in tasks:
        if split!='confirmation':
            with np.load(feature_path('old_sam',split,clip['video_id']))as p:meta={k:p[k].copy()for k in ('labels','xy','t_tgt','t_cam','video_id')}
        else:
            maps=np.stack([ds.gt_label_map(clip,t)for t in range(8)]);q,y,xy=sample_queries(maps,per_frame=256,seed=42+clip['video_id'])
            meta={'labels':y.numpy().astype(np.int16),'xy':xy.numpy().astype(np.int16),'t_tgt':q['t_tgt'][0].numpy(),'t_cam':q['t_cam'][0].numpy(),'video_id':np.int64(clip['video_id'])}
        video=(load_native_video(clip)if family=='native'else prior.load_video(clip)).cuda()
        tokens=prior.resolution.extract_tokens(model,video,448)
        with torch.inference_mode():
            features=patch_at_queries(tokens,meta['xy']).half().cpu().numpy()
            if split=='train':
                source=meta['xy'][:,2]==0;t=prior.resolution.extract_tokens(model,video[:1],448)
                features[source]=patch_at_queries(t,meta['xy'][source]).half().cpu().numpy();del t
        assert np.isfinite(features).all();save_npz(feature_path(family+'_sam',split,clip['video_id']),**meta,decoded=features)
        print('cached RGB',family,split,clip['video_id'],flush=True);del video,tokens
    del model;gc.collect();torch.cuda.empty_cache()


def read(arm,split,manifest):
    clips=read_features('dinov2',split,manifest,path_resolver=lambda _,s,v:feature_path(arm,s,v))
    if split=='train'and arm.endswith('_gt'):
        for c in clips:
            mask=c['xy'][:,2]==0
            for k in ('decoded','labels','xy','t_tgt','t_cam'):c[k]=c[k][mask]
    return clips


def feature_hashes(manifest,splits):
    return {f'{f}_{s}':aggregate_hash([feature_path(f+'_sam',s,c['video_id'])for c in manifest[s]])for f in ('old','native')for s in splits}


def infer_arm(arm,seed,split,clips):
    return infer('dinov2',seed,split,clips,checkpoint_path=checkpoint(arm,seed),output_root=ROOT/arm)


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed round22 exists; verify not rerun')
    ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'));manifest,protocol=freeze(ds)
    cache_features('native',('train','val'),manifest,ds);hashes=feature_hashes(manifest,('train','val'))
    for arm in ('native_gt','native_sam'):
        train=read(arm,'train',manifest);val=read(arm,'val',manifest)
        for s in range(3):
            report=train_dino(s,train,val,dict(protocol,arm=arm,feature_sha256=hashes),output_root=ROOT/arm)
            baseline=json.loads(checkpoint('old_'+arm.split('_')[1],s).with_suffix('.json').read_text())
            assert report['sampled_class_counts']==baseline['sampled_class_counts']
        del train,val
    offsets=dict(protocol['fixed_offsets']);curves={};validation={}
    previous=json.loads((prior.ROOT/'summary.json').read_text())
    for arm in ARMS:
        clips=read(arm,'val',manifest);validation[arm]=[infer_arm(arm,s,'val',clips)for s in range(3)];del clips
        if arm.startswith('native_'):offsets[arm],curves[arm]=pick_offset(validation[arm],protocol['calibration_grid'])
        else:
            for s in range(3):assert evaluate(validation[arm][s],offsets[arm])==previous['results']['448_'+arm.split('_')[1]]['runs'][s]['calibrated_validation']
    scores={a:float(np.mean([evaluate(p,offsets[a])['macro_f1_present']for p in validation[a]]))for a in ARMS};selected=max(ARMS,key=lambda a:scores[a])
    selection={'offsets':offsets,'validation_curves':curves,'validation_selected_arm':selected,'mean_validation_macro_f1':scores,
               'checkpoint_sha256':{f'{a}_{s}':file_hash(checkpoint(a,s))for a in ARMS for s in range(3)}}
    path=ROOT/'frozen_selection.json'
    if path.exists()and json.loads(path.read_text())!=selection:raise ValueError('Selection changed')
    atomic_json(path,selection);print('FROZEN RGB',selected,scores,offsets,flush=True)
    for family in ('old','native'):cache_features(family,('confirmation',),manifest,ds)
    results={}
    for arm in ARMS:
        clips=read(arm,'confirmation',manifest);runs=[]
        for s in range(3):
            p=infer_arm(arm,s,'confirmation',clips);pred=predictions(p['logits'],offsets[arm])
            per_video={str(c['video_id']):classification_metrics(pred[p['video_ids']==c['video_id']],p['labels'][p['video_ids']==c['video_id']],26)for c in manifest['confirmation']}
            runs.append({'seed':s,'raw_validation':evaluate(validation[arm][s],0),'calibrated_validation':evaluate(validation[arm][s],offsets[arm]),
                         'raw_confirmation':evaluate(p,0),'calibrated_confirmation':evaluate(p,offsets[arm]),'per_video':per_video})
            print('confirmed RGB',arm,s,runs[-1]['calibrated_confirmation']['macro_f1_present'],flush=True)
        results[arm]={'offset':offsets[arm],'runs':runs};del clips
    terms={'native_SAM':[('native_sam',1),('old_sam',-1)],'native_GT':[('native_gt',1),('old_gt',-1)],
           'SAM_native':[('native_sam',1),('native_gt',-1)],'SAM_old':[('old_sam',1),('old_gt',-1)],
           'interaction':[('native_sam',1),('native_gt',-1),('old_sam',-1),('old_gt',1)]}
    contrasts={n:prior.comparison(t,results,manifest)for n,t in terms.items()}
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    assert feature_hashes(manifest,('train','val'))==hashes
    reserve={c['video_id']for c in manifest['final_reserve']}
    assert not reserve&{c['video_id']for c in manifest['confirmation']}
    assert not any(feature_path(f+'_sam','confirmation',v).exists()for f in ('old','native')for v in reserve)
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'manifest':manifest,'feature_sha256':feature_hashes(manifest,('train','val','confirmation')),
        'frozen_selection_sha256':file_hash(path),'validation_selected_arm':selected,'results':results,'contrasts':contrasts,
        'verification':{'only_six_native_heads_trained':True,'old_models_labels_and_sampling_unchanged':True,'final40_reserved_not_evaluated':True,
                        'confirmation_excludes_prior500':True,'confirmation_after_selection_freeze':True}})
    print('DONE native RGB',contrasts['native_SAM']['macro_f1_present'],contrasts['native_SAM']['video_bootstrap'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
