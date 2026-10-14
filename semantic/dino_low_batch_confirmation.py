"""Last27 video confirmation after freezing all ten validation candidates."""
import gc
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from semantic import dino_native_rgb_control as rgb
from semantic import dino_label_weight_control as weight
from semantic import dino_gt_exposure_control as exposure
from semantic import dino_supplement_budget_control as budget
from semantic import dino_source_batch_control as batch
from semantic.semantic_feature_control import infer,read_features,save_npz,patch_at_queries
from semantic.ytvis import load_dataset,sample_queries
from semantic.fresh_video_confirmation import select_clips,bootstrap_video_differences
from semantic.background_calibration import evaluate,predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics

ROOT=Path('semantic_artifacts/dino_low_batch_confirmation')
CANDIDATES={'source_gt':(rgb,'native_gt'),'native_sam':(rgb,'native_sam'),'lambda0':(weight,'lambda0'),
    'quarter':(weight,'quarter'),'half':(weight,'half'),'prop_quarter':(exposure,'prop_quarter'),'prop_one':(exposure,'prop_one'),
    'gt_quarter':(budget,'gt_quarter'),'gt_one':(budget,'gt_one'),'source_gt32':(batch,'source_gt32')}
ARMS=tuple(CANDIDATES)
SEED=20261033


def checkpoint(arm,seed):
    module,name=CANDIDATES[arm];return module.checkpoint(name,seed)


def feature_path(vid):return ROOT/'features'/f'{vid}.npz'


def freeze(ds):
    ROOT.mkdir(parents=True,exist_ok=True);summaries={};paths=[]
    for module,_ in CANDIDATES.values():
        key=str(module.ROOT)
        if key in summaries:continue
        s=json.loads((module.ROOT/'summary.json').read_text());v=json.loads((module.ROOT/'verification.json').read_text())
        assert file_hash(module.ROOT/'summary.json')==v['summary_sha256'];summaries[key]=s
        paths.extend([module.ROOT/'summary.json',module.ROOT/'verification.json'])
    previous=summaries[str(rgb.ROOT)]['manifest'];excluded=set(previous['excluded_video_ids'])|{c['video_id']for k in ('confirmation','final_reserve')for c in previous[k]}
    assert len(excluded)==580
    clips,count=select_clips(ds,excluded,count=27,seed=SEED);assert count==27
    m={'seed':SEED,'excluded_video_ids':sorted(excluded),'eligible_model_unevaluated':count,'confirmation':clips,
       'source_category_metadata_previously_inspected':True}
    mp=ROOT/'manifest.json'
    if mp.exists()and json.loads(mp.read_text())!=m:raise ValueError('Last27 frozen manifest changed')
    atomic_json(mp,m);selection={}
    for arm,(module,name)in CANDIDATES.items():
        r=summaries[str(module.ROOT)]['results'][name];values=[v['calibrated_validation']['macro_f1_present']for v in r['runs']]
        selection[arm]={'offset':r['offset'],'validation_macro_by_seed':values,'mean_validation_macro':float(np.mean(values)),
             'checkpoint_sha256':{str(s):file_hash(checkpoint(arm,s))for s in range(3)}}
        paths.extend([checkpoint(arm,s)for s in range(3)])
    selected=max(ARMS,key=lambda a:selection[a]['mean_validation_macro'])
    paths.extend([mp,Path('semantic/dino_low_batch_confirmation.py'),Path('semantic/OVIS_DINO_LOW_BATCH_CONFIRMATION_PROTOCOL.md'),
        Path('semantic_artifacts/ovis/annotations_train.json'),rgb.prior.resolution.WEIGHTS])
    paths.extend(Path(p)for c in clips for p in c['frames'])
    vendor=rgb.prior.resolution.VENDOR
    assert file_hash(rgb.prior.resolution.WEIGHTS)=='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()=='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=vendor,check=True)
    paths.extend(sorted(vendor.rglob('*.py')))
    p={'round':33,'candidate_order':list(ARMS),'selected_before_confirmation':selected,'selection':selection,'confirmation_seed':SEED,
       'confirmation_videos':27,'bootstrap_seed':SEED,'bootstrap_repetitions':2000,'training_updates':0,'deployment_change':False,
       'source_category_metadata_previously_inspected':True,'pretraining_ovis_overlap':'unknown','shared_validation_repeatedly_used':True,
       'input_sha256':{str(p):file_hash(p)for p in paths}}
    pp=ROOT/'protocol.json';sp=ROOT/'frozen_selection.json'
    if pp.exists()and json.loads(pp.read_text())!=p:raise ValueError('Frozen confirmation protocol changed')
    atomic_json(pp,p)
    frozen={'selected':selected,'selection':selection,'deployment_unchanged':'native_gt'}
    if sp.exists()and json.loads(sp.read_text())!=frozen:raise ValueError('Frozen selection changed')
    atomic_json(sp,frozen);return m,p


def cache(ds,m):
    assert (ROOT/'frozen_selection.json').exists();tasks=[c for c in m['confirmation']if not feature_path(c['video_id']).exists()]
    if not tasks:return
    model=rgb.prior.resolution.load_dino()
    for clip in tasks:
        maps=np.stack([ds.gt_label_map(clip,t)for t in range(8)]);q,y,xy=sample_queries(maps,per_frame=256,seed=42+clip['video_id'])
        video=rgb.load_native_video(clip).cuda();tokens=rgb.prior.resolution.extract_tokens(model,video,448)
        with torch.inference_mode():features=patch_at_queries(tokens,xy.numpy()).half().cpu().numpy()
        assert np.isfinite(features).all()
        save_npz(feature_path(clip['video_id']),decoded=features,labels=y.numpy().astype(np.int16),xy=xy.numpy().astype(np.int16),
            t_tgt=q['t_tgt'][0].numpy(),t_cam=q['t_cam'][0].numpy(),video_id=np.int64(clip['video_id']))
        print('cached last27',clip['video_id'],flush=True);del video,tokens
    del model;gc.collect();torch.cuda.empty_cache()


def contrast(left,right,results,m):
    d=[results[left]['runs'][s]['calibrated_confirmation']['macro_f1_present']-results[right]['runs'][s]['calibrated_confirmation']['macro_f1_present']for s in range(3)]
    v={str(c['video_id']):float(np.mean([results[left]['runs'][s]['per_video'][str(c['video_id'])]['macro_f1_present']-
        results[right]['runs'][s]['per_video'][str(c['video_id'])]['macro_f1_present']for s in range(3)]))for c in m['confirmation']}
    return {'left':left,'right':right,'per_seed_macro_difference':d,'mean_macro_difference':float(np.mean(d)),
        'per_video_differences':v,'video_bootstrap':bootstrap_video_differences(list(v.values()),seed=SEED,repetitions=2000)}


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed last27 confirmation exists')
    ds=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'));m,p=freeze(ds);cache(ds,m)
    features={str(feature_path(c['video_id'])):file_hash(feature_path(c['video_id']))for c in m['confirmation']}
    clips=read_features('dinov2','confirmation',m,path_resolver=lambda _,split,vid:feature_path(vid));results={}
    for arm in ARMS:
        offset=p['selection'][arm]['offset'];runs=[]
        for seed in range(3):
            package=infer('dinov2',seed,'confirmation',clips,checkpoint_path=checkpoint(arm,seed),output_root=ROOT/arm)
            pred=predictions(package['logits'],offset)
            pv={str(c['video_id']):classification_metrics(pred[package['video_ids']==c['video_id']],package['labels'][package['video_ids']==c['video_id']],26)for c in m['confirmation']}
            runs.append({'seed':seed,'raw_confirmation':evaluate(package,0),'calibrated_confirmation':evaluate(package,offset),'per_video':pv})
            print('last27 frozen',arm,seed,runs[-1]['calibrated_confirmation']['macro_f1_present'],flush=True)
        results[arm]={'offset':offset,'runs':runs}
    selected=p['selected_before_confirmation'];primary=contrast(selected,'source_gt',results,m)
    pairs=[('native_sam','source_gt'),('lambda0','native_sam'),('lambda0','source_gt'),('quarter','native_sam'),('half','native_sam'),
       ('prop_quarter','gt_quarter'),('prop_one','gt_one'),('source_gt32','lambda0'),('source_gt32','source_gt')]
    secondary={a+'_minus_'+b:contrast(a,b,results,m)for a,b in pairs}
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    for path,h in features.items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'manifest':m,'feature_sha256':features,'results':results,'primary_contrast':primary,
        'secondary_descriptive_contrasts':secondary,'frozen_selection_sha256':file_hash(ROOT/'frozen_selection.json'),
        'deployment_unchanged':'native_gt','model_evaluated_videos_after_round':607,'model_unevaluated_videos_remaining':0,
        'verification':{'all_weights_inputs_and_offsets_unchanged':True,'selection_fixed_before_features':True,'no_training_or_recalibration':True,
        'all27_videos_and_three_seeds_retained':True},'limits':['27 videos with previously inspected source category metadata',
        'shared validation adapted across many rounds','secondary intervals unadjusted for multiplicity','not OVIS AP or D4RT truth geometry','backbone pretraining overlap unknown']})
    print('DONE last27 independent confirmation',selected,primary['mean_macro_difference'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)});raise
