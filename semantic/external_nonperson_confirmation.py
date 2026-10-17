"""Frozen four-arm non-Person case-panel check; no tuning or reserve evaluation."""
from collections import defaultdict
import fcntl
import gc
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from semantic import dino_native_rgb_control as rgb
from semantic.external_coco_adapter import ANN,OVIS,label_map
from semantic.external_semantic_confirmation import contrast
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.semantic_feature_control import infer,save_npz,patch_at_queries
from semantic.background_calibration import predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics

ROOT=Path('semantic_artifacts/external_nonperson_confirmation')
PANEL=Path('semantic_artifacts/external_class_panel')
PRIOR=Path('semantic_artifacts/external_semantic_confirmation')
MAPPING=Path('semantic_artifacts/external_coco_adapter/summary.json')
ARMS=('source_gt','source_gt32','lambda0','native_sam')


def feature_path(group):return ROOT/'features'/f'{group}.npz'


def nonperson_queries(labels,image_id,slot):
    points=[];targets=[]
    for cid in sorted(int(c)for c in np.unique(labels)if c>1):
        pool=np.flatnonzero(labels.ravel()==cid)
        chosen=np.random.default_rng(20261052+int(image_id)*32+cid).choice(pool,64,replace=len(pool)<64)
        y,x=np.divmod(chosen,128);points.append(np.column_stack((x,y,np.full(64,slot))).astype(np.int16));targets.append(np.full(64,cid,np.int16))
    return (np.concatenate(points),np.concatenate(targets))if points else(np.empty((0,3),np.int16),np.empty(0,np.int16))


def freeze():
    verify=json.loads((PANEL/'verification.json').read_text());assert file_hash(PANEL/'summary.json')==verify['summary_sha256']
    eligibility=json.loads((PANEL/'frozen_eligibility.json').read_text());m=json.loads((PANEL/'manifest.json').read_text())
    clips=[c for c in m['panel']if c['group']in eligibility['accepted_groups']];assert [c['group']for c in clips]==eligibility['accepted_groups']
    prior=json.loads((PRIOR/'protocol.json').read_text());frozen=json.loads((PRIOR/'frozen_selection.json').read_text());selection=frozen['selection']
    assert torch.__version__==prior['torch_version']and torch.version.cuda==prior['cuda_version']and torch.cuda.get_device_name(0)==prior['device']
    assert frozen['selected']=='source_gt32'
    vendor=rgb.prior.resolution.VENDOR;weights=rgb.prior.resolution.WEIGHTS
    assert file_hash(weights)=='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()=='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=vendor,check=True)
    paths=[PANEL/'summary.json',PANEL/'verification.json',PANEL/'manifest.json',PANEL/'frozen_eligibility.json',PANEL/'image_records.json',
        PRIOR/'protocol.json',PRIOR/'frozen_selection.json',PRIOR/'summary.json',PRIOR/'verification.json',PRIOR/'compatibility.json',
        ANN,OVIS,MAPPING,weights,Path('semantic/external_nonperson_confirmation.py'),Path('semantic/OVIS_EXTERNAL_NONPERSON_CONFIRMATION_PROTOCOL.md'),
        Path('semantic/external_semantic_confirmation.py'),Path('semantic/external_coco_adapter.py'),Path('semantic/dino_native_rgb_control.py'),
        Path('semantic/dino_resolution_control.py'),Path('semantic/semantic_feature_control.py'),Path('semantic/background_calibration.py'),Path('semantic/run.py')]
    paths.extend(sorted(vendor.rglob('*.py')))
    for c in clips:paths.extend(PANEL/'images'/im['file_name']for im in c['frames'])
    for arm in ARMS:
        for seed in range(3):
            path=checkpoint(arm,seed);assert file_hash(path)==selection[arm]['checkpoint_sha256'][str(seed)];paths.append(path)
    p={'round':53,'scope':'non-Person metadata-directed case panel, not random population','candidate_groups':[c['group']for c in clips],
        'excluded_original80':m['excluded_original80'],'metadata_selected_classes':[int(cid)for cid,r in m['selection_by_class'].items()if r['selected_group']is not None],
        'selection':selection,'selected_by_OVIS_validation':'source_gt32','primary':['source_gt32','source_gt'],
        'secondary':[['native_sam','source_gt'],['lambda0','source_gt']],'bootstrap_seed':20261053,'bootstrap_repetitions':2000,
        'query_seed_base':20261052,'per_frame_per_class_queries':64,'Person_excluded_only_from_evaluation_queries':True,'prediction_classes':26,
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0),'head_TF32_disabled':True,
        'training_updates':0,'calibration_updates':0,'input_sha256':{str(path):file_hash(path)for path in paths}}
    for name,data in [('protocol.json',p),('manifest.json',{'confirmation':clips}),('frozen_selection.json',{'selected':'source_gt32','selection':selection,'deployment_unchanged':'native_gt'})]:
        path=ROOT/name
        if path.exists():assert json.loads(path.read_text())==data
        else:atomic_json(path,data)
    return clips,p


def cache_features(clips):
    tasks=[c for c in clips if not feature_path(c['group']).exists()]
    if not tasks:return
    payload=json.loads(ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(MAPPING.read_text())['shared_category_mapping'].items()}
    by_image=defaultdict(list)
    for ann in payload['annotations']:by_image[ann['image_id']].append(ann)
    model=rgb.prior.resolution.load_dino();qp=ROOT/'query_coverage.json';coverage=json.loads(qp.read_text())if qp.exists()else[]
    for clip in tasks:
        points=[];targets=[];local=[]
        for slot,im in enumerate(clip['frames']):
            labels=label_map(by_image[im['id']],mapping,im['height'],im['width']);xy,y=nonperson_queries(labels,im['id'],slot)
            points.append(xy);targets.append(y);local.append({'group':clip['group'],'time':slot,'image_id':im['id'],
                'nonperson_pixels':int(np.sum(labels>1)),'ignored_pixels':int(np.sum(labels<0)),'query_count':len(y),
                'query_class_counts':{str(int(cid)):int(n)for cid,n in zip(*np.unique(y,return_counts=True))}})
        xy=np.concatenate(points).astype(np.int16);y=np.concatenate(targets).astype(np.int16)
        if len(y):
            video=rgb.load_native_video({'frames':[str(PANEL/'images'/im['file_name'])for im in clip['frames']]}).cuda()
            tokens=rgb.prior.resolution.extract_tokens(model,video,448)
            with torch.inference_mode():features=patch_at_queries(tokens,xy).half().cpu().numpy()
            assert np.isfinite(features).all();del video,tokens
        else:features=np.empty((0,384),np.float16)
        coverage=[r for r in coverage if r['group']!=clip['group']]+local;atomic_json(qp,coverage)
        save_npz(feature_path(clip['group']),decoded=features,xy=xy,labels=y,video_id=np.int64(int(clip['group'])))
        print('nonPerson frozen features',clip['group'],len(y),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'summary.json').exists():raise RuntimeError('Completed class-panel confirmation exists')
        torch.backends.cuda.matmul.allow_tf32=False
        clips,p=freeze();cache_features(clips);features={str(feature_path(c['group'])):file_hash(feature_path(c['group']))for c in clips}
        fp=ROOT/'feature_hashes.json'
        if fp.exists():assert json.loads(fp.read_text())==features
        else:atomic_json(fp,features)
        packages=[]
        for c in clips:
            with np.load(feature_path(c['group']))as data:packages.append({k:data[k].copy()for k in data.files})
        if not sum(len(c['labels'])for c in packages):raise ValueError('No valid nonPerson queries; do not replace panel')
        results={}
        for arm in ARMS:
            runs=[];offset=p['selection'][arm]['offset']
            for seed in range(3):
                out=infer('dinov2',seed,'confirmation',packages,checkpoint_path=checkpoint(arm,seed),output_root=ROOT/arm)
                pred=predictions(out['logits'],offset);per_video={}
                for c in clips:
                    rows=out['video_ids']==int(c['group']);per_video[c['group']]=classification_metrics(pred[rows],out['labels'][rows],26)if np.any(rows)else None
                runs.append({'seed':seed,'raw':classification_metrics(predictions(out['logits'],0),out['labels'],26),
                    'calibrated':classification_metrics(pred,out['labels'],26),'per_video':per_video})
                print('nonPerson frozen inference',arm,seed,runs[-1]['calibrated']['macro_f1_present'],flush=True)
            results[arm]={'offset':offset,'runs':runs}
        primary=contrast(*p['primary'],results,clips,p);secondary={a+'_minus_'+b:contrast(a,b,results,clips,p)for a,b in p['secondary']}
        labels=np.concatenate([c['labels']for c in packages]);counts={c['group']:len(d['labels'])for c,d in zip(clips,packages)}
        coverage=json.loads((ROOT/'query_coverage.json').read_text())
        for path,h in p['input_sha256'].items():assert file_hash(path)==h
        for path,h in features.items():assert file_hash(path)==h
        assert not set(p['excluded_original80'])&{path.stem for path in (ROOT/'features').glob('*.npz')}
        atomic_json(ROOT/'summary.json',{'protocol':p,'results':results,'primary_contrast':primary,'secondary_contrasts_unadjusted':secondary,
            'query_counts':counts,'empty_groups':[g for g,n in counts.items()if not n],'empty_frames':sum(not r['query_count']for r in coverage),
            'total_candidate_groups':len(clips),'query_bearing_groups':sum(bool(n)for n in counts.values()),'total_queries':len(labels),
            'actual_support':{str(int(cid)):int(n)for cid,n in zip(*np.unique(labels,return_counts=True))},
            'metadata_classes_without_queries':sorted(set(p['metadata_selected_classes'])-set(map(int,np.unique(labels)))),
            'query_coverage_sha256':file_hash(ROOT/'query_coverage.json'),'feature_hashes':features,
            'original80_model_evaluated_again':False,'training_or_calibration_updates':0,'deployment_unchanged':'native_gt'})
        print('DONE frozen nonPerson panel',primary['conditional_video_bootstrap'],flush=True)


if __name__=='__main__':main()
