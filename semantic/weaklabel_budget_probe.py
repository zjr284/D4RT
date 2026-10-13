"""Weak-label foreground/background comparison at matched head and step budgets."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from semantic.head_budget_probe import trajectory, PROTOCOL as HEAD_PROTOCOL
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/weaklabel_budget_probe')
CACHES={'positive_only':Path('semantic_artifacts/features_ovis/sam'),
        'positive_background':Path('semantic_artifacts/background_ablation/features/sam')}
PROTOCOL=dict(HEAD_PROTOCOL,version=1,diagnostic_training_gt=False,
              label_arms=list(CACHES),train_labels='source-frame GT + fixed SAM positive pseudo labels, optional background pseudo labels',
              fixed_original_foreground_queries=True,unchanged_val_test=True)


def read_split(folder,split):
    clips=[]
    for p in sorted((folder/split).glob('*.npz')):
        with np.load(p,allow_pickle=False) as data:
            clips.append({k:data[k].copy() for k in ('decoded','labels','xy','video_id','alternate_decoded') if k in data.files})
    return clips


def hash_folder(folder):
    h=hashlib.sha256()
    for p in sorted(folder.glob('*.npz')):h.update(p.name.encode());h.update(bytes.fromhex(file_hash(p)))
    return h.hexdigest()


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    manifest_path=Path('semantic_artifacts/manifest.json');manifest=json.loads(manifest_path.read_text())
    protocol=dict(PROTOCOL,manifest_sha256=file_hash(manifest_path),
                  training_hashes={arm:hash_folder(folder/'train') for arm,folder in CACHES.items()},
                  validation_hash=hash_folder(CACHES['positive_only']/'val'),test_hash=hash_folder(CACHES['positive_only']/'test'))
    for split in ('val','test'):
        if (CACHES['positive_background']/split).resolve()!=(CACHES['positive_only']/split).resolve():
            raise ValueError('Held-out caches must be identical')
    old=ROOT/'protocol.json'
    if old.exists() and json.loads(old.read_text())!=protocol:raise ValueError('Protocol/cache changed')
    atomic_json(old,protocol)
    val=read_split(CACHES['positive_only'],'val');test=read_split(CACHES['positive_only'],'test')
    if len(val)!=20 or len(test)!=20:raise ValueError('Need full held-out split')
    for split,clips in [('val',val),('test',test)]:
        if {int(c['video_id']) for c in clips}!={c['video_id'] for c in manifest[split]}:raise ValueError('Held-out manifest mismatch')
    results={}
    for arm,folder in CACHES.items():
        train=read_split(folder,'train')
        if len(train)!=60 or {int(c['video_id']) for c in train}!={c['video_id'] for c in manifest['train']}:raise ValueError('Training manifest mismatch')
        for split,clips in [('val',val),('test',test)]:
            if {int(c['video_id']) for c in train}&{int(c['video_id']) for c in clips}:raise ValueError('Video leakage')
        if arm=='positive_background':
            for c in train:
                with np.load(CACHES['positive_only']/'train'/f"{int(c['video_id'])}.npz") as original:
                    n=len(original['labels'])
                    for k in ('decoded','xy','labels'):
                        if not np.array_equal(c[k][:n],original[k]):raise ValueError('Original queries modified')
                    if np.any(c['labels'][n:]!=0) or np.any(c['xy'][n:,2]<=0):raise ValueError('Added queries must be later background')
        x=np.concatenate([c['decoded'] for c in train]);labels=np.concatenate([c['labels'] for c in train]);times=np.concatenate([c['xy'][:,2] for c in train])
        if not np.isfinite(x).all() or np.any((labels<0)|(labels>=26)):raise ValueError('Invalid cache')
        features=torch.from_numpy(x).float().to('cuda');del x
        local_protocol=dict(protocol,label_arm=arm)
        print('START',arm,flush=True)
        for kind in PROTOCOL['heads']:
            runs=[trajectory(kind,seed,features,labels,times,train,val,test,local_protocol,output_root=ROOT/arm) for seed in PROTOCOL['seeds']]
            for budget in PROTOCOL['budgets']:results[f'{arm}_{kind}_{budget}']=[r['budgets'][str(budget)] for r in runs]
        del features,train;gc.collect();torch.cuda.empty_cache()
    baseline=json.loads(Path('semantic_artifacts/background_ablation/summary.json').read_text())
    for arm,prior in [('positive_only','positive_only'),('positive_background','positive_and_background')]:
        for i,r in enumerate(results[f'{arm}_linear_500']):
            for metric in ('accuracy','macro_f1_present','foreground_macro_f1_present'):
                if abs(r['test'][metric]-baseline['comparisons'][prior][i]['test'][metric])>1e-12:raise ValueError('500 step prior baseline mismatch')
    selected={}
    for arm in CACHES:
        groups=[f'{arm}_{kind}_{budget}' for kind in PROTOCOL['heads'] for budget in PROTOCOL['budgets']]
        selected[arm]=max(groups,key=lambda g:np.mean([r['best_val_macro_f1'] for r in results[g]]))
    overall=max(results,key=lambda g:np.mean([r['best_val_macro_f1'] for r in results[g]]))
    contrasts={}
    pairs=[]
    for kind in PROTOCOL['heads']:
        for budget in PROTOCOL['budgets']:pairs.append((f'positive_background_{kind}_{budget}',f'positive_only_{kind}_{budget}'))
    for arm in CACHES:
        for kind in PROTOCOL['heads']:pairs.append((f'{arm}_{kind}_2000',f'{arm}_{kind}_500'))
    for treatment,control in pairs:
        contrast={}
        for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_accuracy','foreground_to_background_rate'):
            d=[results[treatment][i]['test'][metric]-results[control][i]['test'][metric] for i in range(3)]
            contrast[metric]={'paired_deltas':d,'mean':float(np.mean(d)),'std':float(np.std(d,ddof=1))}
        bg=[results[treatment][i]['test']['per_class_f1']['0']-results[control][i]['test']['per_class_f1']['0'] for i in range(3)]
        contrast['background_f1']={'paired_deltas':bg,'mean':float(np.mean(bg)),'std':float(np.std(bg,ddof=1))}
        contrasts[f'{treatment}_minus_{control}']=contrast
    for arm in CACHES:
        for budget in PROTOCOL['budgets']:
            for i in range(3):
                a=results[f'{arm}_linear_{budget}'][i];b=results[f'{arm}_mlp_{budget}'][i]
                if a['sampling_histogram_time_class']!=b['sampling_histogram_time_class']:raise ValueError('Head sampling differs')
    for runs in results.values():
        for r in runs:
            h=np.array(r['sampling_histogram_time_class'])
            if h[:,0].sum()!=128*r['budget'] or h[:,1:].sum()!=128*r['budget']:raise ValueError('Wrong training budget')
            if r['best_step']>r['budget']:raise ValueError('Checkpoint budget leakage')
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'results':results,'contrasts':contrasts,
               'validation_selected_per_arm':selected,'validation_selected_overall':overall,
               'verification':{'two_500_step_linear_baselines_reproduced':True,'heldout_queries_identical':True,
               'original_training_prefixes_preserved':True,'head_sampling_identical_per_arm_seed_budget':True,'budgets_verified':True}})
    print('DONE selected',overall,flush=True)

if __name__=='__main__':main()
