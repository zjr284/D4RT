"""Matched-budget source-frame GT-only control, with validation-only calibration."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from semantic.head_budget_probe import trajectory
from semantic.weaklabel_budget_probe import read_split,CACHES
from semantic.background_calibration import logits_for,pick_offset,evaluate,PROTOCOL as CAL_PROTOCOL
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/source_gt_control')
PROTOCOL={'version':1,'heads':['linear'],'budgets':[500,2000],'seeds':[0,1,2],
          'train':'only original source-frame GT query features, t_src=0; no later labels',
          'optimizer':'AdamW lr=0.001 weight_decay=0.01','batch':'128 foreground + 128 background',
          'selection':'validation macro F1 every 50 steps','calibration_grid':CAL_PROTOCOL['offset_grid'],
          'calibration_selection':CAL_PROTOCOL['offset_selection'],'ties':CAL_PROTOCOL['ties'],
          'comparison':'GT-only vs SAM positive vs SAM positive/background, linear 2000, same calibration method'}


def source_clips(clips):
    result=[]
    for c in clips:
        selected=c['xy'][:,2]==0
        if not selected.any():raise ValueError('No source-frame queries')
        result.append({k:(v[selected].copy() if k in ('decoded','labels','xy') else v.copy()) for k,v in c.items() if k in ('decoded','labels','xy','video_id')})
    return result


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    cal_path=Path('semantic_artifacts/background_calibration/summary.json');cal=json.loads(cal_path.read_text())
    for split in ('val','test'):
        h=hashlib.sha256()
        for p in sorted((CACHES['positive_only']/split).glob('*.npz')):h.update(p.name.encode());h.update(bytes.fromhex(file_hash(p)))
        if h.hexdigest()!=cal['protocol']['heldout_cache_sha256'][split]:raise ValueError('Calibration held-out cache changed')
    protocol=dict(PROTOCOL,manifest_sha256=file_hash('semantic_artifacts/manifest.json'),previous_calibration_summary_sha256=file_hash(cal_path))
    h=hashlib.sha256()
    for p in sorted((CACHES['positive_only']/'train').glob('*.npz')):h.update(p.name.encode());h.update(bytes.fromhex(file_hash(p)))
    protocol['original_training_cache_sha256']=h.hexdigest()
    file=ROOT/'protocol.json'
    if file.exists() and json.loads(file.read_text())!=protocol:raise ValueError('Changed protocol/inputs')
    atomic_json(file,protocol)
    train=source_clips(read_split(CACHES['positive_only'],'train'));val=read_split(CACHES['positive_only'],'val');test=read_split(CACHES['positive_only'],'test')
    manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text())
    if [len(c) for c in (train,val,test)]!=[60,20,20]:raise ValueError('Incomplete splits')
    for split,clips in [('train',train),('val',val),('test',test)]:
        if {int(c['video_id']) for c in clips}!={int(c['video_id']) for c in manifest[split]}:raise ValueError('Manifest mismatch')
    labels=np.concatenate([c['labels'] for c in train]);times=np.concatenate([c['xy'][:,2] for c in train]);x=np.concatenate([c['decoded'] for c in train])
    if np.any(times!=0) or not np.isfinite(x).all():raise ValueError('GT source control leaked later times or invalid features')
    features=torch.from_numpy(x).float().to('cuda');del x
    runs=[trajectory('linear',seed,features,labels,times,train,val,test,protocol,output_root=ROOT) for seed in PROTOCOL['seeds']]
    prior=json.loads(Path('semantic_artifacts/runs_ovis_balanced/gt_results.json').read_text())
    for i,r in enumerate(runs):
        for metric in ('accuracy','macro_f1_present','foreground_macro_f1_present'):
            if abs(r['budgets']['500']['test'][metric]-prior['runs'][i]['test'][metric])>1e-12:raise ValueError('GT-only 500 baseline mismatch')
    packages=[logits_for('source_gt',seed,'val',val,checkpoint_path=ROOT/'runs'/f'linear_2000_seed{seed}.pt',output_root=ROOT) for seed in PROTOCOL['seeds']]
    offset,curve=pick_offset(packages,PROTOCOL['calibration_grid'])
    calibrated=[]
    for seed in PROTOCOL['seeds']:
        package=logits_for('source_gt',seed,'test',test,checkpoint_path=ROOT/'runs'/f'linear_2000_seed{seed}.pt',output_root=ROOT)
        before=evaluate(package,0.0);after=evaluate(package,offset)
        for metric in ('accuracy','macro_f1_present','foreground_macro_f1_present'):
            if abs(before[metric]-runs[seed]['budgets']['2000']['test'][metric])>1e-12:raise ValueError('Raw calibration mismatch')
        calibrated.append({'seed':seed,'before_test':before,'after_test':after,'before_validation':evaluate(packages[seed],0.0),'after_validation':evaluate(packages[seed],offset)})
    comparisons={'source_gt':{'selected_offset':offset,'runs':calibrated},**cal['results']}
    contrasts={}
    for arm in ('positive_only','positive_background'):
        contrasts[arm+'_minus_source_gt']={}
        for state in ('before_test','after_test'):
            contrasts[arm+'_minus_source_gt'][state]={}
            for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_accuracy'):
                d=[comparisons[arm]['runs'][i][state][metric]-calibrated[i][state][metric] for i in range(3)]
                contrasts[arm+'_minus_source_gt'][state][metric]={'paired_deltas':d,'mean':float(np.mean(d)),'std':float(np.std(d,ddof=1))}
    selected=max(comparisons,key=lambda arm:np.mean([r['after_validation']['macro_f1_present'] for r in comparisons[arm]['runs']]))
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'source_gt_trajectories':runs,'selected_offset':offset,'validation_curve':curve,
                  'comparisons':comparisons,'contrasts':contrasts,'validation_selected_arm':selected,
                  'verification':{'all_training_queries_tsrc_zero':True,'three_gt_500_baselines_reproduced':True,
                                  'zero_offset_2000_reproduced':True,'calibration_validation_only':True}})
    print('DONE source GT offset',offset,'validation-selected arm',selected,flush=True)

if __name__=='__main__':main()
