"""Diagnostic oracle labels and GT coverage; never a deployable weak-label result."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from semantic.sampling_ablation import train_one, load_split, extended_diagnostic
from semantic.model import extract_query_features, SemanticHead
from semantic.ytvis import load_dataset, sample_queries, load_video
from semantic.run import atomic_json
from tools.load_opend4rt import load_model, DEFAULT_VARIANT

ROOT=Path('semantic_artifacts/oracle_probe')
BASE=Path('semantic_artifacts/background_ablation/features/sam')
ARMS=('pseudo','oracle_relabel','oracle_coverage')
PROTOCOL={'version':1,'arms':list(ARMS),'seeds':[0,1,2],'steps':500,
          'sampling':'128 foreground + 128 background, uniformly within label group',
          'optimizer':'AdamW lr=0.001 weight_decay=0.01','selection':'validation macro F1 every 50 steps',
          'oracle_relabel':'same existing feature queries, GT labels, discard GT overlap (-1)',
          'oracle_coverage':'all eight GT frames; sample 128 foreground and 128 background per frame',
          'frozen_encoder_decoder':True,'query_seed':42,'unchanged_val_test':True,
          'diagnostic_only_full_training_gt':True}


def file_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def relabel_queries(xy, maps):
    xy=np.asarray(xy)
    if xy.ndim!=2 or xy.shape[1]!=3:raise ValueError('Expected x,y,t coordinates')
    if np.any(xy<0) or np.any(xy[:,0]>=maps.shape[2]) or np.any(xy[:,1]>=maps.shape[1]) or np.any(xy[:,2]>=maps.shape[0]):
        raise ValueError('Out of bounds query')
    return maps[xy[:,2],xy[:,1],xy[:,0]].astype(np.int16)


def relabel(dataset,manifest):
    out=ROOT/'features/oracle_relabel';out.mkdir(parents=True,exist_ok=True)
    changes=np.zeros((26,26),dtype=np.int64);dropped=total=0
    for clip in manifest['train']:
        vid=clip['video_id']
        with np.load(BASE/'train'/f'{vid}.npz') as package:
            original={k:package[k].copy() for k in ('decoded','labels','xy','video_id')}
        maps=np.stack([dataset.gt_label_map(clip,t) for t in range(8)])
        targets=relabel_queries(original['xy'],maps);valid=targets>=0
        np.add.at(changes,(original['labels'][valid],targets[valid]),1)
        dropped+=int((~valid).sum());total+=len(targets)
        target=out/f'{vid}.npz'
        if not target.exists():
            tmp=target.with_suffix('.partial.npz')
            np.savez_compressed(tmp,decoded=original['decoded'][valid],labels=targets[valid],
                                xy=original['xy'][valid],video_id=original['video_id']);tmp.replace(target)
    audit={'pseudo_to_gt_confusion':changes.tolist(),'original_query_count':total,'ignored_gt_queries_removed':dropped,
           'correct_label_fraction_nonignored':float(np.trace(changes)/changes.sum())}
    atomic_json(ROOT/'relabel_audit.json',audit)
    return out


def coverage(dataset,manifest):
    out=ROOT/'features/oracle_coverage';out.mkdir(parents=True,exist_ok=True)
    missing=[c for c in manifest['train'] if not (out/f"{c['video_id']}.npz").exists()]
    if not missing:return out
    model=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval()
    for clip in missing:
        vid=clip['video_id'];maps=np.stack([dataset.gt_label_map(clip,t) for t in range(8)])
        query,targets,xy=sample_queries(maps,per_frame=256,seed=42+int(vid))
        with np.load(BASE/'train'/f'{vid}.npz') as package:
            memory=torch.from_numpy(package['memory'].copy()).to('cuda',dtype=torch.bfloat16)[None]
        video=load_video(clip)[None].to('cuda',dtype=torch.bfloat16);chunks=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for start in range(0,len(targets),64):
                small={k:v[:,start:start+64].to('cuda') for k,v in query.items()}
                chunk=extract_query_features(model,video,small,memory)['decoded'][0].to(torch.float16).cpu().numpy()
                chunks.append(chunk)
        target=out/f'{vid}.npz';tmp=target.with_suffix('.partial.npz')
        np.savez_compressed(tmp,decoded=np.concatenate(chunks),labels=targets.numpy().astype(np.int16),
                            xy=xy.numpy().astype(np.int16),video_id=np.int64(vid));tmp.replace(target)
        print('GT coverage cache',vid,len(targets),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    return out


def read_oracle(folder):
    return [dict(np.load(p,allow_pickle=False)) for p in sorted(folder.glob('*.npz'))]


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    previous=Path('semantic_artifacts/sampling_ablation/protocol.json')
    protocol=dict(PROTOCOL,manifest_sha256=file_hash('semantic_artifacts/manifest.json'),
                  prior_protocol_sha256=file_hash(previous),annotation_sha256=file_hash('semantic_artifacts/ovis/annotations_train.json'),
                  model_config_sha256=file_hash(DEFAULT_VARIANT/'model.yaml'))
    # Verify baseline content hash before exposing training GT.
    h=hashlib.sha256()
    for split in ('train','val','test'):
        for p in sorted((BASE/split).glob('*.npz')):
            h.update(f'{split}/{p.name}'.encode())
            with p.open('rb') as f:
                while chunk:=f.read(1024*1024):h.update(chunk)
    if h.hexdigest()!=json.loads(previous.read_text())['cache_sha256']:raise ValueError('Baseline cache changed')
    protocol['baseline_cache_sha256']=h.hexdigest()
    p=ROOT/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Changed protocol or inputs')
    atomic_json(p,protocol)
    manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text())
    if [len(manifest[s]) for s in ('train','val','test')]!=[60,20,20]:raise ValueError('Wrong splits')
    ids=[{c['video_id'] for c in manifest[s]} for s in ('train','val','test')]
    if ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]:raise ValueError('Video overlap')
    dataset=load_dataset('ovis',Path('semantic_artifacts/ovis/annotations_train.json'),Path('semantic_artifacts/ovis'))
    folders={'oracle_relabel':relabel(dataset,manifest),'oracle_coverage':coverage(dataset,manifest)}
    val=load_split('val');test=load_split('test')
    previous_results=json.loads(Path('semantic_artifacts/sampling_ablation/summary.json').read_text())
    results={'pseudo':previous_results['results']['baseline']};distributions={}
    for arm,folder in folders.items():
        clips=read_oracle(folder)
        if {int(c['video_id']) for c in clips}!=ids[0]:raise ValueError('Oracle contains nontraining videos')
        labels=np.concatenate([c['labels'] for c in clips]);times=np.concatenate([c['xy'][:,2] for c in clips]);x=np.concatenate([c['decoded'] for c in clips])
        if not np.isfinite(x).all() or np.any((labels<0)|(labels>=26)):raise ValueError('Invalid oracle cache')
        hist=np.zeros((8,26),dtype=np.int64);np.add.at(hist,(times,labels),1);distributions[arm]=hist.tolist()
        features=torch.from_numpy(x).float().to('cuda');del x
        results[arm]=[]
        for seed in PROTOCOL['seeds']:
            p=ROOT/'runs'/f'{arm}_seed{seed}.json'
            if p.exists():r=json.loads(p.read_text())
            else:r=train_one('baseline',seed,features,labels,times,val,test,output_root=ROOT,run_name=arm,protocol=protocol)
            # Selected-model fitting on training queries distinguishes fitting from cross-video generalization.
            model=SemanticHead(num_classes=26).to('cuda').eval()
            model.load_state_dict(torch.load(ROOT/'runs'/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)['model'])
            r['train_diagnostic']=extended_diagnostic(model,clips)
            categories=[int(c) for c in r['test']['support'] if c!='0']
            r['matched_class_diagnostic']={
                'foreground_categories':categories,
                'train_macro_recall_same_classes':float(np.mean([r['train_diagnostic']['per_class_recall'][str(c)] for c in categories])),
                'test_macro_recall_same_classes':float(np.mean([r['test']['per_class_recall'][str(c)] for c in categories])),
                'train_recall_weighted_by_test_support':float(sum(r['train_diagnostic']['per_class_recall'][str(c)]*r['test']['support'][str(c)] for c in categories)/sum(r['test']['support'][str(c)] for c in categories))}
            atomic_json(p,r);results[arm].append(r)
        del features,clips;gc.collect();torch.cuda.empty_cache()
    contrasts={}
    for arm,control in (('oracle_relabel','pseudo'),('oracle_coverage','oracle_relabel'),('oracle_coverage','pseudo')):
        contrasts[f'{arm}_minus_{control}']={}
        for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_accuracy','foreground_to_background_rate'):
            deltas=[results[arm][i]['test'][metric]-results[control][i]['test'][metric] for i in range(3)]
            contrasts[f'{arm}_minus_{control}'][metric]={'paired_deltas':deltas,'mean':float(np.mean(deltas)),'std':float(np.std(deltas,ddof=1))}
    atomic_json(ROOT/'training_distribution.json',distributions)
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'results':results,'contrasts':contrasts,
                                  'relabel_audit':json.loads((ROOT/'relabel_audit.json').read_text())})
    print('DONE oracle probe',flush=True)

if __name__=='__main__':main()
