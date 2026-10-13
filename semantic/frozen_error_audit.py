"""Descriptive decomposition of frozen confirmation outputs, without tuning."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from semantic.background_calibration import predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/frozen_error_audit')
FRESH=Path('semantic_artifacts/fresh_video_confirmation')
CACHES={'source_gt':Path('semantic_artifacts/features_ovis/sam/train'),
        'positive_only':Path('semantic_artifacts/features_ovis/sam/train'),
        'positive_background':Path('semantic_artifacts/background_ablation/features/sam/train')}
PROTOCOL={'version':1,'analysis':'descriptive audit of ninth-round frozen confirmation outputs',
          'model_updates':0,'threshold_updates':0,'seeds':[0,1,2],
          'errors':['correct foreground','foreground predicted background','foreground predicted wrong foreground',
                    'correct background','background predicted foreground'],
          'coverage_bins':'fixed independent training-video counts: 1, 2-3, >=4',
          'category_analysis':'all 22 foreground classes present on fresh confirmation; no class selection',
          'frequent_confusions':'ten largest off-diagonal foreground pairs, counts averaged over three seeds',
          'not_a_new_confirmation':True}


def confusion_counts(labels,predicted,classes=26):
    y=np.asarray(labels);p=np.asarray(predicted)
    if y.ndim!=1 or p.shape!=y.shape or not len(y):raise ValueError('Nonempty paired labels required')
    if np.any((y<0)|(y>=classes)|(p<0)|(p>=classes)):raise ValueError('Labels outside class range')
    m=np.zeros((classes,classes),dtype=np.int64);np.add.at(m,(y,p),1)
    return m


def decompose(m):
    fg=int(m[1:,:].sum());bg=int(m[0,:].sum())
    correct=int(np.trace(m[1:,1:]));to_bg=int(m[1:,0].sum());wrong=int(m[1:,1:].sum())-correct
    if correct+to_bg+wrong!=fg:raise AssertionError('Foreground partition invalid')
    return {'foreground_support':fg,'background_support':bg,
            'foreground_correct_count':correct,'foreground_to_background_count':to_bg,'foreground_wrong_class_count':wrong,
            'background_correct_count':int(m[0,0]),'background_to_foreground_count':int(m[0,1:].sum()),
            'foreground_correct_rate':correct/fg if fg else None,'foreground_to_background_rate':to_bg/fg if fg else None,
            'foreground_wrong_class_rate':wrong/fg if fg else None,
            'background_correct_rate':int(m[0,0])/bg if bg else None,
            'background_to_foreground_rate':int(m[0,1:].sum())/bg if bg else None}


def coverage(arm):
    counts=np.zeros(26,dtype=np.int64);videos={c:set() for c in range(1,26)};unique={c:set() for c in range(1,26)}
    times=np.zeros((8,26),dtype=np.int64)
    for path in sorted(CACHES[arm].glob('*.npz')):
        with np.load(path) as p:
            labels=p['labels'];xy=p['xy'];vid=int(p['video_id'])
        if arm=='source_gt':mask=xy[:,2]==0;labels=labels[mask];xy=xy[mask]
        counts+=np.bincount(labels,minlength=26);np.add.at(times,(xy[:,2],labels),1)
        for c in np.unique(labels[labels>0]):
            c=int(c);videos[c].add(vid);unique[c].update((vid,*map(int,row)) for row in xy[labels==c])
    return {'time_class_query_counts':times.tolist(),'per_class':{str(c):{'queries':int(counts[c]),
            'distinct_video_xy_time_queries':len(unique[c]),'independent_videos':len(videos[c]),
            'video_ids':sorted(videos[c])} for c in range(1,26)}}


def per_class(m,c):
    support=float(m[c].sum());tp=float(m[c,c]);fp=float(m[:,c].sum()-tp);fn=support-tp
    return {'support':support,'f1':2*tp/max(2*tp+fp+fn,1),
            'recall':tp/support if support else None,'to_background_rate':float(m[c,0])/support if support else None,
            'wrong_foreground_rate':float(m[c,1:].sum()-tp)/support if support else None}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((FRESH/'summary.json').read_text())
    packet_hashes={f'{arm}_{seed}':file_hash(FRESH/'logits'/f'{arm}_seed{seed}.npz') for arm in CACHES for seed in PROTOCOL['seeds']}
    train_hashes={}
    for arm,folder in CACHES.items():
        h=hashlib.sha256()
        for p in sorted(folder.glob('*.npz')):h.update(p.name.encode());h.update(bytes.fromhex(file_hash(p)))
        train_hashes[arm]=h.hexdigest()
    protocol=dict(PROTOCOL,fresh_summary_sha256=file_hash(FRESH/'summary.json'),logit_sha256=packet_hashes,training_cache_sha256=train_hashes)
    path=ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Audit inputs changed')
    atomic_json(path,protocol)
    categories=[int(c) for c in previous['results']['source_gt']['runs'][0]['calibrated']['support'] if c!='0']
    results={};reference=None
    for arm in CACHES:
        cov=coverage(arm);runs=[];matrices=[]
        for seed in PROTOCOL['seeds']:
            with np.load(FRESH/'logits'/f'{arm}_seed{seed}.npz') as p:
                logits=p['logits'];labels=p['labels'];vids=p['video_ids'];times=p['times']
            meta=(labels,vids,times)
            if reference is None:reference=tuple(a.copy() for a in meta)
            elif not all(np.array_equal(a,b) for a,b in zip(reference,meta)):raise ValueError('Evaluation points differ')
            predicted=predictions(logits,previous['results'][arm]['offset']);m=confusion_counts(labels,predicted)
            expected=np.array(previous['results'][arm]['runs'][seed]['calibrated']['confusion_matrix'])
            if not np.array_equal(m,expected):raise ValueError('Ninth-round confusion not reproduced')
            matrices.append(m);runs.append({'seed':seed,'decomposition':decompose(m),
                'per_video':{str(v):decompose(confusion_counts(labels[vids==v],predicted[vids==v])) for v in np.unique(vids)}})
        mean_matrix=np.mean(matrices,axis=0)
        pairs=[]
        for c in range(1,26):
            for predicted in range(1,26):
                if c!=predicted and mean_matrix[c,predicted]>0:
                    pairs.append({'truth':c,'predicted':predicted,'mean_count':float(mean_matrix[c,predicted])})
        pairs.sort(key=lambda p:(-p['mean_count'],p['truth'],p['predicted']))
        bins={}
        for name,lo,hi in [('1_video',1,1),('2_3_videos',2,3),('at_least_4_videos',4,60)]:
            ids=[c for c in categories if lo<=cov['per_class'][str(c)]['independent_videos']<=hi]
            bins[name]={'class_ids':ids,'macro_f1':float(np.mean([per_class(mean_matrix,c)['f1'] for c in ids])) if ids else None,
                        'query_weighted_foreground_recall':float(np.trace(mean_matrix[ids][:,ids])/mean_matrix[ids].sum()) if ids else None}
            # Diagonal of indexed submatrix retains only same-class successes.
        results[arm]={'runs':runs,'mean_decomposition':{k:float(np.mean([r['decomposition'][k] for r in runs])) for k in runs[0]['decomposition']},
                    'coverage':cov,'per_class':{str(c):per_class(mean_matrix,c) for c in categories},
                    'coverage_bins':bins,'top_foreground_confusions':pairs[:10],
                    'mean_confusion_matrix':mean_matrix.tolist()}
    a=results['positive_only']['coverage']['per_class'];b=results['positive_background']['coverage']['per_class']
    if a!=b:raise ValueError('Adding background unexpectedly changed positive coverage')
    differences={}
    for c in categories:
        differences[str(c)]={'f1_sam_background_minus_gt':results['positive_background']['per_class'][str(c)]['f1']-results['source_gt']['per_class'][str(c)]['f1'],
                            'video_coverage_gt':results['source_gt']['coverage']['per_class'][str(c)]['independent_videos'],
                            'video_coverage_sam':results['positive_background']['coverage']['per_class'][str(c)]['independent_videos']}
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'results':results,'class_differences':differences,
                'verification':{'all_nine_confusion_matrices_match_frozen_confirmation':True,'all_query_metadata_identical':True,
                'sam_positive_coverage_unchanged_by_background_addition':True,'no_model_or_threshold_updates':True}})
    print('DONE frozen error audit',flush=True)

if __name__=='__main__':main()
