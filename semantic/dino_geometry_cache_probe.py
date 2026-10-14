"""Whole-batch geometry encoding reuse: exact original decoder and paired costs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from semantic.dino_native_cache import CachedNativeDinoSemanticD4RT
from semantic.dino_geometry_cache import CachedGeometryNativeDinoSemanticD4RT
from semantic.dino_native_interface import load_native_selected_adapter
from semantic.dino_native_rgb_control import ROOT as PREVIOUS,checkpoint,feature_path
from semantic.dino_interface_probe import queries,measured,delta
from semantic.model import GEOMETRY_KEYS
from semantic.ytvis import load_video
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/dino_geometry_cache_probe')


def compare(a,b):
    result={k:delta(a[k],b[k])for k in GEOMETRY_KEYS+('semantic_raw_logits','semantic_logits')}
    assert all(v['bitwise_equal']for v in result.values())
    assert torch.equal(a['semantic_labels'],b['semantic_labels'])
    return result


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed cache probe exists; audit not rerun')
    previous=json.loads((PREVIOUS/'summary.json').read_text());verified=json.loads((PREVIOUS/'verification.json').read_text())
    assert file_hash(PREVIOUS/'summary.json')==verified['summary_sha256']and previous['validation_selected_arm']=='native_gt'
    clips=previous['manifest']['confirmation'][:3]
    paths=[PREVIOUS/'summary.json',PREVIOUS/'verification.json']+[checkpoint('native_gt',s)for s in range(3)]
    paths+=[feature_path('native_gt','confirmation',c['video_id'])for c in clips]
    paths+=[Path(p)for c in clips for p in c['frames']]
    paths+=[Path(p)for p in ('semantic/dino_geometry_cache.py','semantic/dino_geometry_cache_probe.py','semantic/OVIS_DINO_GEOMETRY_CACHE_PROTOCOL.md','tests/test_dino_geometry_cache.py','semantic/dino_native_cache.py')]
    hashes={str(p):file_hash(p)for p in paths}
    protocol={'round':26,'analysis':'frozen cache correctness and paired latency, no new model evaluation or generalization claim',
        'video_ids':[c['video_id']for c in clips],'B2_ids':[c['video_id']for c in clips[:2]],'seeds':[0,1,2],'queries_per_video':2048,
        'paired_repetitions':10,'geometry_cache_batches':2,'geometry_key':'whole video shape/dtype/device/bytes, aspect_ratio and parameter version','max_cached_videos':2,'head':'native_gt','background_offset':-1.5,'input_sha256':hashes,
        'additional_query_checks':'target+1 camera+3 mod8; source+1 mod8; first256query subset; aspect_ratio linspace1.1-1.5','input_content_checked_every_call':True}
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=protocol:raise ValueError('Cache protocol changed')
    atomic_json(path,protocol)
    ref=load_native_selected_adapter(0);semantic_only=CachedNativeDinoSemanticD4RT(ref.base,ref.dino,ref.background_offset,max_cached_videos=2).cuda()
    cached=CachedGeometryNativeDinoSemanticD4RT(ref.base,ref.dino,ref.background_offset,max_cached_videos=2,max_geometry_batches=2).cuda()
    batches=[]
    for c in clips:
        with np.load(feature_path('native_gt','confirmation',c['video_id']))as p:data={k:p[k].copy()for k in p.files}
        q=queries(data,np.arange(2048),extra=False);video=load_video(c)[None].cuda().to(torch.bfloat16)
        batches.append({'video':video,'source_frames':[c['frames']],'query':q})
    paired={'video':torch.cat([b['video']for b in batches[:2]]),'source_frames':[b['source_frames'][0]for b in batches[:2]],
            'query':{k:torch.cat([b['query'][k]for b in batches[:2]])for k in batches[0]['query']}}
    cases=[]
    for kind,b,ids in [('B1',b,[c['video_id']])for b,c in zip(batches,clips)]+[('B2',paired,protocol['B2_ids'])]:
        cached.clear_cache();semantic_only.clear_cache()
        for seed in range(3):
            head=checkpoint('native_gt',seed);sha=verified['checkpoint_sha256'][f'native_gt_{seed}'];ref.load_head(head,sha);cached.load_head(head,sha);semantic_only.load_head(head,sha)
            with torch.inference_mode():
                if seed==0:
                    first,first_cost=measured(lambda:cached(b));reference=ref(b);first_delta=compare(first,reference)
                    assert cached.misses==len(ids)
                base=ref(b);warm=cached(b);checks={'original':compare(warm,base)}
                before=cached.cache_info().copy()
                for name in ('target_camera','source','subset256','aspect'):
                    q={k:v.clone()for k,v in b['query'].items()}
                    if name=='target_camera':q['t_tgt']=(q['t_tgt']+1)%8;q['t_cam']=(q['t_cam']+3)%8
                    elif name=='source':q['t_src']=(q['t_src']+1)%8
                    elif name=='subset256':q={k:v[:,:256]for k,v in q.items()}
                    altered={**b,'query':q}
                    if name=='aspect':altered['aspect_ratio']=torch.linspace(1.1,1.5,len(ids),device=b['video'].device)
                    expected=ref(altered);actual=cached(altered);checks[name]=compare(actual,expected)
                    if name=='target_camera':assert torch.equal(actual['semantic_logits'],warm['semantic_logits'])
                assert cached.misses==before['misses']
                assert cached.geometry_misses==2
                compare(semantic_only(b),base)
                timings=[]
                for i in range(10):
                    order=('semantic_only','combined')if i%2==0 else('combined','semantic_only');cost={}
                    for name in order:
                        value,cost[name]=measured(lambda:semantic_only(b)if name=='semantic_only'else cached(b))
                        compare(value,base)
                    timings.append({'repetition':i,'order':list(order),**cost})
                assert cached.misses==len(ids)and cached.cache_info()['entries']==len(ids)
                case={'kind':kind,'video_ids':ids,'seed':seed,'first_cache_cost':first_cost,'first_cache_equality':first_delta,
                    'checks':checks,'timings':timings,'cache_info':cached.cache_info()}
                cases.append(case);atomic_json(ROOT/'progress.json',{'protocol_sha256':file_hash(path),'cases':cases})
                print('cache verified',kind,ids,seed,'uncached/cached median',float(np.median([t['semantic_only']['seconds']for t in timings])),float(np.median([t['combined']['seconds']for t in timings])),cached.cache_info(),flush=True)
    for p,h in hashes.items():assert file_hash(p)==h
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'cases':cases,'verification':{'all_original_changed_and_subset_outputs_bitwise_equal':True,
        'heads_reloaded_without_reusing_scores':True,'one_token_extraction_per_video_per_case':True,'two_geometry_encodes_per_case_for_two_aspect_keys':True,'all_inputs_and_weights_unchanged':True,
        'no_training_calibration_or_new_videos':True},'limits':['warm repeated source reuse workload, not cold disk or new video throughput','real B2 only first2 fixed videos','frozen backbone version tracking does not promise detection of arbitrary unsafe .data manipulation']})
    print('DONE whole geometry cache probe',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
