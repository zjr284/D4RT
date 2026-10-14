"""Fixed point-set streaming, exact same-block references and cross-block deltas."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from semantic.dino_native_interface import load_native_selected_adapter
from semantic.dino_geometry_cache import CachedGeometryNativeDinoSemanticD4RT
from semantic.dino_native_rgb_control import ROOT as PREVIOUS,checkpoint,feature_path
from semantic.dino_interface import cuda_bf16
from semantic.dino_interface_probe import queries,measured,delta
from semantic.dino_native_cache_probe import compare
from semantic.model import GEOMETRY_KEYS
from semantic.ytvis import load_video
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/dino_query_stream_probe')
SIZES=(1,8,32,64)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed stream probe exists; audit rather than rerun')
    previous=json.loads((PREVIOUS/'summary.json').read_text());verified=json.loads((PREVIOUS/'verification.json').read_text());assert file_hash(PREVIOUS/'summary.json')==verified['summary_sha256']
    clips=previous['manifest']['confirmation'][:3];indices=np.arange(0,2048,32)
    paths=[PREVIOUS/'summary.json',PREVIOUS/'verification.json']+[checkpoint('native_gt',s)for s in range(3)]
    paths+=[feature_path('native_gt','confirmation',c['video_id'])for c in clips]+[Path(p)for c in clips for p in c['frames']]
    paths+=[Path(p)for p in ('semantic/dino_query_stream_probe.py','semantic/OVIS_DINO_QUERY_STREAM_PROTOCOL.md','semantic/dino_geometry_cache.py','semantic/dino_native_cache.py')]
    hashes={str(p):file_hash(p)for p in paths}
    protocol={'round':27,'analysis':'frozen point-set batching cost and numerical diagnostic, not new generalization',
        'video_ids':[c['video_id']for c in clips],'B2_ids':[c['video_id']for c in clips[:2]],'seeds':[0,1,2],'indices':indices.tolist(),
        'sizes':list(SIZES),'timing_repetitions':3,'strategy_order':'rotate sizes left0/1/2','background_offset':-1.5,'input_sha256':hashes}
    path=ROOT/'protocol.json'
    if path.exists()and json.loads(path.read_text())!=protocol:raise ValueError('Streaming protocol changed')
    atomic_json(path,protocol);ref=load_native_selected_adapter(0)
    cached=CachedGeometryNativeDinoSemanticD4RT(ref.base,ref.dino,ref.background_offset).cuda();batches=[]
    for c in clips:
        with np.load(feature_path('native_gt','confirmation',c['video_id']))as p:data={k:p[k].copy()for k in p.files}
        batches.append({'video':load_video(c)[None].cuda().to(torch.bfloat16),'source_frames':[c['frames']],'query':queries(data,indices,extra=False)})
    paired={'video':torch.cat([b['video']for b in batches[:2]]),'source_frames':[b['source_frames'][0]for b in batches[:2]],
            'query':{k:torch.cat([b['query'][k]for b in batches[:2]])for k in batches[0]['query']}}
    cases=[]
    for kind,b,ids in [('B1',b,[c['video_id']])for b,c in zip(batches,clips)]+[('B2',paired,protocol['B2_ids'])]:
        cached.clear_cache()
        with torch.inference_mode():
            native=ref.source_rgb(b);features=ref.native_features(native,b['query'])
            with cuda_bf16(b['video'].device):memory=ref.base.encode_video(b['video'],aspect_ratio=b.get('aspect_ratio'))
            for seed in range(3):
                sha=verified['checkpoint_sha256'][f'native_gt_{seed}'];ref.load_head(checkpoint('native_gt',seed),sha);cached.load_head(checkpoint('native_gt',seed),sha)
                reference=ref(b);compare(cached(b),reference);strategies={}
                def stream(size,check=False):
                    values=[]
                    for start in range(0,64,size):
                        q={k:v[:,start:start+size]for k,v in b['query'].items()};part={**b,'query':q};actual=cached(part)
                        if check:
                            expected_features=features[:,start:start+size].contiguous();assert torch.equal(cached.cached_features(part),expected_features)
                            with cuda_bf16(b['video'].device):expected=dict(ref.base.decode_queries(b['video'],q,memory))
                            expected.update(ref.semantic_outputs(expected_features));compare(actual,expected)
                        values.append(actual)
                    return {k:torch.cat([v[k]for v in values],dim=1)for k in values[0]}
                for size in SIZES:
                    output=stream(size,check=True)
                    strategies[str(size)]={'calls_per_pass':64//size,'cross_block_geometry':{k:delta(output[k],reference[k])for k in GEOMETRY_KEYS},
                        'cross_block_raw_logits':delta(output['semantic_raw_logits'],reference['semantic_raw_logits']),
                        'cross_block_label_changes':int((output['semantic_labels']!=reference['semantic_labels']).sum()),'same_block_original_reference_bitwise':True,'timings':[]}
                orders=[]
                for repetition in range(3):
                    order=SIZES[repetition:]+SIZES[:repetition];orders.append(list(order))
                    for size in order:
                        output,cost=measured(lambda:stream(size));strategies[str(size)]['timings'].append({'repetition':repetition,**cost})
                        assert int((output['semantic_labels']!=reference['semantic_labels']).sum())==strategies[str(size)]['cross_block_label_changes']
                assert cached.geometry_misses==1 and cached.misses==len(ids)
                case={'kind':kind,'video_ids':ids,'seed':seed,'strategies':strategies,'orders':orders,'cache_info':cached.cache_info()}
                cases.append(case);atomic_json(ROOT/'progress.json',{'protocol_sha256':file_hash(path),'cases':cases})
                print('stream verified',kind,ids,seed,'seconds',[float(np.median([t['seconds']for t in strategies[str(n)]['timings']]))for n in SIZES],
                      'labels',[strategies[str(n)]['cross_block_label_changes']for n in SIZES],flush=True)
    for p,h in hashes.items():assert file_hash(p)==h
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'cases':cases,'verification':{'same_block_features_and_original_outputs_bitwise_equal':True,
        'all_preregistered_strategies_and_times_recorded':True,'all_inputs_weights_and_model_choice_unchanged':True,'no_training_calibration_or_new_videos':True},
        'limits':['cross-block differences are numerical output changes, not truth geometry accuracy','warm repeated64point set, not cold or arbitrary production workload']})
    print('DONE fixed query streaming probe',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
