"""Whole-batch reuse count, exact equivalence and paired timing closure."""
from __future__ import annotations
import json
from semantic.dino_geometry_cache_probe import ROOT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from semantic.model import GEOMETRY_KEYS


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    cases=s['cases'];assert len(cases)==12
    expected={('B1',(v,),seed)for v in p['video_ids']for seed in range(3)}|{('B2',tuple(p['B2_ids']),seed)for seed in range(3)}
    assert {(c['kind'],tuple(c['video_ids']),c['seed'])for c in cases}==expected
    for c in cases:
        assert set(c['checks'])=={'original','target_camera','source','subset256','aspect'}
        for check in c['checks'].values():
            assert set(check)==set(GEOMETRY_KEYS+('semantic_raw_logits','semantic_logits'))
            assert all(v['bitwise_equal']and v['max_absolute_difference']==v['mean_absolute_difference']==0 for v in check.values())
        info=c['cache_info'];assert info['misses']==len(c['video_ids'])and info['entries']==len(c['video_ids'])
        assert info['geometry_misses']==2 and info['geometry_entries']==2 and info['geometry_evictions']==0
        assert info['geometry_hits']==14+15*c['seed']and info['hits']==15*(c['seed']+1)*len(c['video_ids'])
        assert len(c['timings'])==10
        for i,t in enumerate(c['timings']):
            assert t['repetition']==i and t['order']==(['semantic_only','combined']if i%2==0 else['combined','semantic_only'])
            for name in ('semantic_only','combined'):assert t[name]['seconds']>0 and t[name]['peak_cuda_allocated_bytes']>0
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'counts':{'cases':12,'paired_timings':120,'timed_forwards':240,'query_aspect_variants_per_case':5},
        'verification':{'all_same_input_geometric_and_semantic_variants_bitwise_equal':True,'one_semantic_extraction_per_video':True,
                        'two_geometry_encodes_for_two_aspect_keys_per_case':True,'all_alternating_timings_present':True,'all_registered_inputs_unchanged':True}})
    print('DONE whole geometry cache record audit',flush=True)


if __name__=='__main__':main()
