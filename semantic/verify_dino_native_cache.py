"""Check cache equivalence, extraction counts and every paired timing record."""
from __future__ import annotations
import json
from semantic.dino_native_cache_probe import ROOT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from semantic.model import GEOMETRY_KEYS


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    cases=s['cases'];assert len(cases)==12
    expected={('B1',(vid,),seed)for vid in p['video_ids']for seed in range(3)}|{('B2',tuple(p['B2_ids']),seed)for seed in range(3)}
    assert {(c['kind'],tuple(c['video_ids']),c['seed'])for c in cases}==expected
    for c in cases:
        for check in c['checks'].values():
            assert set(check)==set(GEOMETRY_KEYS+('semantic_raw_logits','semantic_logits'))
            assert all(v['bitwise_equal']and v['max_absolute_difference']==v['mean_absolute_difference']==0 for v in check.values())
        assert c['cache_info']['misses']==len(c['video_ids'])and c['cache_info']['entries']==len(c['video_ids'])and c['cache_info']['evictions']==0
        assert c['cache_info']['hits']==14*(c['seed']+1)*len(c['video_ids'])
        assert len(c['timings'])==10
        for i,t in enumerate(c['timings']):
            assert t['repetition']==i and t['order']==(['uncached','cached']if i%2==0 else['cached','uncached'])
            for method in ('uncached','cached'):assert t[method]['seconds']>0 and t[method]['peak_cuda_allocated_bytes']>0
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'counts':{'cases':12,'timed_pairs':120,'timed_forwards':240,'case_groups_with_first_call':4},
        'verification':{'all_same_input_geometry_semantic_checks_bitwise':True,'all_original_time_source_and_subset_cases_present':True,
                        'one_extraction_per_video_head_reload_keeps_tokens':True,'all_alternating_timing_records_present':True,'all_registered_inputs_unchanged':True}})
    print('DONE cache record audit',flush=True)


if __name__=='__main__':main()
