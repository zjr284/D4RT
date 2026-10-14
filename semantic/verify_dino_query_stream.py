"""Audit fixed streaming cases, numerical deltas and timing completeness."""
import json
import math
from semantic.dino_query_stream_probe import ROOT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from semantic.model import GEOMETRY_KEYS


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol']
    assert p==json.loads((ROOT/'protocol.json').read_text())
    assert p['indices']==list(range(0,2048,32)) and p['sizes']==[1,8,32,64]
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    expected={('B1',(v,),seed)for v in p['video_ids']for seed in range(3)}|{('B2',tuple(p['B2_ids']),seed)for seed in range(3)}
    assert len(s['cases'])==12 and {(c['kind'],tuple(c['video_ids']),c['seed'])for c in s['cases']}==expected
    for c in s['cases']:
        assert c['orders']==[[1,8,32,64],[8,32,64,1],[32,64,1,8]]
        assert set(c['strategies'])=={'1','8','32','64'}
        info=c['cache_info'];n=c['seed']+1
        assert info['misses']==info['entries']==len(c['video_ids'])
        assert info['geometry_misses']==info['geometry_entries']==1
        assert info['geometry_hits']==301*n-1 and info['hits']==(376*n-1)*len(c['video_ids'])
        assert info['evictions']==info['geometry_evictions']==0
        for size,a in c['strategies'].items():
            assert a['calls_per_pass']==64//int(size) and a['same_block_original_reference_bitwise']
            assert set(a['cross_block_geometry'])==set(GEOMETRY_KEYS)
            for d in list(a['cross_block_geometry'].values())+[a['cross_block_raw_logits']]:
                assert all(math.isfinite(d[k])and d[k]>=0 for k in ['max_absolute_difference','mean_absolute_difference'])
                assert d['mean_absolute_difference']<=d['max_absolute_difference']
                if size=='64':assert d['bitwise_equal'] and d['max_absolute_difference']==0
            assert a['cross_block_label_changes']==0
            assert len(a['timings'])==3 and [t['repetition']for t in a['timings']]==[0,1,2]
            for t in a['timings']:assert t['seconds']>0 and t['peak_cuda_allocated_bytes']>0
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),
        'counts':{'cases':12,'strategies':48,'timed_passes':144,'timed_model_calls':2700,'same_block_reference_comparisons':900},
        'verification':{'all_cases_strategies_orders_and_timings_present':True,'same_block_reference_bitwise_checks_recorded':True,
        'all_cross_block_differences_retained':True,'zero_semantic_label_changes':True,'cache_counts_match_calls':True,'all_registered_inputs_unchanged':True},
        'limit':'Independent record audit; no second timing run or ground-truth geometry evaluation.'})
    print('DONE query streaming record audit')


if __name__=='__main__':main()
