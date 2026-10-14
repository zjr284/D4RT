"""Recompute all geometry deltas from saved arrays, not summary flags."""
import json
import numpy as np
from semantic.dino_decode_precision_probe import ROOT,ARMS,PERMUTATIONS,SIZES
from semantic.model import GEOMETRY_KEYS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def check(a,b,record):
    e=np.abs(a-b)
    assert float(e.max())==record['max_absolute_difference']
    assert np.isclose(e.astype(np.float64).mean(),record['mean_absolute_difference'],rtol=2e-6,atol=1e-9)
    assert bool(np.array_equal(a,b))==record['bitwise_equal']


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol']
    assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert {(c['kind'],tuple(c['video_ids']))for c in s['cases']}=={('B1',(345,)),('B1',(574,)),('B1',(573,)),('B2',(345,574))}
    assert len(s['cases'])==4;count=0
    expected={(a,n,z)for a in ARMS for n in PERMUTATIONS for z in SIZES}
    for c in s['cases']:
        refs={}
        for arm,path in c['reference_paths'].items():
            assert file_hash(path)==c['reference_sha256'][arm]
            with np.load(path)as z:refs[arm]={k:z[k].copy()for k in z.files}
        for k in GEOMETRY_KEYS:assert np.array_equal(refs[ARMS[0]][k],refs[ARMS[1]][k])
        assert len(c['records'])==36 and {(r['arm'],r['permutation'],r['size'])for r in c['records']}==expected
        for r in c['records']:
            assert file_hash(r['path'])==r['sha256']
            with np.load(r['path'])as out:
                assert set(out.files)==set(GEOMETRY_KEYS)
                for k in GEOMETRY_KEYS:
                    assert np.isfinite(out[k]).all()
                    check(out[k],refs[r['arm']][k],r['within_arm'][k])
                    check(out[k],refs[ARMS[0]][k],r['vs_original_full64'][k]);count+=2
        assert len(c['timings'])==36
        assert {(r['arm'],r['size'],r['repetition'])for r in c['timings']}=={(a,z,i)for a in ARMS for z in SIZES for i in range(3)}
        for t in c['timings']:assert t['seconds']>0 and t['peak_cuda_allocated_bytes']>0
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),
        'counts':{'cases':4,'saved_strategy_arrays':144,'reference_arrays':12,'recomputed_field_deltas':count,'timed_passes':144},
        'verification':{'all_saved_arrays_hashed_and_finite':True,'all_deltas_independently_recomputed':True,
        'original_and_fixed_tokens_full64_equal':True,'all_registered_input_hashes_unchanged':True,'all_timings_present':True},
        'limits':['Array verification does not repeat GPU timing','FP32 weights retain BF16 quantization; no geometry ground truth']})
    print('DONE decoder precision independent array audit',count)


if __name__=='__main__':main()
