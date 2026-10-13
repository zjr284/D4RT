"""Independent counts, full predictions, metrics and foundation audit."""
from __future__ import annotations
import json
import subprocess
import numpy as np
import torch
from semantic.dino_448_interface_probe import ROOT, old_predictions, legacy, LEGACY_ARM
from semantic.verify_dino_full_interface import metrics
from semantic.semantic_feature_control import VENDOR, WEIGHTS
from semantic.oracle_probe import file_hash
from semantic.model import GEOMETRY_KEYS
from semantic.run import atomic_json
from tools.load_opend4rt import DEFAULT_VARIANT


def main():
    summary=json.loads((ROOT/'summary.json').read_text());protocol=summary['protocol']
    assert protocol==json.loads((ROOT/'protocol.json').read_text())
    for p,h in protocol['input_sha256'].items():assert file_hash(p)==h
    ids=protocol['video_ids'];cases=summary['cases']
    assert len(ids)==len(set(ids))==40 and len(cases)==126
    assert {(c['video_ids'][0],c['seed'])for c in cases if c['kind']=='B1'}=={(v,s)for v in ids for s in range(3)}
    assert [(c['video_ids'],c['seed'])for c in cases if c['kind']=='B2']==[([549,191],s)for s in range(3)]
    assert [(c['video_ids'],c['seed'])for c in cases if c['kind']=='legacy224']==[([153],s)for s in range(3)]
    for c in cases:
        assert c['queries_per_video']==2048
        assert c['feature_size']==(224 if c['kind']=='legacy224' else 448)
        assert c['background_offset']==(-1.75 if c['kind']=='legacy224'else-1.5)
        for name in ('geometry','changed_time_geometry'):
            assert set(c[name])==set(GEOMETRY_KEYS)
            assert all(v['bitwise_equal'] and v['max_absolute_difference']==v['mean_absolute_difference']==0 for v in c[name].values())
        assert c['semantic_time']['bitwise_equal'] and c['semantic_time']['max_absolute_difference']==0
        for name in ('base_forward_cost','full_interface_cost'):
            assert c[name]['seconds']>0 and c[name]['peak_cuda_allocated_bytes']>0
    for name,h in summary['output_sha256'].items():assert file_hash(ROOT/name)==h
    seed_metrics={};differences={};legacy_differences={}
    for seed in range(3):
        for compat in (False,True):
            name=f"{'legacy_'if compat else''}online_seed{seed}.npz"
            with np.load(ROOT/name)as p:online={k:p[k].copy()for k in p.files}
            old_path=legacy.ROOT/LEGACY_ARM/'logits'/f'dinov2_{seed}_confirmation.npz' if compat else old_predictions(seed)
            with np.load(old_path)as p:old={k:p[k].copy()for k in p.files}
            expected_points=2048 if compat else 81920;offset=-1.75 if compat else -1.5
            assert online['logits'].shape==(expected_points,26) and np.isfinite(online['logits']).all()
            calibrated=online['logits'].copy();calibrated[:,0]+=offset
            assert np.array_equal(calibrated.argmax(-1),online['predictions'])
            changes=0;raw_diffs=[]
            for vid in ([153]if compat else ids):
                a=online['video_ids']==vid;b=old['video_ids']==vid
                assert a.sum()==b.sum()==2048 and np.array_equal(online['labels'][a],old['labels'][b])
                previous=old['logits'][b].copy();previous[:,0]+=offset
                count=int((online['predictions'][a]!=previous.argmax(-1)).sum());changes+=count
                error=np.abs(online['logits'][a]-old['logits'][b]);raw_diffs.append(error)
                kind='legacy224'if compat else'B1'
                case=next(c for c in cases if c['kind']==kind and c['seed']==seed and c['video_ids']==[vid])
                assert case['online_vs_saved_label_changes']==count
                assert case['online_vs_saved_raw_logits']['max_absolute_difference']==float(error.max())
                recomputed=metrics(online['predictions'][a],online['labels'][a])
                assert recomputed==case['per_video'][str(vid)]['online']
            recomputed=metrics(online['predictions'],online['labels'])
            stats={'label_changes':changes,'max_raw_logit_difference':float(np.concatenate(raw_diffs).max()),'bitwise_raw_logits_equal':bool(np.concatenate(raw_diffs).max()==0)}
            if compat:legacy_differences[str(seed)]=stats
            else:
                assert recomputed==summary['metrics'][str(seed)]['online'];seed_metrics[str(seed)]=recomputed;differences[str(seed)]=stats
    fp=protocol['inherited_protocol']['inherited_protocol']['inherited_feature_coverage_protocol']['inherited_feature_protocol']
    foundations={str(WEIGHTS):file_hash(WEIGHTS),str(DEFAULT_VARIANT/'opend4rt.ckpt'):file_hash(DEFAULT_VARIANT/'opend4rt.ckpt'),str(DEFAULT_VARIANT/'model.yaml'):file_hash(DEFAULT_VARIANT/'model.yaml')}
    assert foundations[str(WEIGHTS)]==fp['dinov2_weight_sha256']
    assert foundations[str(DEFAULT_VARIANT/'opend4rt.ckpt')]==fp['d4rt_backbone_sha256']
    assert foundations[str(DEFAULT_VARIANT/'model.yaml')]==fp['d4rt_model_config_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()==fp['dinov2_source_commit']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'foundation_sha256_after_probe':foundations,
        'counts':{'B1_448_cases':120,'B2_448_cases':3,'legacy224_cases':3,'geometry_output_array_comparisons':1512,
                  'B1_448_seed_query_comparisons':245760,'B2_448_seed_query_comparisons':12288,'legacy224_seed_query_comparisons':6144},
        'independent_metrics':seed_metrics,'independent448_vs_previous':differences,'independent224_vs_previous':legacy_differences,
        'verification':{'case_counts_complete':True,'same_input_geometry_and_semantic_time_bitwise_equal':True,
                        'independent_confusion_matrix_metrics_match':True,'all_inputs_foundations_and_outputs_unchanged':True},
        'environment':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'device':torch.cuda.get_device_name(0)}})
    print('DONE independent448 interface audit',differences,'legacy',legacy_differences,flush=True)


if __name__=='__main__':main()
