"""Independently close counts, metrics, predictions and frozen fingerprints."""
from __future__ import annotations
import json
import subprocess
import numpy as np
import torch
from semantic.dino_full_interface_probe import ROOT, ARM, PREVIOUS, old_predictions
from semantic.semantic_feature_control import VENDOR, WEIGHTS
from semantic.oracle_probe import file_hash
from semantic.model import GEOMETRY_KEYS
from semantic.run import atomic_json
from tools.load_opend4rt import DEFAULT_VARIANT


def metrics(pred, truth):
    confusion = np.bincount(truth.astype(np.int64)*26+pred, minlength=26*26).reshape(26, 26)
    support = confusion.sum(1); present = support > 0
    f1 = 2*confusion.diagonal()/np.maximum(confusion.sum(0)+support, 1)
    fg = present.copy(); fg[0] = False
    return {'accuracy': float(confusion.diagonal().sum()/support.sum()),
            'macro_f1_present': float(f1[present].mean()), 'foreground_macro_f1_present': float(f1[fg].mean()),
            'support': {str(i): int(support[i]) for i in np.where(present)[0]},
            'per_class_f1': {str(i): float(f1[i]) for i in np.where(present)[0]}}


def main():
    summary = json.loads((ROOT/'summary.json').read_text()); protocol = summary['protocol']
    assert protocol == json.loads((ROOT/'protocol.json').read_text())
    for path, expected in protocol['input_sha256'].items(): assert file_hash(path) == expected
    ids = protocol['video_ids']; cases = summary['cases']
    assert len(ids) == len(set(ids)) == 40 and len(cases) == 123
    assert {(c['video_ids'][0], c['seed']) for c in cases if c['kind'] == 'B1'} == {(v, s) for v in ids for s in range(3)}
    assert [(c['video_ids'], c['seed']) for c in cases if c['kind'] == 'B2'] == [([153, 52], s) for s in range(3)]
    for c in cases:
        assert c['queries_per_video'] == 2048
        for key in ('geometry', 'changed_time_geometry'):
            assert set(c[key]) == set(GEOMETRY_KEYS)
            assert all(v['bitwise_equal'] and v['max_absolute_difference'] == v['mean_absolute_difference'] == 0 for v in c[key].values())
        assert c['semantic_time']['bitwise_equal'] and c['semantic_time']['max_absolute_difference'] == 0
        for key in ('base_forward_cost', 'full_interface_cost'):
            assert c[key]['seconds'] > 0 and c[key]['peak_cuda_allocated_bytes'] > 0
    seed_metrics = {}; differences = {}
    for s in range(3):
        path = ROOT/f'online_seed{s}.npz'; assert file_hash(path) == summary['output_sha256'][path.name]
        with np.load(path) as p: online = {k: p[k].copy() for k in p.files}
        with np.load(old_predictions(s)) as p: old = {k: p[k].copy() for k in p.files}
        assert online['logits'].shape == (81920, 26) and np.isfinite(online['logits']).all()
        calibrated = online['logits'].copy(); calibrated[:, 0] -= 1.75
        assert np.array_equal(calibrated.argmax(-1), online['predictions'])
        raw_diffs = []; changes = 0
        for vid in ids:
            a = online['video_ids'] == vid; b = old['video_ids'] == vid
            assert a.sum() == b.sum() == 2048 and np.array_equal(online['labels'][a], old['labels'][b])
            original = old['logits'][b].copy(); original[:, 0] -= 1.75
            count = int((online['predictions'][a] != original.argmax(-1)).sum()); changes += count
            error = np.abs(online['logits'][a]-old['logits'][b]); raw_diffs.append(error)
            case = next(c for c in cases if c['kind'] == 'B1' and c['seed'] == s and c['video_ids'] == [vid])
            assert count == case['online_vs_saved_label_changes']
            assert float(error.max()) == case['online_vs_saved_raw_logits']['max_absolute_difference']
            assert metrics(online['predictions'][a], online['labels'][a]) == case['per_video'][str(vid)]['online']
        recomputed = metrics(online['predictions'], online['labels']); assert recomputed == summary['metrics'][str(s)]['online']
        seed_metrics[str(s)] = recomputed
        differences[str(s)] = {'label_changes': changes, 'max_raw_logit_difference': float(np.concatenate(raw_diffs).max())}
    fp = protocol['inherited_protocol']['inherited_feature_coverage_protocol']['inherited_feature_protocol']
    foundations = {str(WEIGHTS): file_hash(WEIGHTS), str(DEFAULT_VARIANT/'opend4rt.ckpt'): file_hash(DEFAULT_VARIANT/'opend4rt.ckpt'),
                   str(DEFAULT_VARIANT/'model.yaml'): file_hash(DEFAULT_VARIANT/'model.yaml')}
    assert foundations[str(WEIGHTS)] == fp['dinov2_weight_sha256']
    assert foundations[str(DEFAULT_VARIANT/'opend4rt.ckpt')] == fp['d4rt_backbone_sha256']
    assert foundations[str(DEFAULT_VARIANT/'model.yaml')] == fp['d4rt_model_config_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip() == fp['dinov2_source_commit']
    atomic_json(ROOT/'verification.json', {'summary_sha256': file_hash(ROOT/'summary.json'), 'foundation_sha256_after_probe': foundations,
        'counts': {'B1_cases': 120, 'B2_cases': 3, 'geometry_output_array_comparisons': 1476,
                   'B1_seed_query_comparisons': 245760, 'B2_seed_query_comparisons': 12288},
        'independent_metrics': seed_metrics, 'independent_online_vs_previous': differences,
        'verification': {'case_counts_complete': True, 'same_input_geometry_and_semantic_time_bitwise_equal': True,
                         'independent_confusion_matrix_metrics_match': True, 'all_inputs_foundations_and_outputs_unchanged': True},
        'environment': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'device': torch.cuda.get_device_name(0)}})
    print('DONE independent full-interface audit', differences)


if __name__ == '__main__': main()
