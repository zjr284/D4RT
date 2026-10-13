"""Audit saved interface cases and recheck frozen foundations after inference."""
from __future__ import annotations
import json
import subprocess
import torch
from semantic.dino_interface_probe import ROOT
from semantic.semantic_feature_control import VENDOR, WEIGHTS
from semantic.model import GEOMETRY_KEYS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from tools.load_opend4rt import DEFAULT_VARIANT


def main():
    summary = json.loads((ROOT/'summary.json').read_text())
    protocol = summary['protocol']; assert protocol == json.loads((ROOT/'protocol.json').read_text())
    for path, expected in protocol['input_sha256'].items(): assert file_hash(path) == expected
    expected_cases = {(v, s) for v in (153, 52, 395) for s in (0, 1, 2)}
    assert {(c['video_id'], c['seed']) for c in summary['cases']} == expected_cases and len(summary['cases']) == 9
    for case in summary['cases']:
        assert case['cache_query_indices'] == list(range(0, 2048, 32)) and case['query_count'] == 72
        for field in ('geometry', 'changed_time_geometry'):
            assert set(case[field]) == set(GEOMETRY_KEYS)
            assert all(v['bitwise_equal'] and v['max_absolute_difference'] == 0 and v['mean_absolute_difference'] == 0 for v in case[field].values())
        assert case['semantic_time']['bitwise_equal'] and case['semantic_time']['max_absolute_difference'] == 0
        assert case['semantic_head_reload_bitwise_equal']
        for name in ('base_forward_cost', 'full_interface_cost'):
            assert case[name]['seconds'] > 0 and case[name]['peak_cuda_allocated_bytes'] > 0
    fp = protocol['inherited_protocol']['inherited_feature_coverage_protocol']['inherited_feature_protocol']
    foundations = {str(WEIGHTS): file_hash(WEIGHTS), str(DEFAULT_VARIANT/'opend4rt.ckpt'): file_hash(DEFAULT_VARIANT/'opend4rt.ckpt'),
                   str(DEFAULT_VARIANT/'model.yaml'): file_hash(DEFAULT_VARIANT/'model.yaml')}
    assert foundations[str(WEIGHTS)] == fp['dinov2_weight_sha256']
    assert foundations[str(DEFAULT_VARIANT/'opend4rt.ckpt')] == fp['d4rt_backbone_sha256']
    assert foundations[str(DEFAULT_VARIANT/'model.yaml')] == fp['d4rt_model_config_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip() == fp['dinov2_source_commit']
    atomic_json(ROOT/'verification.json', {'summary_sha256': file_hash(ROOT/'summary.json'), 'foundation_sha256_after_probe': foundations,
        'verification': {'nine_cases_three_seeds_all_present': True, 'geometry_time_reload_checks_and_counts_close': True,
                         'all_recorded_inputs_and_foundations_unchanged': True},
        'environment': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                        'device': torch.cuda.get_device_name(0), 'device_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory}})
    print('DONE interface record/foundation audit')


if __name__ == '__main__': main()
