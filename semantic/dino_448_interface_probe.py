"""Round19: frozen448 interface, full queries and legacy224 compatibility."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from semantic.dino_interface import load_resolution_selected_adapter, cuda_bf16
from semantic.dino_interface_probe import queries, measured, delta
from semantic.dino_resolution_control import ROOT as PREVIOUS, checkpoint, feature_path
from semantic import dino_sam_control as legacy
from semantic.model import GEOMETRY_KEYS
from semantic.ytvis import load_video
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json, classification_metrics

ROOT = Path('semantic_artifacts/dino_448_interface_probe')
ARM = 'resolution448'
LEGACY_ARM = 'coverage80_sam_background'


def old_predictions(seed):
    return PREVIOUS/ARM/'logits'/f'dinov2_{seed}_confirmation.npz'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'summary.json').exists():
        raise RuntimeError('Completed experiment exists; verify it rather than repeat inference')
    previous = json.loads((PREVIOUS/'summary.json').read_text())
    verified = json.loads((PREVIOUS/'verification.json').read_text())
    assert file_hash(PREVIOUS/'summary.json') == verified['summary_sha256']
    assert previous['validation_selected_arm'] == ARM and previous['results'][ARM]['offset'] == -1.5
    clips = previous['manifest']['confirmation']; assert len(clips) == 40
    assert [c['video_id'] for c in clips[:2]] == [549, 191]
    inputs = [PREVIOUS/'summary.json', PREVIOUS/'verification.json', PREVIOUS/'frozen_selection.json']
    assert file_hash(PREVIOUS/'frozen_selection.json') == previous['frozen_selection_sha256']
    legacy_summary = json.loads((legacy.ROOT/'summary.json').read_text())
    legacy_verify = json.loads((legacy.ROOT/'verification.json').read_text())
    assert file_hash(legacy.ROOT/'summary.json') == legacy_verify['summary_sha256']
    legacy_clip = legacy_summary['manifest']['confirmation'][0]; assert legacy_clip['video_id'] == 153
    legacy_paths = [legacy.ROOT/LEGACY_ARM/'logits'/f'dinov2_{s}_confirmation.npz' for s in range(3)]
    inputs += [legacy.ROOT/'summary.json', legacy.ROOT/'verification.json']+legacy_paths
    inputs += [legacy.checkpoint(LEGACY_ARM, s) for s in range(3)]
    inputs += [legacy.path_resolver('dinov2', 'confirmation', 153)]+[Path(f) for f in legacy_clip['frames']]
    inputs += [checkpoint(ARM, s) for s in range(3)] + [old_predictions(s) for s in range(3)]
    inputs += [feature_path(ARM, 'confirmation', c['video_id']) for c in clips]
    inputs += [Path(frame) for c in clips for frame in c['frames']]
    inputs += [Path(p) for p in ('semantic/dino_interface.py', 'semantic/dino_448_interface_probe.py',
                                'semantic/OVIS_DINO_448_INTERFACE_PROTOCOL.md', 'tests/test_dino_interface.py')]
    hashes = {str(p): file_hash(p) for p in inputs}
    protocol = {'round': 19, 'analysis': 'descriptive frozen interface audit, not new generalization confirmation',
        'video_ids': [c['video_id'] for c in clips], 'seeds': [0, 1, 2], 'queries_per_video': 2048,
        'real_batch_sizes': [1, 2], 'real_b2_ids': [549, 191], 'background_offset': -1.5, 'feature_size': 448, 'legacy224_ids': [153],
        'geometry_rgb': 'BF16 original input', 'semantic_rgb': 'float32, exact quantization correspondence',
        'time_change': 'target+1, camera+3 mod8', 'training_updates': 0, 'calibration_updates': 0,
        'input_sha256': hashes, 'selected_arm': ARM, 'inherited_protocol': previous['protocol']}
    path = ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Preregistered inputs changed')
    atomic_json(path, protocol)
    old = []
    for s in range(3):
        with np.load(old_predictions(s)) as p: old.append({k: p[k].copy() for k in p.files})
    legacy_old = []
    for legacy_path in legacy_paths:
        with np.load(legacy_path) as p: legacy_old.append({k: p[k].copy() for k in p.files})
    adapter = load_resolution_selected_adapter(0); assert adapter.feature_size == 448 and adapter.background_offset == -1.5
    cases = []; outputs = {s: [] for s in range(3)}; legacy_outputs = {}
    first_pair = []

    def check_batch(group, caches, batch, kind):
        compat = kind == 'legacy224'
        adapter.feature_size = 224 if compat else 448
        adapter.background_offset = -1.75 if compat else -1.5
        reference_outputs = legacy_old if compat else old
        reference_summary = legacy_summary if compat else previous
        reference_arm = LEGACY_ARM if compat else ARM
        changed = {k: v.clone() for k, v in batch['query'].items()}
        changed['t_tgt'] = (changed['t_tgt']+1)%8; changed['t_cam'] = (changed['t_cam']+3)%8
        changed_batch = {**batch, 'query': changed}
        saved = {k: v.clone() for k, v in batch['query'].items()}
        saved_rgb = batch['semantic_video'].clone(); saved_geo_rgb = batch['video'].clone()
        with torch.inference_mode():
            def original(b):
                with cuda_bf16(batch['video'].device): return adapter.base(b)
            ref, base_cost = measured(lambda: original(batch)); ref_changed = original(changed_batch)
            features = adapter.semantic_features(batch['semantic_video'], batch['query'])
            cached = torch.stack([torch.from_numpy(c['decoded']).float() for c in caches]).cuda()
            feature_error = delta(features, cached)
            for seed in range(3):
                head = legacy.checkpoint(LEGACY_ARM, seed) if compat else checkpoint(ARM, seed)
                head_hash = (legacy_verify if compat else verified)['checkpoint_sha256'][f'{reference_arm}_{seed}']
                adapter.load_head(head, head_hash)
                result, cost = measured(lambda: adapter(batch)); altered = adapter(changed_batch)
                geom = {k: delta(result[k], ref[k]) for k in GEOMETRY_KEYS}
                geom_changed = {k: delta(altered[k], ref_changed[k]) for k in GEOMETRY_KEYS}
                time_check = delta(result['semantic_logits'], altered['semantic_logits'])
                assert all(v['bitwise_equal'] for v in geom.values())
                assert all(v['bitwise_equal'] for v in geom_changed.values())
                assert time_check['bitwise_equal'] and torch.equal(result['semantic_labels'], altered['semantic_labels'])
                assert result['semantic_logits'].shape == (len(group), 2048, 26)
                assert torch.equal(result['semantic_logits'].argmax(-1), result['semantic_labels'])
                assert torch.equal(result['semantic_logits'][..., 0], result['semantic_raw_logits'][..., 0]+adapter.background_offset)
                assert torch.equal(result['semantic_logits'][..., 1:], result['semantic_raw_logits'][..., 1:])
                assert all(torch.isfinite(result[k]).all() for k in GEOMETRY_KEYS+('semantic_logits', 'semantic_raw_logits'))
                raw = result['semantic_raw_logits'].cpu().numpy(); labels = result['semantic_labels'].cpu().numpy()
                old_raw = np.stack([reference_outputs[seed]['logits'][reference_outputs[seed]['video_ids'] == c['video_id']] for c in group])
                old_cal = old_raw.copy(); old_cal[..., 0] += adapter.background_offset
                cache_logits_delta = delta(result['semantic_raw_logits'], torch.from_numpy(old_raw).cuda())
                per_video = {}
                for i, clip in enumerate(group):
                    vid = clip['video_id']; truth = caches[i]['labels']
                    assert np.array_equal(truth, reference_outputs[seed]['labels'][reference_outputs[seed]['video_ids'] == vid])
                    metric = classification_metrics(labels[i], truth, 26)
                    per_video[str(vid)] = {'online': metric,
                        'previous': reference_summary['results'][reference_arm]['runs'][seed]['per_video'][str(vid)]}
                    if compat: legacy_outputs[seed] = {'logits': raw[i], 'predictions': labels[i], 'labels': truth, 'video_ids': np.full(2048, vid, dtype=np.int32)}
                    if kind == 'B1': outputs[seed].append({'logits': raw[i], 'predictions': labels[i], 'labels': truth,
                        'video_ids': np.full(2048, vid, dtype=np.int32)})
                case = {'kind': kind, 'video_ids': [c['video_id'] for c in group], 'seed': seed, 'queries_per_video': 2048, 'feature_size': adapter.feature_size, 'background_offset': adapter.background_offset,
                    'geometry': geom, 'changed_time_geometry': geom_changed, 'semantic_time': time_check,
                    'online_vs_cached_features': feature_error, 'online_vs_saved_raw_logits': cache_logits_delta,
                    'online_vs_saved_label_changes': int((labels != old_cal.argmax(-1)).sum()),
                    'per_video': per_video, 'base_forward_cost': base_cost, 'full_interface_cost': cost}
                if kind == 'B2':
                    single = np.stack([next(p['logits'] for p in outputs[seed] if int(p['video_ids'][0]) == c['video_id']) for c in group])
                    case['B2_vs_B1_raw_logits'] = delta(result['semantic_raw_logits'], torch.from_numpy(single).cuda())
                cases.append(case)
                atomic_json(ROOT/'progress.json', {'protocol_sha256': file_hash(path), 'cases': cases})
                print('verified', kind, case['video_ids'], seed, 'feature max', feature_error['max_absolute_difference'],
                      'saved-logit max', cache_logits_delta['max_absolute_difference'], 'label changes', case['online_vs_saved_label_changes'], flush=True)
        assert torch.equal(saved_rgb, batch['semantic_video']) and torch.equal(saved_geo_rgb, batch['video'])
        assert all(torch.equal(saved[k], batch['query'][k]) for k in saved)

    for clip in clips:
        with np.load(feature_path(ARM, 'confirmation', clip['video_id'])) as p: cache = {k: p[k].copy() for k in p.files}
        assert len(cache['labels']) == 2048
        query = queries(cache, np.arange(2048), extra=False)
        video = load_video(clip)[None].cuda(); batch = {'video': video.to(torch.bfloat16), 'semantic_video': video, 'query': query}
        check_batch([clip], [cache], batch, 'B1')
        if len(first_pair) < 2: first_pair.append((clip, cache, batch))
    paired = {'video': torch.cat([p[2]['video'] for p in first_pair]),
              'semantic_video': torch.cat([p[2]['semantic_video'] for p in first_pair]),
              'query': {k: torch.cat([p[2]['query'][k] for p in first_pair]) for k in first_pair[0][2]['query']}}
    check_batch([p[0] for p in first_pair], [p[1] for p in first_pair], paired, 'B2')
    with np.load(legacy.path_resolver('dinov2', 'confirmation', 153)) as p: compat_cache = {k: p[k].copy() for k in p.files}
    compat_query = queries(compat_cache, np.arange(2048), extra=False)
    compat_video = load_video(legacy_clip)[None].cuda()
    check_batch([legacy_clip], [compat_cache], {'video': compat_video.to(torch.bfloat16), 'semantic_video': compat_video, 'query': compat_query}, 'legacy224')
    for seed in range(3): np.savez_compressed(ROOT/f'legacy_online_seed{seed}.npz', **legacy_outputs[seed])
    metrics = {}
    for seed in range(3):
        package = {k: np.concatenate([p[k] for p in outputs[seed]]) for k in outputs[seed][0]}
        np.savez_compressed(ROOT/f'online_seed{seed}.npz', **package)
        metric = classification_metrics(package['predictions'], package['labels'], 26)
        original_metric = previous['results'][ARM]['runs'][seed]['calibrated_confirmation']
        metrics[str(seed)] = {'online': metric, 'previous': original_metric,
            'macro_f1_difference': metric['macro_f1_present']-original_metric['macro_f1_present'],
            'foreground_macro_f1_difference': metric['foreground_macro_f1_present']-original_metric['foreground_macro_f1_present']}
    assert hashes == {str(p): file_hash(p) for p in inputs}
    assert all(not p.requires_grad for p in adapter.parameters())
    atomic_json(ROOT/'summary.json', {'protocol': protocol, 'cases': cases, 'metrics': metrics,
        'output_sha256': {name: file_hash(ROOT/name) for name in [f'{prefix}online_seed{s}.npz' for prefix in ('', 'legacy_') for s in range(3)]},
        'verification': {'all_geometry_same_input_bitwise_equal': True, 'all_semantic_time_checks_bitwise_equal': True,
                         'all_inputs_unchanged': True, 'all_parameters_frozen': True, 'legacy224_checked': True},
        'limits': ['already observed40 videos, not new model confirmation', 'B2 only first2 fixed videos',
                   'same batch geometry equivalence, not geometry accuracy or across-batch equality',
                   'single sequential timing, no standardized throughput benchmark']})
    print('DONE448 interface', {s: {'macro_f1': m['online']['macro_f1_present'], 'foreground_macro_f1': m['online']['foreground_macro_f1_present'], 'macro_difference': m['macro_f1_difference']} for s, m in metrics.items()}, flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(ROOT/'failure.json', {'error_type': type(error).__name__, 'message': str(error)})
        raise
