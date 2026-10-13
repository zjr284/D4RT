"""Frozen correctness diagnostics of the DINOv2 + D4RT inference API."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
from semantic.dino_interface import load_selected_adapter, cuda_bf16
from semantic.dino_sam_control import ROOT as PREVIOUS, checkpoint, path_resolver
from semantic.model import GEOMETRY_KEYS
from semantic.ytvis import load_video
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT = Path('semantic_artifacts/dino_interface_probe')
PROTOCOL = {'version': 1, 'analysis': 'frozen inference interface correctness; no new generalization confirmation',
    'seeds': [0, 1, 2], 'clip_selection': 'first3 round15 confirmation clips, no score-based selection',
    'cached_queries_per_clip': 64, 'query_selection': 'equally spaced indices in existing2048 query cache',
    'extra_queries': 'eight source times, corners, center and fractional coordinates',
    'background_offset': -1.75, 'training_updates': 0, 'calibration_updates': 0,
    'reference': 'original D4RT exact input batch and same CUDA BF16 autocast',
    'time_change': 't_tgt+1 and t_cam+3 modulo8, source fields unchanged',
    'geometry_requirement': 'bitwise equality for all six outputs', 'semantic_time_requirement': 'bitwise equality',
    'online_cache_differences': 'report all errors and label changes without score-based adjustment',
    'input_precision': 'primary float32 RGB, diagnostic BF16 RGB on first clip',
    'performance': 'one sequential invocation per case, original geometry first; not throughput benchmark',
    'real_batch_size': 1, 'unit_batch_size': 2, 'previously_processed_video_count_unchanged': 420}


def queries(cache, indices, extra=True):
    xy = cache['xy'][indices]
    query = {'u': torch.from_numpy(xy[:, 0].astype(np.float32)/127)[None].cuda(),
             'v': torch.from_numpy(xy[:, 1].astype(np.float32)/127)[None].cuda(),
             't_src': torch.from_numpy(xy[:, 2].astype(np.int64))[None].cuda(),
             't_tgt': torch.from_numpy(cache['t_tgt'][indices].astype(np.int64))[None].cuda(),
             't_cam': torch.from_numpy(cache['t_cam'][indices].astype(np.int64))[None].cuda()}
    if extra:
        uv = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1], [.5, .5], [.125, .875], [.333, .667], [.1, .9]], device='cuda')
        times = torch.arange(8, device='cuda')
        for key, value in {'u': uv[:, 0], 'v': uv[:, 1], 't_src': times, 't_tgt': (times+1)%8, 't_cam': (times+3)%8}.items():
            query[key] = torch.cat([query[key], value[None]], dim=1)
    return query


def measured(call):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    result = call(); torch.cuda.synchronize()
    return result, {'seconds': time.perf_counter()-started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}


def delta(a, b):
    error = (a.float()-b.float()).abs()
    return {'bitwise_equal': torch.equal(a, b), 'max_absolute_difference': float(error.max()), 'mean_absolute_difference': float(error.mean())}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((PREVIOUS / 'summary.json').read_text()); verified = json.loads((PREVIOUS / 'verification.json').read_text())
    assert file_hash(PREVIOUS / 'summary.json') == verified['summary_sha256']
    arm = previous['validation_selected_arm']; assert arm == 'coverage80_sam_background' and previous['results'][arm]['offset'] == -1.75
    clips = previous['manifest']['confirmation'][:3]; assert [c['video_id'] for c in clips] == [153, 52, 395]
    input_paths = [PREVIOUS/'summary.json', PREVIOUS/'verification.json']
    input_paths += [checkpoint(arm, seed) for seed in range(3)]
    input_paths += [path_resolver('dinov2', 'confirmation', c['video_id']) for c in clips]
    input_paths += [Path(frame) for c in clips for frame in c['frames']]
    hashes = {str(p): file_hash(p) for p in input_paths}
    protocol = dict(PROTOCOL, selected_arm=arm, input_sha256=hashes, inherited_protocol=previous['protocol'])
    path = ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Frozen interface inputs changed')
    atomic_json(path, protocol)
    adapter = load_selected_adapter(seed=0)
    cases = []; precision = []
    for clip in clips:
        vid = clip['video_id']
        with np.load(path_resolver('dinov2', 'confirmation', vid)) as p: cache = {k: p[k].copy() for k in p.files}
        assert len(cache['labels']) == 2048
        indices = np.arange(0, 2048, 32); query = queries(cache, indices)
        video = load_video(clip)[None].cuda(); batch = {'video': video, 'query': query}
        saved_query = {k: v.clone() for k, v in query.items()}; saved_video = video.clone()
        changed = {k: v.clone() for k, v in query.items()}
        changed['t_tgt'] = (changed['t_tgt']+1)%8; changed['t_cam'] = (changed['t_cam']+3)%8
        with torch.inference_mode():
            def original(b):
                with cuda_bf16(video.device): return adapter.base(b)
            ref, base_cost = measured(lambda: original(batch))
            ref_changed = original({'video': video, 'query': changed})
            online = adapter.semantic_features(video, query)
            cached = torch.from_numpy(cache['decoded'][indices]).float().cuda()[None]
            feature_error = delta(online[:, :64], cached)
            for seed in range(3):
                head = checkpoint(arm, seed); adapter.load_head(head, verified['checkpoint_sha256'][f'{arm}_{seed}'])
                result, cost = measured(lambda: adapter(batch))
                altered = adapter({'video': video, 'query': changed})
                geometry = {k: delta(result[k], ref[k]) for k in GEOMETRY_KEYS}
                geometry_changed = {k: delta(altered[k], ref_changed[k]) for k in GEOMETRY_KEYS}
                time_check = delta(result['semantic_logits'], altered['semantic_logits'])
                assert all(v['bitwise_equal'] for v in geometry.values()) and all(v['bitwise_equal'] for v in geometry_changed.values())
                assert time_check['bitwise_equal'] and torch.equal(result['semantic_labels'], altered['semantic_labels'])
                assert result['semantic_logits'].shape == (1, 72, 26) and result['semantic_labels'].shape == (1, 72)
                assert torch.equal(result['semantic_labels'], result['semantic_logits'].argmax(-1))
                assert torch.equal(result['semantic_raw_logits'][..., 1:], result['semantic_logits'][..., 1:])
                assert torch.equal(result['semantic_logits'][..., 0], result['semantic_raw_logits'][..., 0]-1.75)
                assert all(torch.isfinite(result[k]).all() for k in GEOMETRY_KEYS+('semantic_logits', 'semantic_raw_logits'))
                cached_output = adapter.semantic_outputs(cached)
                adapter.load_head(head, verified['checkpoint_sha256'][f'{arm}_{seed}'])
                reloaded = adapter.semantic_outputs(online)
                assert torch.equal(reloaded['semantic_logits'], result['semantic_logits'])
                cases.append({'video_id': vid, 'seed': seed, 'cache_query_indices': indices.tolist(), 'query_count': 72,
                    'geometry': geometry, 'changed_time_geometry': geometry_changed, 'semantic_time': time_check,
                    'online_vs_cached_features': feature_error,
                    'online_vs_cached_raw_logits': delta(result['semantic_raw_logits'][:, :64], cached_output['semantic_raw_logits']),
                    'online_vs_cached_label_changes': int((result['semantic_labels'][:, :64] != cached_output['semantic_labels']).sum()),
                    'semantic_head_reload_bitwise_equal': True, 'base_forward_cost': base_cost, 'full_interface_cost': cost})
                if vid == clips[0]['video_id']:
                    low_batch = {'video': video.to(torch.bfloat16), 'query': query}
                    low_ref = original(low_batch); low = adapter(low_batch)
                    assert all(torch.equal(low[k], low_ref[k]) for k in GEOMETRY_KEYS)
                    precision.append({'video_id': vid, 'seed': seed, 'geometry_unchanged_for_same_bf16_input': True,
                        'float32_vs_bf16_raw_logits': delta(result['semantic_raw_logits'], low['semantic_raw_logits']),
                        'float32_vs_bf16_label_changes': int((result['semantic_labels'] != low['semantic_labels']).sum()), 'query_count': 72})
                print('verified live interface', vid, seed, 'cached label changes', cases[-1]['online_vs_cached_label_changes'], flush=True)
        assert torch.equal(video, saved_video) and all(torch.equal(query[k], saved_query[k]) for k in query)
    assert all(not p.requires_grad for p in adapter.parameters())
    assert hashes == {str(p): file_hash(p) for p in input_paths}
    atomic_json(ROOT/'summary.json', {'protocol': protocol, 'cases': cases, 'input_precision_diagnostic': precision,
        'verification': {'all_108_original_and_changed_geometry_fields_bitwise_equal': True,
                         'all_nine_semantic_time_checks_bitwise_equal': True, 'all_three_heads_reload_bitwise_equal': True,
                         'all_inputs_and_checkpoint_hashes_unchanged': True, 'all_model_parameters_frozen': True,
                         'no_training_calibration_or_new_confirmation': True},
        'limits': ['three already-observed clips, 192 cache points plus24 extra points, three model seeds',
                   'CUDA real B1 only; B2 verified with deterministic unit stubs',
                   'geometry equivalence is not an independent geometry-accuracy benchmark',
                   'online/cache and BF16 RGB differences reported without adjusting models']})
    print('DONE DINO D4RT live interface probe', flush=True)


if __name__ == '__main__': main()
