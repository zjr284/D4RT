"""Preregistered fixed-backbone DINOv2 spatial resolution control."""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from semantic import dino_sam_control as prior
from semantic.semantic_feature_control import (VENDOR, WEIGHTS, patch_at_queries, save_npz, read_features,
    train_dino, infer, aggregate_hash, PROTOCOL as FEATURE_PROTOCOL)
from semantic.ytvis import load_dataset, load_video, sample_queries
from semantic.fresh_video_confirmation import select_clips, bootstrap_video_differences
from semantic.background_calibration import evaluate, predictions, pick_offset
from semantic.run import atomic_json, classification_metrics
from semantic.oracle_probe import file_hash

ROOT = Path('semantic_artifacts/dino_resolution_control')
BASELINE = 'coverage80_sam_background'
ARMS = ('resolution224', 'resolution448')
PROTOCOL = {'version': 1, 'round': 18, 'arms': list(ARMS), 'seeds': [0, 1, 2], 'confirmation_seed': 20261018,
    'train_videos': 80, 'validation_videos': 20, 'confirmation_videos': 40,
    'new_training': 'only three resolution448 heads; resolution224 selected heads and offset frozen',
    'input_rgb_resolution': 128, 'feature_dimensions': 384, 'output_classes': 26, 'head_parameters': 10010,
    'feature_recipe': 'frozen official ViT-S/14, float32 RGB bicubic to224/448, ImageNet norm, CUDA BF16',
    'patch_recipe': '16x16/32x32, pixel centers, bilinear border align_corners=False, float16 cache',
    'image_batch': 'all8 frames, with separate single-frame source-prefix extraction for training',
    'no_gt_features': True, 'steps': 2000, 'optimizer': 'AdamW lr.001 weight_decay.01',
    'batch': 'uniform128 foreground +128 background', 'checkpoint_selection': 'raw val macroF1 every50, ties earlier',
    'calibration_grid': FEATURE_PROTOCOL['calibration_grid'], 'calibration_selection': FEATURE_PROTOCOL['calibration_selection'],
    'frozen224_offset': -1.75, 'confirmation_after_selection_freeze': True,
    'source_metadata_previously_inspected': True, 'pretraining_ovis_overlap': 'unknown',
    'bootstrap_seed': 20261018, 'bootstrap_repetitions': 2000}


def checkpoint(arm, seed):
    if arm == 'resolution224': return prior.checkpoint(BASELINE, seed)
    return ROOT/arm/'dinov2'/'runs'/f'linear_seed{seed}.pt'


def feature_path(arm, split, vid):
    if arm == 'resolution224' and split != 'confirmation': return prior.path_resolver('dinov2', split, vid)
    return ROOT/'features'/arm/split/f'{vid}.npz'


def freeze(dataset):
    ROOT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((prior.ROOT/'summary.json').read_text()); verified = json.loads((prior.ROOT/'verification.json').read_text())
    assert file_hash(prior.ROOT/'summary.json') == verified['summary_sha256']
    assert previous['validation_selected_arm'] == BASELINE and previous['results'][BASELINE]['offset'] == -1.75
    old = previous['manifest']; excluded = set(old['excluded_video_ids']) | {c['video_id'] for c in old['confirmation']}
    assert len(excluded) == 420
    clips, count = select_clips(dataset, excluded, count=40, seed=20261018); assert count == 187
    manifest = {'seed': 20261018, 'excluded_video_ids': sorted(excluded), 'eligible_model_unevaluated': count,
                'train': old['training_sets']['coverage80'], 'val': old['val'], 'confirmation': clips,
                'source_category_metadata_previously_inspected': True}
    path = ROOT/'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest: raise ValueError('Preregistered manifest changed')
    atomic_json(path, manifest)
    fp = previous['protocol']['inherited_feature_coverage_protocol']['inherited_feature_protocol']
    assert file_hash(WEIGHTS) == fp['dinov2_weight_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip() == fp['dinov2_source_commit']
    paths = [prior.ROOT/'summary.json', prior.ROOT/'verification.json', WEIGHTS, Path('semantic_artifacts/ovis/annotations_train.json'), path]
    paths += [checkpoint('resolution224', s) for s in range(3)]
    paths += [checkpoint('resolution224', s).with_suffix('.json') for s in range(3)]
    paths += [feature_path('resolution224', split, c['video_id']) for split in ('train', 'val') for c in manifest[split]]
    paths += [prior.coverage.PREVIOUS/'teacher'/f"{c['video_id']}.npz" for c in manifest['train']]
    paths += [Path(frame) for split in ('train', 'val', 'confirmation') for c in manifest[split] for frame in c['frames']]
    paths += [Path('semantic/dino_resolution_control.py'), Path('semantic/OVIS_DINO_RESOLUTION_PROTOCOL.md')]
    hashes = {str(p): file_hash(p) for p in paths}
    for s in range(3): assert file_hash(checkpoint('resolution224', s)) == verified['checkpoint_sha256'][f'{BASELINE}_{s}']
    protocol = dict(PROTOCOL, manifest_sha256=file_hash(path), input_sha256=hashes,
                    dinov2_source_commit=fp['dinov2_source_commit'], inherited_protocol=previous['protocol'])
    path = ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Frozen protocol/inputs changed')
    atomic_json(path, protocol)
    return manifest, protocol


def load_dino():
    os.environ['XFORMERS_DISABLED'] = '1'; sys.path.insert(0, str(VENDOR.resolve()))
    from dinov2.hub.backbones import dinov2_vits14
    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    return model.to('cuda').eval().requires_grad_(False)


@torch.inference_mode()
def extract_tokens(model, video, size):
    mean = torch.tensor([.485, .456, .406], device=video.device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=video.device)[None, :, None, None]
    image = (F.interpolate(video.float(), size=(size, size), mode='bicubic', align_corners=False)-mean)/std
    with torch.autocast('cuda', dtype=torch.bfloat16): tokens = model.forward_features(image)['x_norm_patchtokens']
    grid = size//14
    assert tokens.shape == (len(video), grid*grid, 384)
    return tokens.reshape(len(video), grid, grid, 384).permute(0, 3, 1, 2)


def preflight(manifest, protocol):
    path = ROOT/'preflight.json'
    if path.exists():
        report = json.loads(path.read_text()); assert report['protocol_sha256'] == file_hash(ROOT/'protocol.json')
        return report
    model = load_dino(); video = load_video(manifest['val'][0]).cuda(); report = {}
    for size in (224, 448):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        tokens = extract_tokens(model, video, size); torch.cuda.synchronize()
        assert torch.isfinite(tokens).all()
        report[str(size)] = {'patch_shape': list(tokens.shape), 'seconds': time.perf_counter()-started,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}
        del tokens
    report.update({'video_id': manifest['val'][0]['video_id'], 'protocol_sha256': file_hash(ROOT/'protocol.json'),
                   'note': 'first validation video, no classifier or GT; single sequential cost, not throughput benchmark'})
    atomic_json(path, report); del model, video; gc.collect(); torch.cuda.empty_cache()
    print('preflight passed', report, flush=True); return report


def metadata(split, clip):
    vid = clip['video_id']
    path = ROOT/'query_metadata'/'confirmation'/f'{vid}.npz' if split == 'confirmation' else feature_path('resolution224', split, vid)
    with np.load(path) as p: return {k: p[k].copy() for k in ('labels', 'xy', 't_tgt', 't_cam', 'video_id')}


def confirmation_queries(dataset, manifest):
    for clip in manifest['confirmation']:
        path = ROOT/'query_metadata'/'confirmation'/f"{clip['video_id']}.npz"
        if path.exists(): continue
        maps = np.stack([dataset.gt_label_map(clip, t) for t in range(8)])
        q, y, xy = sample_queries(maps, per_frame=256, seed=42+clip['video_id'])
        save_npz(path, labels=y.numpy().astype(np.int16), xy=xy.numpy().astype(np.int16),
                 t_tgt=q['t_tgt'][0].numpy(), t_cam=q['t_cam'][0].numpy(), video_id=np.int64(clip['video_id']))


def cache_features(arm, splits, manifest):
    tasks = [(s, c) for s in splits for c in manifest[s] if not feature_path(arm, s, c['video_id']).exists()]
    if not tasks: return
    if 'confirmation' in splits: assert (ROOT/'frozen_selection.json').exists()
    model = load_dino(); size = 224 if arm == 'resolution224' else 448
    for split, clip in tasks:
        data = metadata(split, clip); video = load_video(clip).cuda()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        tokens = extract_tokens(model, video, size)
        with torch.inference_mode():
            data['decoded'] = patch_at_queries(tokens, data['xy']).to(torch.float16).cpu().numpy()
            if split == 'train':
                mask = data['xy'][:, 2] == 0; source = extract_tokens(model, video[:1], size)
                data['decoded'][mask] = patch_at_queries(source, data['xy'][mask]).to(torch.float16).cpu().numpy()
                del source
        torch.cuda.synchronize()
        cost = {'seconds': time.perf_counter()-started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}
        assert np.isfinite(data['decoded']).all()
        path = feature_path(arm, split, clip['video_id']); save_npz(path, **data)
        atomic_json(path.with_suffix('.cost.json'), cost)
        print('cached', arm, split, clip['video_id'], len(data['labels']), flush=True)
        del tokens, video
    del model; gc.collect(); torch.cuda.empty_cache()


def read(arm, split, manifest):
    return read_features('dinov2', split, manifest, path_resolver=lambda _, s, v: feature_path(arm, s, v))


def feature_hashes(manifest, splits):
    return {f'{arm}_{s}': aggregate_hash([feature_path(arm, s, c['video_id']) for c in manifest[s]]) for arm in ARMS for s in splits}


def infer_arm(arm, seed, split, clips):
    return infer('dinov2', seed, split, clips, checkpoint_path=checkpoint(arm, seed), output_root=ROOT/arm)


def main():
    if (ROOT/'summary.json').exists(): raise RuntimeError('Completed experiment exists; verify rather than rerun')
    dataset = load_dataset('ovis', Path('semantic_artifacts/ovis/annotations_train.json'), Path('semantic_artifacts/ovis'))
    manifest, protocol = freeze(dataset); resources = preflight(manifest, protocol)
    cache_features('resolution448', ('train', 'val'), manifest)
    hashes = feature_hashes(manifest, ('train', 'val'))
    path = ROOT/'training_feature_hashes.json'
    if path.exists() and json.loads(path.read_text()) != hashes: raise ValueError('Training features changed')
    atomic_json(path, hashes)
    train = read('resolution448', 'train', manifest); val = read('resolution448', 'val', manifest)
    for s in range(3):
        report = train_dino(s, train, val, dict(protocol, arm='resolution448', feature_sha256=hashes), output_root=ROOT/'resolution448')
        old_report = json.loads(checkpoint('resolution224', s).with_suffix('.json').read_text())
        assert report['sampled_class_counts'] == old_report['sampled_class_counts'] and report['parameter_count'] == 10010
    del train, val
    validation = {}; offsets = {'resolution224': -1.75}; curves = {}
    previous = json.loads((prior.ROOT/'summary.json').read_text())
    for arm in ARMS:
        clips = read(arm, 'val', manifest); validation[arm] = [infer_arm(arm, s, 'val', clips) for s in range(3)]; del clips
        if arm == 'resolution448': offsets[arm], curves[arm] = pick_offset(validation[arm], protocol['calibration_grid'])
        else:
            for s in range(3): assert evaluate(validation[arm][s], -1.75) == previous['results'][BASELINE]['runs'][s]['calibrated_validation']
    selected = max(ARMS, key=lambda a: np.mean([evaluate(p, offsets[a])['macro_f1_present'] for p in validation[a]]))
    selection = {'offsets': offsets, 'validation_curves': curves, 'validation_selected_arm': selected,
                 'mean_validation_macro_f1': {a: float(np.mean([evaluate(p, offsets[a])['macro_f1_present'] for p in validation[a]])) for a in ARMS},
                 'checkpoint_sha256': {f'{a}_{s}': file_hash(checkpoint(a, s)) for a in ARMS for s in range(3)}}
    path = ROOT/'frozen_selection.json'
    if path.exists() and json.loads(path.read_text()) != selection: raise ValueError('Validation selection changed')
    atomic_json(path, selection); print('FROZEN validation selection', selected, selection['mean_validation_macro_f1'], offsets, flush=True)
    confirmation_queries(dataset, manifest)
    for arm in ARMS: cache_features(arm, ('confirmation',), manifest)
    results = {}
    for arm in ARMS:
        clips = read(arm, 'confirmation', manifest); runs = []
        for s in range(3):
            p = infer_arm(arm, s, 'confirmation', clips); predicted = predictions(p['logits'], offsets[arm])
            per_video = {str(c['video_id']): classification_metrics(predicted[p['video_ids'] == c['video_id']], p['labels'][p['video_ids'] == c['video_id']], 26) for c in manifest['confirmation']}
            runs.append({'seed': s, 'raw_validation': evaluate(validation[arm][s], 0), 'calibrated_validation': evaluate(validation[arm][s], offsets[arm]),
                         'raw_confirmation': evaluate(p, 0), 'calibrated_confirmation': evaluate(p, offsets[arm]), 'per_video': per_video})
            print('confirmed', arm, s, runs[-1]['calibrated_confirmation']['macro_f1_present'], flush=True)
        results[arm] = {'offset': offsets[arm], 'runs': runs}; del clips
    contrasts = {}
    for metric in ('macro_f1_present', 'foreground_macro_f1_present', 'foreground_accuracy'):
        differences = [results['resolution448']['runs'][s]['calibrated_confirmation'][metric]-results['resolution224']['runs'][s]['calibrated_confirmation'][metric] for s in range(3)]
        contrasts[metric] = {'paired_differences': differences, 'mean': float(np.mean(differences)), 'std': float(np.std(differences, ddof=1))}
    per_video = {str(c['video_id']): float(np.mean([results['resolution448']['runs'][s]['per_video'][str(c['video_id'])]['macro_f1_present']-results['resolution224']['runs'][s]['per_video'][str(c['video_id'])]['macro_f1_present'] for s in range(3)])) for c in manifest['confirmation']}
    contrasts['per_video_differences'] = per_video; contrasts['video_bootstrap'] = bootstrap_video_differences(list(per_video.values()), seed=20261018)
    assert feature_hashes(manifest, ('train', 'val')) == hashes
    for p, h in protocol['input_sha256'].items(): assert file_hash(p) == h
    assert selection['checkpoint_sha256'] == {f'{a}_{s}': file_hash(checkpoint(a, s)) for a in ARMS for s in range(3)}
    atomic_json(ROOT/'summary.json', {'protocol': protocol, 'manifest': manifest, 'preflight': resources,
        'feature_sha256': feature_hashes(manifest, ('train', 'val', 'confirmation')), 'results': results,
        'contrasts': contrasts, 'validation_selected_arm': selected, 'frozen_selection_sha256': file_hash(path),
        'verification': {'only_three448_heads_trained': True, 'baseline_weights_offsets_labels_and_sampling_unchanged': True,
                         'confirmation_excludes_prior420': True, 'all_confirmation_features_and_outputs_after_selection_freeze': True}})
    print('DONE DINO resolution control', contrasts['macro_f1_present'], contrasts['video_bootstrap'], flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as error:
        ROOT.mkdir(parents=True, exist_ok=True); atomic_json(ROOT/'failure.json', {'type': type(error).__name__, 'message': str(error)})
        raise
