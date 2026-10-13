"""Frozen DINOv2 features: source-GT vs existing SAM supervision in two video sets."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from semantic import feature_coverage_control as coverage
from semantic.semantic_feature_control import (cache_dino, train_dino, read_features, infer, aggregate_hash,
    save_npz, PROTOCOL as FEATURE_PROTOCOL)
from semantic.video_coverage_intervention import training_union
from semantic.ytvis import load_dataset, sample_queries
from semantic.fresh_video_confirmation import select_clips, bootstrap_video_differences
from semantic.background_calibration import evaluate, predictions, pick_offset
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json, classification_metrics

ROOT = Path('semantic_artifacts/dino_sam_control')
ARMS = ('random80_source_gt', 'random80_sam_background', 'coverage80_source_gt', 'coverage80_sam_background')
PROTOCOL = {'version': 1, 'arms': list(ARMS), 'seeds': [0, 1, 2], 'confirmation_seed': 20261017,
    'confirmation_videos': 40, 'train_videos_each': 80, 'validation_videos': 20,
    'new_training': 'only two SAM arms, six new heads; source-GT heads and offsets frozen',
    'frozen_gt_offset': -1.5, 'features': 'unchanged round13 frozen DINOv2 ViT-S/14 patch recipe',
    'source_features': 'copy exact existing source-GT query feature values into SAM source prefix',
    'teacher': 'reuse round11 SAM2.1 tiny labels, masks and confidence rules unchanged; later GT never filters labels',
    'steps': 2000, 'optimizer': 'AdamW lr=0.001 weight_decay=0.01',
    'batch': '128 foreground + 128 background, uniform within group',
    'selection': 'raw validation macro F1 every50 steps, ties earlier',
    'calibration_grid': FEATURE_PROTOCOL['calibration_grid'], 'calibration_selection': FEATURE_PROTOCOL['calibration_selection'],
    'confirmation_after_model_and_offset_freeze': True, 'source_metadata_previously_inspected': True,
    'bootstrap_seed': 20261017, 'bootstrap_repetitions': 2000, 'pretraining_ovis_overlap': 'unknown'}


def checkpoint(arm, seed):
    group, label = arm.split('_', 1)
    if label == 'source_gt': return coverage.checkpoint(f'{group}_dinov2', seed)
    return ROOT / arm / 'dinov2' / 'runs' / f'linear_seed{seed}.pt'


def path_resolver(feature, split, vid):
    if feature == 'd4rt':
        if split == 'confirmation': return ROOT / 'query_metadata' / split / f'{vid}.npz'
        return coverage.PREVIOUS / 'features' / split / f'{vid}.npz'
    if split == 'val': return coverage.SEMANTIC / 'features' / 'dinov2' / 'val' / f'{vid}.npz'
    return ROOT / 'features' / 'dinov2' / split / f'{vid}.npz'


def group_manifest(manifest, group):
    return {'train': manifest['training_sets'][group], 'val': manifest['val'], 'confirmation': manifest['confirmation']}


def freeze(dataset):
    ROOT.mkdir(parents=True, exist_ok=True)
    old, inherited = coverage.freeze(dataset)
    previous = json.loads((coverage.ROOT / 'summary.json').read_text())
    verify = json.loads((coverage.ROOT / 'verification.json').read_text())
    assert verify['summary_sha256'] == file_hash(coverage.ROOT / 'summary.json')
    excluded = set(old['excluded_video_ids']) | {c['video_id'] for c in old['confirmation']}; assert len(excluded) == 380
    path = ROOT / 'manifest.json'
    if path.exists(): manifest = json.loads(path.read_text())
    else:
        clips, count = select_clips(dataset, excluded, count=40, seed=20261017)
        manifest = {'seed': 20261017, 'excluded_video_ids': sorted(excluded), 'eligible_model_unevaluated': count,
                    'training_sets': old['training_sets'], 'val': old['val'], 'confirmation': clips,
                    'source_category_metadata_previously_inspected': True}
        atomic_json(path, manifest)
    assert manifest['excluded_video_ids'] == sorted(excluded) and manifest['eligible_model_unevaluated'] == 227
    assert manifest['training_sets'] == old['training_sets'] and manifest['val'] == old['val']
    assert len(manifest['confirmation']) == 40 and len({c['video_id'] for c in manifest['confirmation']}) == 40
    assert not {c['video_id'] for c in manifest['confirmation']} & excluded
    hashes = {}
    for group in ('random80', 'coverage80'):
        prior_arm = f'{group}_dinov2'; assert previous['results'][prior_arm]['offset'] == -1.5
        for seed in range(3):
            sha = file_hash(checkpoint(f'{group}_source_gt', seed)); hashes[f'{group}_{seed}'] = sha
            assert sha == verify['checkpoint_sha256'][f'{prior_arm}_{seed}']
    clips = training_union(manifest)
    protocol = dict(PROTOCOL, manifest_sha256=file_hash(path), inherited_feature_coverage_protocol=inherited,
        frozen_gt_checkpoint_sha256=hashes, round14_summary_sha256=file_hash(coverage.ROOT / 'summary.json'),
        teacher_sha256=aggregate_hash([coverage.PREVIOUS / 'teacher' / f"{c['video_id']}.npz" for c in clips]),
        teacher_query_sha256=aggregate_hash([path_resolver('d4rt', 'train', c['video_id']) for c in clips]))
    path = ROOT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Protocol or teacher inputs changed')
    atomic_json(path, protocol)
    return manifest, protocol


def cache_confirmation_queries(dataset, manifest):
    for clip in manifest['confirmation']:
        path = path_resolver('d4rt', 'confirmation', clip['video_id'])
        if path.exists(): continue
        maps = np.stack([dataset.gt_label_map(clip, t) for t in range(8)])
        q, labels, xy = sample_queries(maps, per_frame=256, seed=42 + clip['video_id'])
        save_npz(path, labels=labels.numpy().astype(np.int16), xy=xy.numpy().astype(np.int16),
                 t_tgt=q['t_tgt'][0].numpy(), t_cam=q['t_cam'][0].numpy(), video_id=np.int64(clip['video_id']))


def reuse_source_prefix(manifest):
    for clip in training_union(manifest):
        vid = clip['video_id']; path = path_resolver('dinov2', 'train', vid)
        with np.load(path) as p: data = {k: p[k].copy() for k in p.files}
        with np.load(coverage.feature_path('dinov2', 'train', vid)) as p: source = {k: p[k].copy() for k in p.files}
        mask = data['xy'][:, 2] == 0
        assert int(source['video_id']) == vid
        for k in ('labels', 'xy', 't_tgt', 't_cam'): assert np.array_equal(data[k][mask], source[k])
        if not np.array_equal(data['decoded'][mask], source['decoded']):
            data['decoded'][mask] = source['decoded']; save_npz(path, **data)


def read(arm, split, manifest):
    group, label = arm.split('_', 1)
    clips = read_features('dinov2', split, group_manifest(manifest, group), path_resolver=path_resolver)
    if split == 'train' and label == 'source_gt':
        for c in clips:
            mask = c['xy'][:, 2] == 0
            for k in ('decoded', 'labels', 'xy', 't_tgt', 't_cam'): c[k] = c[k][mask]
    return clips


def infer_arm(arm, seed, split, clips):
    return infer('dinov2', seed, split, clips, checkpoint_path=checkpoint(arm, seed), output_root=ROOT / arm)


def feature_hashes(manifest):
    return {f'{group}_{split}': aggregate_hash([path_resolver('dinov2', split, c['video_id']) for c in group_manifest(manifest, group)[split]])
            for group in ('random80', 'coverage80') for split in ('train', 'val', 'confirmation')}


def contrast(treatment, control, results, manifest):
    entry = {}
    for metric in ('macro_f1_present', 'foreground_macro_f1_present', 'foreground_accuracy'):
        d = [results[treatment]['runs'][i]['calibrated_confirmation'][metric] - results[control]['runs'][i]['calibrated_confirmation'][metric] for i in range(3)]
        entry[metric] = {'paired_differences': d, 'mean': float(np.mean(d)), 'std': float(np.std(d, ddof=1))}
    d = {str(c['video_id']): float(np.mean([results[treatment]['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present'] - results[control]['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present'] for i in range(3)])) for c in manifest['confirmation']}
    entry['per_video_differences'] = d; entry['video_bootstrap'] = bootstrap_video_differences(list(d.values()), seed=20261017)
    return entry


def main():
    dataset = load_dataset('ovis', Path('semantic_artifacts/ovis/annotations_train.json'), Path('semantic_artifacts/ovis'))
    manifest, protocol = freeze(dataset); cache_confirmation_queries(dataset, manifest)
    cache_manifest = {'train': training_union(manifest), 'val': manifest['val'], 'confirmation': manifest['confirmation']}
    cache_dino(cache_manifest, path_resolver=path_resolver, training_source_only=False)
    reuse_source_prefix(manifest)
    hashes = feature_hashes(manifest); path = ROOT / 'feature_hashes.json'
    if path.exists() and json.loads(path.read_text()) != hashes: raise ValueError('Features changed')
    atomic_json(path, hashes)
    reports = {}
    for group in ('random80', 'coverage80'):
        arm = f'{group}_sam_background'; train = read(arm, 'train', manifest); val = read(arm, 'val', manifest)
        reports[arm] = [train_dino(seed, train, val, dict(protocol, arm=arm, feature_sha256=hashes), output_root=ROOT / arm) for seed in range(3)]
        del train, val
    offsets = {}; curves = {}; validation = {}
    for arm in ARMS:
        clips = read(arm, 'val', manifest); validation[arm] = [infer_arm(arm, seed, 'val', clips) for seed in range(3)]
        if arm.endswith('source_gt'): offsets[arm] = -1.5
        else: offsets[arm], curves[arm] = pick_offset(validation[arm], protocol['calibration_grid'])
        del clips
    selection = {'offsets': offsets, 'validation_curves': curves,
                 'checkpoint_sha256': {f'{a}_{s}': file_hash(checkpoint(a, s)) for a in ARMS for s in range(3)}}
    path = ROOT / 'frozen_selection.json'
    if path.exists() and json.loads(path.read_text()) != selection: raise ValueError('Selection changed')
    atomic_json(path, selection)
    results = {}; previous = json.loads((coverage.ROOT / 'summary.json').read_text())
    for arm in ARMS:
        group = arm.split('_')[0]; clips = read(arm, 'confirmation', manifest); runs = []
        for seed in range(3):
            package = infer_arm(arm, seed, 'confirmation', clips); metric = evaluate(package, offsets[arm]); valmetric = evaluate(validation[arm][seed], offsets[arm])
            if arm.endswith('source_gt'): assert valmetric == previous['results'][f'{group}_dinov2']['runs'][seed]['calibrated_validation']
            predicted = predictions(package['logits'], offsets[arm])
            per_video = {str(c['video_id']): classification_metrics(predicted[package['video_ids'] == c['video_id']], package['labels'][package['video_ids'] == c['video_id']], 26) for c in clips}
            runs.append({'seed': seed, 'raw_confirmation': evaluate(package, 0), 'calibrated_confirmation': metric,
                         'raw_validation': evaluate(validation[arm][seed], 0), 'calibrated_validation': valmetric, 'per_video': per_video})
            print('confirmed', arm, seed, metric['macro_f1_present'], flush=True)
        results[arm] = {'offset': offsets[arm], 'runs': runs}; del clips
    pairs = [(f'{g}_sam_background', f'{g}_source_gt') for g in ('random80', 'coverage80')]
    contrasts = {f'{a}_minus_{b}': contrast(a, b, results, manifest) for a, b in pairs}
    assert feature_hashes(manifest) == hashes
    _, end = freeze(dataset); assert end == protocol
    assert selection['checkpoint_sha256'] == {f'{a}_{s}': file_hash(checkpoint(a, s)) for a in ARMS for s in range(3)}
    atomic_json(ROOT / 'summary.json', {'protocol': protocol, 'manifest': manifest, 'feature_sha256': hashes,
        'results': results, 'contrasts': contrasts,
        'validation_selected_arm': max(results, key=lambda a: np.mean([r['calibrated_validation']['macro_f1_present'] for r in results[a]['runs']])),
        'verification': {'only_six_sam_heads_trained': True, 'all_six_gt_validation_scores_reproduced': True,
                         'baseline_heads_offsets_teacher_and_features_unchanged': True, 'confirmation_excludes_prior380': True,
                         'all_confirmation_outputs_after_selection_freeze': True}})
    print('DONE DINOv2 SAM control', flush=True)


if __name__ == '__main__': main()
