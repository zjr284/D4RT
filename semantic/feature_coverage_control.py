"""2x2 fixed feature family x equal-size video selection, source-GT only."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np
from semantic.semantic_feature_control import (ROOT as SEMANTIC, PREVIOUS, PROTOCOL as FEATURE_PROTOCOL,
    aggregate_hash, cache_new_d4rt, cache_dino, read_features, train_dino, infer, VENDOR, WEIGHTS, DEFAULT_VARIANT)
from semantic.ytvis import load_dataset
from semantic.fresh_video_confirmation import select_clips, bootstrap_video_differences
from semantic.background_calibration import evaluate, predictions, pick_offset
from semantic.video_coverage_intervention import source_coverage, training_union
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json, classification_metrics

ROOT = Path('semantic_artifacts/feature_coverage_control')
ARMS = ('random80_d4rt', 'coverage80_d4rt', 'random80_dinov2', 'coverage80_dinov2')
OLD_OFFSETS = {'random80_d4rt': -1.25, 'coverage80_d4rt': -.75, 'random80_dinov2': -1.5}
PROTOCOL = {'version': 1, 'analysis': '2x2 source-GT feature x equal-size video selection',
    'arms': list(ARMS), 'confirmation_seed': 20261016, 'confirmation_videos': 40,
    'train_videos_each': 80, 'validation_videos': 20, 'seeds': [0, 1, 2],
    'new_training_arm': 'coverage80_dinov2', 'frozen_offsets': OLD_OFFSETS,
    'feature_recipe': 'unchanged from round13 official frozen DINOv2 ViT-S/14 patch feature',
    'optimizer': 'AdamW lr=0.001 weight_decay=0.01', 'steps': 2000,
    'sampling': '128 foreground + 128 background, uniform within group',
    'selection': 'raw validation macro F1 every 50 steps, tie earlier',
    'calibration_grid': FEATURE_PROTOCOL['calibration_grid'], 'calibration_selection': FEATURE_PROTOCOL['calibration_selection'],
    'confirmation_after_all_selection': True, 'source_metadata_previously_inspected': True,
    'interaction': '(DINO coverage - DINO random) - (D4RT coverage - D4RT random)',
    'bootstrap_seed': 20261016, 'bootstrap_repetitions': 2000,
    'pretraining_ovis_overlap': 'unknown'}


def checkpoint(arm, seed):
    group, feature = arm.split('_')
    if feature == 'd4rt': return PREVIOUS / f'{group}_source_gt' / 'runs' / f'linear_2000_seed{seed}.pt'
    if group == 'random80': return SEMANTIC / 'dinov2' / 'runs' / f'linear_seed{seed}.pt'
    return ROOT / 'coverage80' / 'dinov2' / 'runs' / f'linear_seed{seed}.pt'


def feature_path(feature, split, vid):
    if split == 'confirmation': return ROOT / 'features' / feature / split / f'{vid}.npz'
    if feature == 'd4rt': return PREVIOUS / 'features' / split / f'{vid}.npz'
    prior = SEMANTIC / 'features' / 'dinov2' / split / f'{vid}.npz'
    return prior if prior.exists() else ROOT / 'features' / 'dinov2' / split / f'{vid}.npz'


def group_manifest(manifest, group):
    return {'train': manifest['training_sets'][group], 'val': manifest['val'], 'confirmation': manifest['confirmation']}


def freeze(dataset):
    ROOT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((PREVIOUS / 'summary.json').read_text())
    semantic = json.loads((SEMANTIC / 'summary.json').read_text())
    old = prior['manifest']
    excluded = set(semantic['manifest']['excluded_video_ids']) | {c['video_id'] for c in semantic['manifest']['confirmation']}
    assert len(excluded) == 340
    path = ROOT / 'manifest.json'
    if path.exists(): manifest = json.loads(path.read_text())
    else:
        clips, eligible = select_clips(dataset, excluded, count=40, seed=20261016)
        manifest = {'seed': 20261016, 'excluded_video_ids': sorted(excluded), 'eligible_model_unevaluated': eligible,
                    'training_sets': old['training_sets'], 'val': old['val'], 'confirmation': clips,
                    'source_category_metadata_previously_inspected': True}
        atomic_json(path, manifest)
    assert manifest['excluded_video_ids'] == sorted(excluded) and manifest['eligible_model_unevaluated'] == 267
    assert manifest['training_sets'] == old['training_sets'] and manifest['val'] == old['val']
    assert len(manifest['confirmation']) == 40 and len({c['video_id'] for c in manifest['confirmation']}) == 40
    assert not {c['video_id'] for c in manifest['confirmation']} & excluded
    verified = json.loads((PREVIOUS / 'verification.json').read_text())
    semverified = json.loads((SEMANTIC / 'verification.json').read_text())
    assert verified['summary_sha256'] == file_hash(PREVIOUS / 'summary.json')
    assert semverified['summary_sha256'] == file_hash(SEMANTIC / 'summary.json')
    fp = semantic['protocol']
    assert file_hash(WEIGHTS) == fp['dinov2_weight_sha256']
    assert file_hash(DEFAULT_VARIANT / 'opend4rt.ckpt') == fp['d4rt_backbone_sha256']
    assert file_hash(DEFAULT_VARIANT / 'model.yaml') == fp['d4rt_model_config_sha256']
    assert file_hash('semantic_artifacts/ovis/annotations_train.json') == fp['annotation_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip() == fp['dinov2_source_commit']
    assert aggregate_hash([feature_path('d4rt', 'train', c['video_id']) for c in training_union(old)]) == verified['feature_sha256']['train']
    assert aggregate_hash([feature_path('d4rt', 'val', c['video_id']) for c in old['val']]) == verified['feature_sha256']['val']
    for split in ('train', 'val'):
        clips = semantic['manifest'][split]
        assert aggregate_hash([feature_path('dinov2', split, c['video_id']) for c in clips]) == semantic['feature_sha256'][f'dinov2_{split}']
    hashes = {}
    for arm in OLD_OFFSETS:
        group, feature = arm.split('_')
        expected_offset = prior['results'][f'{group}_source_gt']['offset'] if feature == 'd4rt' else semantic['results']['dinov2']['offset']
        assert expected_offset == OLD_OFFSETS[arm]
        for seed in PROTOCOL['seeds']:
            sha = file_hash(checkpoint(arm, seed)); hashes[f'{arm}_{seed}'] = sha
            expected = verified['checkpoint_sha256'][f'{group}_source_gt_2000_{seed}'] if feature == 'd4rt' else semverified['checkpoint_sha256'][f'dinov2_{seed}']
            assert sha == expected
    protocol = dict(PROTOCOL, manifest_sha256=file_hash(path), frozen_checkpoint_sha256=hashes,
                    round11_summary_sha256=file_hash(PREVIOUS / 'summary.json'), round13_summary_sha256=file_hash(SEMANTIC / 'summary.json'),
                    inherited_feature_protocol=semantic['protocol'])
    path = ROOT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Changed frozen protocol or inputs')
    atomic_json(path, protocol)
    return manifest, protocol


def read(arm, split, manifest):
    group, feature = arm.split('_')
    return read_features(feature, split, group_manifest(manifest, group), path_resolver=feature_path)


def infer_arm(arm, seed, split, clips):
    feature = arm.split('_')[1]
    return infer(feature, seed, split, clips, checkpoint_path=checkpoint(arm, seed), output_root=ROOT / arm)


def feature_hashes(manifest):
    return {f'{arm}_{split}': aggregate_hash([feature_path(arm.split('_')[1], split, c['video_id'])
        for c in group_manifest(manifest, arm.split('_')[0])[split]]) for arm in ARMS for split in ('train', 'val', 'confirmation')}


def contrast(treatment, control, results, manifest):
    entry = {}
    for metric in ('macro_f1_present', 'foreground_macro_f1_present', 'foreground_accuracy'):
        d = [results[treatment]['runs'][i]['calibrated_confirmation'][metric] - results[control]['runs'][i]['calibrated_confirmation'][metric] for i in range(3)]
        entry[metric] = {'paired_differences': d, 'mean': float(np.mean(d)), 'std': float(np.std(d, ddof=1))}
    d = {str(c['video_id']): float(np.mean([results[treatment]['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present'] - results[control]['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present'] for i in range(3)])) for c in manifest['confirmation']}
    entry['per_video_differences'] = d; entry['video_bootstrap'] = bootstrap_video_differences(list(d.values()), seed=20261016)
    return entry


def main():
    dataset = load_dataset('ovis', Path('semantic_artifacts/ovis/annotations_train.json'), Path('semantic_artifacts/ovis'))
    manifest, protocol = freeze(dataset)
    cache_new_d4rt(dataset, group_manifest(manifest, 'coverage80'), output_root=ROOT)
    cache_dino(group_manifest(manifest, 'coverage80'), path_resolver=feature_path)
    hashes = feature_hashes(manifest); path = ROOT / 'feature_hashes.json'
    if path.exists() and json.loads(path.read_text()) != hashes: raise ValueError('Cached features changed')
    atomic_json(path, hashes)
    coverage = {}
    for group in ('random80', 'coverage80'):
        for split in ('train', 'val', 'confirmation'):
            a = read(f'{group}_d4rt', split, manifest); b = read(f'{group}_dinov2', split, manifest)
            for x, y in zip(a, b):
                for key in ('labels', 'xy', 'video_id', 't_tgt', 't_cam'): assert np.array_equal(x[key], y[key])
            if split == 'train': coverage[group] = source_coverage(a)
            del a, b
    assert min(coverage['coverage80'].values()) >= 3
    train = read('coverage80_dinov2', 'train', manifest); val = read('coverage80_dinov2', 'val', manifest)
    reports = [train_dino(seed, train, val, dict(protocol, feature_sha256=hashes), output_root=ROOT / 'coverage80') for seed in PROTOCOL['seeds']]
    del train, val
    validation = {}; offsets = dict(OLD_OFFSETS); curves = {}
    for arm in ARMS:
        clips = read(arm, 'val', manifest)
        validation[arm] = [infer_arm(arm, seed, 'val', clips) for seed in PROTOCOL['seeds']]
        if arm == 'coverage80_dinov2': offsets[arm], curves[arm] = pick_offset(validation[arm], protocol['calibration_grid'])
        del clips
    selection = {'offsets': offsets, 'new_arm_validation_curves': curves,
                 'checkpoint_sha256': {f'{arm}_{seed}': file_hash(checkpoint(arm, seed)) for arm in ARMS for seed in PROTOCOL['seeds']}}
    path = ROOT / 'frozen_selection.json'
    if path.exists() and json.loads(path.read_text()) != selection: raise ValueError('Frozen selection changed')
    atomic_json(path, selection)
    results = {}; previous = json.loads((PREVIOUS / 'summary.json').read_text()); semantic = json.loads((SEMANTIC / 'summary.json').read_text())
    for arm in ARMS:
        group, feature = arm.split('_'); clips = read(arm, 'confirmation', manifest); runs = []
        for seed in PROTOCOL['seeds']:
            package = infer_arm(arm, seed, 'confirmation', clips); metric = evaluate(package, offsets[arm])
            validation_metric = evaluate(validation[arm][seed], offsets[arm])
            if arm in OLD_OFFSETS:
                old = previous['results'][f'{group}_source_gt'] if feature == 'd4rt' else semantic['results']['dinov2']
                assert validation_metric == old['runs'][seed]['calibrated_validation']
            predicted = predictions(package['logits'], offsets[arm])
            per_video = {str(c['video_id']): classification_metrics(predicted[package['video_ids'] == c['video_id']], package['labels'][package['video_ids'] == c['video_id']], 26) for c in clips}
            runs.append({'seed': seed, 'raw_confirmation': evaluate(package, 0), 'calibrated_confirmation': metric,
                         'raw_validation': evaluate(validation[arm][seed], 0), 'calibrated_validation': validation_metric, 'per_video': per_video})
            print('confirmed', arm, seed, metric['macro_f1_present'], flush=True)
        results[arm] = {'offset': offsets[arm], 'runs': runs}; del clips
    pairs = [('coverage80_d4rt', 'random80_d4rt'), ('coverage80_dinov2', 'random80_dinov2'),
             ('random80_dinov2', 'random80_d4rt'), ('coverage80_dinov2', 'coverage80_d4rt')]
    contrasts = {f'{a}_minus_{b}': contrast(a, b, results, manifest) for a, b in pairs}
    interaction = {}
    for metric in ('macro_f1_present', 'foreground_macro_f1_present', 'foreground_accuracy'):
        values = np.array(contrasts['coverage80_dinov2_minus_random80_dinov2'][metric]['paired_differences']) - np.array(contrasts['coverage80_d4rt_minus_random80_d4rt'][metric]['paired_differences'])
        interaction[metric] = {'paired_differences': values.tolist(), 'mean': float(values.mean()), 'std': float(values.std(ddof=1))}
    assert hashes == feature_hashes(manifest)
    _, end = freeze(dataset); assert end == protocol
    assert selection['checkpoint_sha256'] == {f'{arm}_{seed}': file_hash(checkpoint(arm, seed)) for arm in ARMS for seed in PROTOCOL['seeds']}
    atomic_json(ROOT / 'summary.json', {'protocol': protocol, 'manifest': manifest, 'feature_sha256': hashes,
        'actual_source_video_coverage': coverage, 'results': results, 'contrasts': contrasts, 'interaction': interaction,
        'validation_selected_arm': max(results, key=lambda a: np.mean([r['calibrated_validation']['macro_f1_present'] for r in results[a]['runs']])),
        'verification': {'all_query_metadata_identical_within_training_group': True, 'all_nine_baseline_validation_scores_reproduced': True,
                         'baseline_checkpoints_offsets_and_all_features_unchanged': True, 'coverage80_actual_source_target_met': True,
                         'confirmation_disjoint_from_prior_340': True, 'only_three_new_heads_trained': True}})
    print('DONE feature coverage control', flush=True)


if __name__ == '__main__': main()
