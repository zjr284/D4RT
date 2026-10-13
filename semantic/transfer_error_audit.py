"""Descriptive source-GT fitting vs frozen cross-video errors, with no tuning."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from semantic.video_coverage_intervention import ROOT as PREVIOUS, training_union
from semantic.background_calibration import predictions, evaluate
from semantic.frozen_error_audit import confusion_counts, decompose
from semantic.model import SemanticHead
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json, classification_metrics

ROOT = Path('semantic_artifacts/transfer_error_audit')
PROTOCOL = {'version': 1, 'analysis': 'descriptive audit of round 11 frozen outputs',
            'seeds': [0, 1, 2], 'training_updates': 0, 'calibration_updates': 0,
            'training_evaluation': 'source-frame GT only, own 80 and common 60 training videos',
            'heldout_subsets': ['all_frames', 'source_frame', 'later_frames'],
            'matched_categories': 'foreground present in all four training-source, validation-source and confirmation-source subsets',
            'matched_f1': 'full-subset false positives retained; average class F1 on matched category IDs',
            'class_groups': {'low_training_recall_threshold': 0.5, 'transfer_collapse_confirmation_recall_threshold': 0.1},
            'top_foreground_confusions': 10, 'not_a_new_confirmation': True}


def describe(labels, predicted, videos):
    matrix = confusion_counts(labels, predicted)
    result = classification_metrics(predicted, labels, 26)
    result['confusion_matrix'] = matrix.tolist()
    result['error_partition'] = decompose(matrix)
    result['per_class'] = {}
    for c in range(26):
        support = int(matrix[c].sum()); tp = int(matrix[c, c]); fp = int(matrix[:, c].sum()) - tp
        result['per_class'][str(c)] = {'queries': support, 'independent_videos': len(np.unique(videos[labels == c])),
            'f1': 2 * tp / max(support + tp + fp, 1) if support else None,
            'recall': tp / support if support else None,
            'to_background_rate': float(matrix[c, 0] / support) if support and c else None,
            'wrong_foreground_rate': float((matrix[c, 1:].sum() - tp) / support) if support and c else None}
    pairs = [{'truth': c, 'predicted': p, 'count': int(matrix[c, p])}
             for c in range(1, 26) for p in range(1, 26) if c != p and matrix[c, p] > 0]
    result['top_foreground_confusions'] = sorted(pairs, key=lambda p: (-p['count'], p['truth'], p['predicted']))[:10]
    return result


def subset(package, mask, offset):
    if not np.any(mask):
        raise ValueError('Empty evaluation subset')
    labels = package['labels'][mask]; videos = package['video_ids'][mask]
    predicted = predictions(package['logits'][mask], offset)
    result = describe(labels, predicted, videos)
    result['per_video'] = {str(v): classification_metrics(predicted[videos == v], labels[videos == v], 26) for v in np.unique(videos)}
    return result


def group_classes(train, confirmation):
    groups = {'unsupported_in_training': [], 'low_training_fit': [], 'fitted_training_low_confirmation': [], 'other': []}
    for c, values in confirmation.items():
        if c == '0' or not values['queries']:
            continue
        source = train[c]
        if not source['queries']:
            group = 'unsupported_in_training'
        elif source['recall'] < 0.5:
            group = 'low_training_fit'
        elif values['recall'] < 0.1:
            group = 'fitted_training_low_confirmation'
        else:
            group = 'other'
        groups[group].append(int(c))
    return groups


def mean_description(runs):
    sample = runs[0]
    result = {k: float(np.mean([r[k] for r in runs])) for k in ('accuracy', 'macro_f1_present', 'foreground_macro_f1_present')}
    result['error_partition'] = {k: float(np.mean([r['error_partition'][k] for r in runs]))
                               if sample['error_partition'][k] is not None else None for k in sample['error_partition']}
    result['per_class'] = {c: {k: float(np.mean([r['per_class'][c][k] for r in runs])) if values[k] is not None else None
                             for k in values} for c, values in sample['per_class'].items()}
    matrix = np.mean([r['confusion_matrix'] for r in runs], axis=0)
    pairs = [{'truth': c, 'predicted': p, 'mean_count': float(matrix[c, p])}
             for c in range(1, 26) for p in range(1, 26) if c != p and matrix[c, p] > 0]
    result['top_foreground_confusions'] = sorted(pairs, key=lambda p: (-p['mean_count'], p['truth'], p['predicted']))[:10]
    return result


def input_hashes(previous):
    files = [PREVIOUS / 'summary.json', PREVIOUS / 'manifest.json', PREVIOUS / 'verification.json']
    files += list((PREVIOUS / 'features').glob('*/*.npz'))
    for arm in previous['results']:
        files += [PREVIOUS / arm / 'runs' / f'linear_2000_seed{seed}.pt' for seed in PROTOCOL['seeds']]
        files += list((PREVIOUS / arm / 'logits').glob('*.npz'))
    return {str(p): file_hash(p) for p in sorted(files)}


def infer_source(model, clips):
    values = []; labels = []; videos = []
    with torch.inference_mode():
        for clip in sorted(clips, key=lambda c: str(c['video_id'])):
            with np.load(PREVIOUS / 'features' / 'train' / f"{clip['video_id']}.npz") as p:
                mask = p['xy'][:, 2] == 0
                x = p['decoded'][mask]; y = p['labels'][mask]
            values.append(model(torch.from_numpy(x).float().to('cuda')).cpu().numpy())
            labels.append(y); videos.append(np.full(len(y), clip['video_id'], dtype=np.int32))
    y = np.concatenate(labels)
    return {'logits': np.concatenate(values), 'labels': y, 'video_ids': np.concatenate(videos), 'times': np.zeros(len(y), dtype=np.int16)}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((PREVIOUS / 'summary.json').read_text())
    verified = json.loads((PREVIOUS / 'verification.json').read_text())
    assert verified['summary_sha256'] == file_hash(PREVIOUS / 'summary.json')
    for arm in previous['results']:
        for seed in PROTOCOL['seeds']:
            assert verified['checkpoint_sha256'][f'{arm}_2000_{seed}'] == file_hash(PREVIOUS / arm / 'runs' / f'linear_2000_seed{seed}.pt')
    for split, clips in [('train', training_union(previous['manifest'])), ('val', previous['manifest']['val']),
                         ('confirmation', previous['manifest']['confirmation'])]:
        digest = hashlib.sha256()
        for clip in clips:
            path = PREVIOUS / 'features' / split / f"{clip['video_id']}.npz"
            digest.update(path.name.encode()); digest.update(bytes.fromhex(file_hash(path)))
        assert digest.hexdigest() == verified['feature_sha256'][split]
    hashes = input_hashes(previous)
    protocol = dict(PROTOCOL, offsets={a: r['offset'] for a, r in previous['results'].items()}, input_sha256=hashes)
    file = ROOT / 'protocol.json'
    if file.exists() and json.loads(file.read_text()) != protocol:
        raise ValueError('Frozen audit inputs or protocol changed')
    atomic_json(file, protocol)  # Freeze before any new model inference or diagnostic calculation.
    manifest = previous['manifest']; groups = manifest['training_sets']
    common_videos = {c['video_id'] for c in groups['random80']} & {c['video_id'] for c in groups['coverage80']}
    assert len(common_videos) == 60
    results = {}; reference = {}
    for arm, prior in previous['results'].items():
        group = arm.split('_')[0]; runs = []
        for seed in PROTOCOL['seeds']:
            model = SemanticHead(num_classes=26).to('cuda').eval()
            state = torch.load(PREVIOUS / arm / 'runs' / f'linear_2000_seed{seed}.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state['model']); offset = prior['offset']
            package = infer_source(model, groups[group]); del model
            key = ('train', group)
            metadata = tuple(package[k] for k in ('labels', 'video_ids', 'times'))
            if key not in reference: reference[key] = metadata
            else: assert all(np.array_equal(a, b) for a, b in zip(metadata, reference[key]))
            path = ROOT / 'logits' / f'{arm}_{seed}_train_source.npz'; path.parent.mkdir(exist_ok=True)
            temp = path.with_suffix('.partial.npz'); np.savez_compressed(temp, **package); temp.replace(path)
            run = {'seed': seed, 'subsets': {
                'train_source': subset(package, np.ones(len(package['labels']), dtype=bool), offset),
                'common_train_source': subset(package, np.isin(package['video_ids'], list(common_videos)), offset)}}
            common_key = ('train', 'common60')
            common_metadata = tuple(a[np.isin(package['video_ids'], list(common_videos))] for a in metadata)
            if common_key not in reference: reference[common_key] = common_metadata
            else: assert all(np.array_equal(a, b) for a, b in zip(common_metadata, reference[common_key]))
            for split in ('val', 'confirmation'):
                with np.load(PREVIOUS / arm / 'logits' / f'{arm}_{seed}_{split}.npz') as p:
                    package = {k: p[k].copy() for k in p.files}
                metadata = tuple(package[k] for k in ('labels', 'video_ids', 'times'))
                if split not in reference: reference[split] = metadata
                else: assert all(np.array_equal(a, b) for a, b in zip(metadata, reference[split]))
                assert evaluate(package, offset) == prior['runs'][seed]['calibrated_validation' if split == 'val' else 'calibrated_confirmation']
                for name, mask in [('all_frames', np.ones(len(package['labels']), dtype=bool)),
                                   ('source_frame', package['times'] == 0), ('later_frames', package['times'] > 0)]:
                    run['subsets'][f'{split}_{name}'] = subset(package, mask, offset)
            runs.append(run)
        means = {name: mean_description([r['subsets'][name] for r in runs]) for name in runs[0]['subsets']}
        results[arm] = {'offset': prior['offset'], 'runs': runs, 'means': means,
            'confirmation_class_fit_groups': group_classes(means['train_source']['per_class'], means['confirmation_all_frames']['per_class'])}
        print('audited', arm, flush=True)
    categories = set(range(1, 26))
    for result in results.values():
        for name in ('train_source', 'val_source_frame', 'confirmation_source_frame'):
            categories &= {int(c) for c, v in result['means'][name]['per_class'].items() if v['queries']}
    assert categories
    for result in results.values():
        result['matched_source_categories'] = sorted(categories)
        result['matched_source_metrics'] = {name: {metric: float(np.mean([result['means'][name]['per_class'][str(c)][metric] for c in categories]))
            for metric in ('recall', 'f1')} for name in ('train_source', 'val_source_frame', 'confirmation_source_frame')}
        result['matched_source_per_seed'] = [{name: {metric: float(np.mean([r['subsets'][name]['per_class'][str(c)][metric] for c in categories]))
            for metric in ('recall', 'f1')} for name in ('train_source', 'val_source_frame', 'confirmation_source_frame')} for r in result['runs']]
    if hashes != input_hashes(previous): raise ValueError('Frozen inputs changed during analysis')
    atomic_json(ROOT / 'summary.json', {'protocol': protocol, 'matched_source_categories': sorted(categories), 'results': results,
        'verification': {'all_24_previous_validation_confirmation_metrics_reproduced': True,
                         'all_features_and_checkpoints_match_round11_independent_verification': True,
                         'all_frozen_inputs_unchanged': True, 'source_gt_comparison_metadata_equal': True,
                         'common60_training_metadata_equal_across_all_arms': True, 'training_and_calibration_updates': 0}})
    print('DONE transfer error audit', flush=True)


if __name__ == '__main__':
    main()
