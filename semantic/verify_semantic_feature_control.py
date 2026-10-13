"""Independently reload heads, replay queries and check round 13 frozen results."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from semantic.semantic_feature_control import ROOT, PREVIOUS, BASELINE, PROTOCOL, source_path, read_features, aggregate_hash
from semantic.ytvis import load_dataset, sample_queries
from semantic.background_calibration import evaluate, predictions, pick_offset
from semantic.frozen_error_audit import decompose, confusion_counts
from semantic.model import SemanticHead
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json, classification_metrics


def main():
    summary = json.loads((ROOT / 'summary.json').read_text()); manifest = summary['manifest']
    selection = json.loads((ROOT / 'frozen_selection.json').read_text())
    assert summary['protocol'] == json.loads((ROOT / 'protocol.json').read_text())
    assert summary['protocol']['manifest_sha256'] == file_hash(ROOT / 'manifest.json')
    assert len(manifest['train']) == 80 and len(manifest['val']) == 20 and len(manifest['confirmation']) == 40
    sets = [{c['video_id'] for c in manifest[s]} for s in ('train', 'val', 'confirmation')]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert not sets[2] & set(manifest['excluded_video_ids'])
    dataset = load_dataset('ovis', Path('semantic_artifacts/ovis/annotations_train.json'), Path('semantic_artifacts/ovis'))
    # Confirmation queries independently replay from original GT rather than feature-label metadata.
    for clip in manifest['confirmation']:
        maps = np.stack([dataset.gt_label_map(clip, t) for t in range(8)])
        queries, targets, xy = sample_queries(maps, per_frame=256, seed=42 + clip['video_id'])
        with np.load(source_path('d4rt', 'confirmation', clip['video_id'])) as p:
            assert np.array_equal(p['labels'], targets.numpy()) and np.array_equal(p['xy'], xy.numpy())
            assert np.array_equal(p['t_tgt'], queries['t_tgt'][0].numpy())
            assert np.array_equal(p['t_cam'], queries['t_cam'][0].numpy())
    hashes = {f'{a}_{s}': aggregate_hash([source_path(a, s, c['video_id']) for c in manifest[s]])
              for a in PROTOCOL['arms'] for s in ('train', 'val', 'confirmation')}
    assert hashes == summary['feature_sha256']
    for split in ('train', 'val', 'confirmation'):
        a = read_features('d4rt', split, manifest); b = read_features('dinov2', split, manifest)
        for x, y in zip(a, b):
            for k in ('labels', 'xy', 'video_id', 't_tgt', 't_cam'): assert np.array_equal(x[k], y[k])
        del a, b
    diagnostics = {}; checkpoint_hashes = {}; val_packages = {}
    for arm in PROTOCOL['arms']:
        diagnostic_runs = []; val_packages[arm] = []
        for seed in PROTOCOL['seeds']:
            ckpt = PREVIOUS / BASELINE / 'runs' / f'linear_2000_seed{seed}.pt' if arm == 'd4rt' else ROOT / 'dinov2' / 'runs' / f'linear_seed{seed}.pt'
            sha = file_hash(ckpt); checkpoint_hashes[f'{arm}_{seed}'] = sha
            state = torch.load(ckpt, map_location='cpu', weights_only=True)
            if arm == 'd4rt': assert sha == summary['protocol']['frozen_baseline_checkpoint_sha256'][str(seed)]
            else:
                report = json.loads(ckpt.with_suffix('.json').read_text())
                assert sha == report['checkpoint_sha256'] == selection['checkpoint_sha256'][str(seed)]
                assert report['best_step'] == max(report['history'], key=lambda h: h['val_macro_f1'])['step']
                assert report['best_val'] == max(h['val_macro_f1'] for h in report['history'])
                hist = np.array(report['sampled_class_counts'])
                assert hist[0] == 256000 and hist[1:].sum() == 256000
                baseline_report = json.loads((PREVIOUS / BASELINE / 'runs' / f'linear_seed{seed}.json').read_text())
                baseline_hist = np.asarray(baseline_report['budgets']['2000']['sampling_histogram_time_class'])
                assert baseline_hist[1:].sum() == 0 and np.array_equal(hist, baseline_hist[0])
            model = SemanticHead(hidden_dim=PROTOCOL['feature_dims'][arm], num_classes=26).to('cuda').eval()
            model.load_state_dict(state['model']); offset = summary['results'][arm]['offset']
            run = {'seed': seed, 'subsets': {}}
            for split in ('train', 'val', 'confirmation'):
                clips = read_features(arm, split, manifest)
                labels = np.concatenate([c['labels'] for c in clips]); times = np.concatenate([c['xy'][:, 2] for c in clips])
                videos = np.concatenate([np.full(len(c['labels']), int(c['video_id']), dtype=np.int32) for c in clips])
                with torch.inference_mode(): logits = np.concatenate([model(torch.from_numpy(c['decoded']).float().to('cuda')).cpu().numpy() for c in clips])
                package = {'logits': logits, 'labels': labels, 'times': times, 'video_ids': videos}
                if split != 'train':
                    with np.load(ROOT / 'logits' / f'{arm}_{seed}_{split}.npz') as p:
                        for k in ('labels', 'times', 'video_ids'): assert np.array_equal(package[k], p[k])
                        assert np.allclose(logits, p['logits'], rtol=1e-6, atol=1e-6)
                        assert str(p['checkpoint_sha256']) == sha
                    expected = summary['results'][arm]['runs'][seed]['calibrated_validation' if split == 'val' else 'calibrated_confirmation']
                    assert evaluate(package, offset) == expected
                    if split == 'confirmation': assert evaluate(package, 0) == summary['results'][arm]['runs'][seed]['raw_confirmation']
                    else:
                        val_packages[arm].append(package)
                        run['raw_validation'] = evaluate(package, 0)
                        if arm == 'dinov2': assert abs(run['raw_validation']['macro_f1_present'] - report['best_val']) < 1e-12
                masks = [('all_frames', np.ones(len(labels), dtype=bool))]
                if split != 'train': masks += [('source_frame', times == 0), ('later_frames', times > 0)]
                for name, mask in masks:
                    predicted = predictions(logits[mask], offset)
                    result = classification_metrics(predicted, labels[mask], 26)
                    result['error_partition'] = decompose(confusion_counts(labels[mask], predicted))
                    run['subsets'][f'{split}_{name}'] = result
                del clips
            diagnostic_runs.append(run); del model
        if arm == 'dinov2':
            offset, curve = pick_offset(val_packages[arm], PROTOCOL['calibration_grid'])
            assert offset == summary['results'][arm]['offset'] == selection['offsets'][arm]
            assert curve == selection['validation_curves'][arm]
        diagnostics[arm] = {'runs': diagnostic_runs}
        print('verified all head outputs and diagnostics', arm, flush=True)
    support = set(int(c) for c in diagnostics['d4rt']['runs'][0]['subsets']['train_all_frames']['support'] if c != '0')
    tested = set(int(c) for c in summary['results']['d4rt']['runs'][0]['calibrated_confirmation']['support'] if c != '0')
    common = sorted(support & tested)
    for arm in PROTOCOL['arms']:
        diagnostics[arm]['common_supported_confirmation_foreground_f1_by_seed'] = [float(np.mean([r['calibrated_confirmation']['per_class_f1'][str(c)] for c in common])) for r in summary['results'][arm]['runs']]
    chosen = max(summary['results'], key=lambda a: np.mean([r['calibrated_validation']['macro_f1_present'] for r in summary['results'][a]['runs']]))
    assert chosen == summary['validation_selected_arm']
    atomic_json(ROOT / 'verification.json', {'summary_sha256': file_hash(ROOT / 'summary.json'), 'checkpoint_sha256': checkpoint_hashes,
        'feature_sha256': hashes, 'diagnostics': diagnostics, 'common_training_supported_confirmation_classes': common,
        'test_classes_without_training_source_support': sorted(tested - support),
        'verification': {'all_40_new_query_sets_replayed_from_gt': True, 'all_140_video_metadata_identical_between_arms': True,
                         'all_12_heldout_logit_packages_recomputed': True, 'all_raw_and_calibrated_scores_reproduced': True,
                         'all_dinov2_sampling_budgets_and_validation_selection_valid': True,
                         'class_sampling_counts_identical_to_baseline_for_every_seed': True,
                         'baseline_heads_match_original_frozen_hashes': True}})
    print('DONE independent round 13 verification', flush=True)


if __name__ == '__main__': main()
