"""Independent query, checkpoint and score verification for the 2x2 control."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from semantic.feature_coverage_control import ROOT, ARMS, OLD_OFFSETS, PREVIOUS, SEMANTIC, checkpoint, feature_path, feature_hashes, read, freeze
from semantic.semantic_feature_control import PROTOCOL as FEATURE_PROTOCOL
from semantic.ytvis import load_dataset, sample_queries
from semantic.background_calibration import evaluate, predictions, pick_offset
from semantic.video_coverage_intervention import source_coverage
from semantic.frozen_error_audit import confusion_counts, decompose
from semantic.model import SemanticHead
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json, classification_metrics


def main():
    summary = json.loads((ROOT / 'summary.json').read_text()); manifest = summary['manifest']
    selection = json.loads((ROOT / 'frozen_selection.json').read_text())
    dataset = load_dataset('ovis', Path('semantic_artifacts/ovis/annotations_train.json'), Path('semantic_artifacts/ovis'))
    frozen_manifest, protocol = freeze(dataset)
    assert manifest == frozen_manifest and protocol == summary['protocol']
    assert feature_hashes(manifest) == summary['feature_sha256']
    for clip in manifest['confirmation']:
        maps = np.stack([dataset.gt_label_map(clip, t) for t in range(8)])
        q, y, xy = sample_queries(maps, per_frame=256, seed=42 + clip['video_id'])
        with np.load(feature_path('d4rt', 'confirmation', clip['video_id'])) as p:
            assert np.array_equal(p['labels'], y.numpy()) and np.array_equal(p['xy'], xy.numpy())
            for key in ('t_tgt', 't_cam'): assert np.array_equal(p[key], q[key][0].numpy())
    for group in ('random80', 'coverage80'):
        for split in ('train', 'val', 'confirmation'):
            a = read(f'{group}_d4rt', split, manifest); b = read(f'{group}_dinov2', split, manifest)
            for x, y in zip(a, b):
                for k in ('labels', 'xy', 'video_id', 't_tgt', 't_cam'): assert np.array_equal(x[k], y[k])
            if split == 'train':
                coverage = source_coverage(a)
                assert {str(k): v for k, v in coverage.items()} == summary['actual_source_video_coverage'][group]
                if group == 'coverage80': assert min(coverage.values()) >= 3
            del a, b
    diagnostics = {}; hashes = {}
    for arm in ARMS:
        group, feature = arm.split('_'); runs = []; validation = []
        for seed in range(3):
            path = checkpoint(arm, seed); sha = file_hash(path); hashes[f'{arm}_{seed}'] = sha
            assert sha == selection['checkpoint_sha256'][f'{arm}_{seed}']
            state = torch.load(path, map_location='cpu', weights_only=True)
            if arm in OLD_OFFSETS: assert sha == protocol['frozen_checkpoint_sha256'][f'{arm}_{seed}']
            else:
                report = json.loads(path.with_suffix('.json').read_text())
                assert report['checkpoint_sha256'] == sha
                assert report['protocol'] == dict(protocol, feature_sha256=summary['feature_sha256'])
                best = max(report['history'], key=lambda h: h['val_macro_f1'])
                assert report['best_step'] == best['step'] and report['best_val'] == best['val_macro_f1']
                hist = np.asarray(report['sampled_class_counts'])
                assert hist[0] == 256000 and hist[1:].sum() == 256000
                prior = json.loads((PREVIOUS / 'coverage80_source_gt' / 'runs' / f'linear_seed{seed}.json').read_text())
                assert np.array_equal(hist, np.asarray(prior['budgets']['2000']['sampling_histogram_time_class'])[0])
            model = SemanticHead(hidden_dim=FEATURE_PROTOCOL['feature_dims'][feature], num_classes=26).to('cuda').eval()
            model.load_state_dict(state['model']); offset = summary['results'][arm]['offset']; run = {'seed': seed, 'subsets': {}}
            for split in ('train', 'val', 'confirmation'):
                clips = read(arm, split, manifest)
                labels = np.concatenate([c['labels'] for c in clips]); times = np.concatenate([c['xy'][:, 2] for c in clips])
                videos = np.concatenate([np.full(len(c['labels']), int(c['video_id']), dtype=np.int32) for c in clips])
                with torch.inference_mode(): logits = np.concatenate([model(torch.from_numpy(c['decoded']).float().to('cuda')).cpu().numpy() for c in clips])
                package = {'logits': logits, 'labels': labels, 'times': times, 'video_ids': videos}
                if split != 'train':
                    with np.load(ROOT / arm / 'logits' / f'{feature}_{seed}_{split}.npz') as p:
                        assert np.allclose(logits, p['logits'], rtol=1e-6, atol=1e-6)
                        for k in ('labels', 'times', 'video_ids'): assert np.array_equal(package[k], p[k])
                        assert str(p['checkpoint_sha256']) == sha
                    result = summary['results'][arm]['runs'][seed]
                    assert evaluate(package, offset) == result['calibrated_validation' if split == 'val' else 'calibrated_confirmation']
                    assert evaluate(package, 0) == result['raw_validation' if split == 'val' else 'raw_confirmation']
                    if split == 'val': validation.append(package)
                subsets = [('all_frames', np.ones(len(labels), dtype=bool))]
                if split != 'train': subsets += [('source_frame', times == 0), ('later_frames', times > 0)]
                for name, mask in subsets:
                    predicted = predictions(logits[mask], offset)
                    metric = classification_metrics(predicted, labels[mask], 26)
                    metric['error_partition'] = decompose(confusion_counts(labels[mask], predicted))
                    run['subsets'][f'{split}_{name}'] = metric
                del clips
            if arm == 'coverage80_dinov2': assert evaluate(validation[-1], 0)['macro_f1_present'] == report['best_val']
            runs.append(run); del model
        if arm == 'coverage80_dinov2':
            offset, curve = pick_offset(validation, protocol['calibration_grid'])
            assert offset == summary['results'][arm]['offset'] == selection['offsets'][arm]
            assert curve == selection['new_arm_validation_curves'][arm]
        else: assert summary['results'][arm]['offset'] == OLD_OFFSETS[arm]
        diagnostics[arm] = {'runs': runs}
        print('verified', arm, flush=True)
    supported = {int(c) for c in diagnostics['random80_dinov2']['runs'][0]['subsets']['train_all_frames']['support'] if c != '0'}
    tested = {int(c) for c in summary['results']['random80_dinov2']['runs'][0]['calibrated_confirmation']['support'] if c != '0'}
    common = sorted(supported & tested)
    for arm in ARMS:
        diagnostics[arm]['common_training_supported_confirmation_foreground_f1'] = [float(np.mean([r['calibrated_confirmation']['per_class_f1'][str(c)] for c in common])) for r in summary['results'][arm]['runs']]
    assert feature_hashes(manifest) == summary['feature_sha256']
    atomic_json(ROOT / 'verification.json', {'summary_sha256': file_hash(ROOT / 'summary.json'), 'checkpoint_sha256': hashes,
        'diagnostics': diagnostics, 'common_random_training_supported_confirmation_classes': common,
        'confirmation_classes_missing_from_random_training': sorted(tested-supported),
        'verification': {'all_40_new_query_sets_reproduced': True, 'all_group_query_metadata_identical': True,
                         'all_24_heldout_logit_packages_recomputed': True, 'all_raw_calibrated_metrics_reproduced': True,
                         'all_nine_old_heads_and_offsets_unchanged': True,
                         'new_head_sampling_counts_match_old_coverage_gt': True,
                         'new_head_checkpoint_and_calibration_selected_from_validation_only': True,
                         'coverage_target_met_and_all_feature_hashes_unchanged': True}})
    print('DONE independent round14 verification', flush=True)


if __name__ == '__main__': main()
