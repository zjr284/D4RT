"""Replay teacher/source metadata and frozen DINOv2 SAM supervision controls."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from semantic.dino_sam_control import ROOT, ARMS, coverage, checkpoint, path_resolver, feature_hashes, read, freeze
from semantic.semantic_feature_control import PROTOCOL as FEATURE_PROTOCOL
from semantic.video_coverage_intervention import training_union
from semantic.ytvis import load_dataset, sample_queries
from semantic.background_calibration import evaluate, predictions, pick_offset
from semantic.frozen_error_audit import confusion_counts, decompose
from semantic.model import SemanticHead
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json, classification_metrics


def queried_coverage(clips):
    hist = np.zeros((8, 26), dtype=np.int64); videos = {c: set() for c in range(1, 26)}
    for clip in clips:
        y = clip['labels']; times = clip['xy'][:, 2]; vid = int(clip['video_id'])
        np.add.at(hist, (times, y), 1)
        for c in np.unique(y[y > 0]): videos[int(c)].add(vid)
    return {'time_class_counts': hist.tolist(), 'per_class': {str(c): {'queries': int(hist[:, c].sum()),
        'independent_queried_videos': len(videos[c]), 'video_ids': sorted(videos[c])} for c in range(1, 26)}}


def main():
    summary = json.loads((ROOT / 'summary.json').read_text()); manifest = summary['manifest']
    selection = json.loads((ROOT / 'frozen_selection.json').read_text())
    ds = load_dataset('ovis', Path('semantic_artifacts/ovis/annotations_train.json'), Path('semantic_artifacts/ovis'))
    frozen_manifest, protocol = freeze(ds); assert manifest == frozen_manifest and protocol == summary['protocol']
    assert feature_hashes(manifest) == summary['feature_sha256']
    for clip in training_union(manifest):
        vid = clip['video_id']
        with np.load(path_resolver('dinov2', 'train', vid)) as p: data = {k: p[k].copy() for k in p.files}
        with np.load(path_resolver('d4rt', 'train', vid)) as p: previous = {k: p[k].copy() for k in p.files if k != 'decoded'}
        with np.load(coverage.PREVIOUS / 'teacher' / f'{vid}.npz') as p: maps = p['labels'].copy()
        assert np.array_equal(maps[0], ds.gt_label_map(clip, 0))
        for key in ('labels', 'xy', 't_tgt', 't_cam', 'video_id'): assert np.array_equal(data[key], previous[key])
        xy = data['xy']; assert np.array_equal(data['labels'], maps[xy[:, 2], xy[:, 1], xy[:, 0]])
        with np.load(coverage.feature_path('dinov2', 'train', vid)) as p:
            mask = xy[:, 2] == 0
            for k in ('decoded', 'labels', 'xy', 't_tgt', 't_cam'): assert np.array_equal(data[k][mask], p[k])
    for clip in manifest['confirmation']:
        maps = np.stack([ds.gt_label_map(clip, t) for t in range(8)])
        q, y, xy = sample_queries(maps, per_frame=256, seed=42 + clip['video_id'])
        with np.load(path_resolver('dinov2', 'confirmation', clip['video_id'])) as p:
            assert np.array_equal(p['labels'], y.numpy()) and np.array_equal(p['xy'], xy.numpy())
            for k in ('t_tgt', 't_cam'): assert np.array_equal(p[k], q[k][0].numpy())
    coverage_stats = {}
    for arm in ARMS:
        clips = read(arm, 'train', manifest); coverage_stats[arm] = queried_coverage(clips); del clips
    results = {}; hashes = {}; metadata_reference = {}
    for arm in ARMS:
        group, label = arm.split('_', 1); runs = []; validation = []
        for seed in range(3):
            path = checkpoint(arm, seed); sha = file_hash(path); hashes[f'{arm}_{seed}'] = sha
            assert sha == selection['checkpoint_sha256'][f'{arm}_{seed}']
            state = torch.load(path, map_location='cpu', weights_only=True)
            if label == 'source_gt': assert sha == protocol['frozen_gt_checkpoint_sha256'][f'{group}_{seed}']
            else:
                report = json.loads(path.with_suffix('.json').read_text())
                assert report['checkpoint_sha256'] == sha
                assert report['protocol'] == dict(protocol, arm=arm, feature_sha256=summary['feature_sha256'])
                best = max(report['history'], key=lambda h: h['val_macro_f1'])
                assert report['best_step'] == best['step'] and report['best_val'] == best['val_macro_f1']
                hist = np.asarray(report['sampled_class_counts']); assert hist[0] == 256000 and hist[1:].sum() == 256000
            model = SemanticHead(hidden_dim=384, num_classes=26).to('cuda').eval(); model.load_state_dict(state['model'])
            offset = summary['results'][arm]['offset']; run = {'seed': seed, 'subsets': {}}
            for split in ('val', 'confirmation'):
                clips = read(arm, split, manifest)
                y = np.concatenate([c['labels'] for c in clips]); times = np.concatenate([c['xy'][:, 2] for c in clips])
                videos = np.concatenate([np.full(len(c['labels']), int(c['video_id']), dtype=np.int32) for c in clips])
                meta = (y, times, videos)
                if split not in metadata_reference: metadata_reference[split] = meta
                else: assert all(np.array_equal(a, b) for a, b in zip(meta, metadata_reference[split]))
                with torch.inference_mode(): logits = np.concatenate([model(torch.from_numpy(c['decoded']).float().to('cuda')).cpu().numpy() for c in clips])
                package = {'logits': logits, 'labels': y, 'times': times, 'video_ids': videos}
                with np.load(ROOT / arm / 'logits' / f'dinov2_{seed}_{split}.npz') as p:
                    assert np.allclose(logits, p['logits'], rtol=1e-6, atol=1e-6)
                    for k in ('labels', 'times', 'video_ids'): assert np.array_equal(package[k], p[k])
                    assert str(p['checkpoint_sha256']) == sha
                saved = summary['results'][arm]['runs'][seed]
                assert evaluate(package, offset) == saved['calibrated_validation' if split == 'val' else 'calibrated_confirmation']
                assert evaluate(package, 0) == saved['raw_validation' if split == 'val' else 'raw_confirmation']
                if split == 'val': validation.append(package)
                for name, mask in [('all_frames', np.ones(len(y), dtype=bool)), ('source_frame', times == 0), ('later_frames', times > 0)]:
                    predicted = predictions(logits[mask], offset)
                    metric = classification_metrics(predicted, y[mask], 26)
                    metric['error_partition'] = decompose(confusion_counts(y[mask], predicted))
                    run['subsets'][f'{split}_{name}'] = metric
                del clips
            if label != 'source_gt': assert evaluate(validation[-1], 0)['macro_f1_present'] == report['best_val']
            runs.append(run); del model
        if label == 'source_gt': assert offset == -1.5
        else:
            selected, curve = pick_offset(validation, protocol['calibration_grid'])
            assert selected == offset == selection['offsets'][arm]
            assert curve == selection['validation_curves'][arm]
        results[arm] = {'runs': runs}; print('verified', arm, flush=True)
    assert feature_hashes(manifest) == summary['feature_sha256']
    atomic_json(ROOT / 'verification.json', {'summary_sha256': file_hash(ROOT / 'summary.json'), 'checkpoint_sha256': hashes,
        'queried_training_coverage': coverage_stats, 'diagnostics': results,
        'verification': {'all_100_sam_query_sets_match_teacher_and_old_cache': True,
                         'all_100_source_gt_feature_prefixes_identical_to_original': True,
                         'all_40_new_gt_query_sets_reproduced': True, 'all_24_heldout_logit_packages_recomputed': True,
                         'all_raw_and_calibrated_metrics_reproduced': True, 'all_six_gt_heads_offsets_unchanged': True,
                         'all_six_new_heads_sampling_selection_and_calibration_valid': True,
                         'all_evaluation_metadata_shared_between_arms': True, 'all_feature_hashes_unchanged': True}})
    print('DONE independent DINO SAM verification', flush=True)


if __name__ == '__main__': main()
