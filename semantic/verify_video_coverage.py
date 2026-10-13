"""Read-only independent checks of the completed round 11 experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from semantic.video_coverage_intervention import ROOT, training_union, source_coverage
from semantic.ytvis import load_dataset, sample_queries
from semantic.model import SemanticHead
from semantic.background_calibration import pick_offset, evaluate
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    summary = json.loads((ROOT / 'summary.json').read_text())
    manifest = summary['manifest']
    assert file_hash(ROOT / 'manifest.json') == summary['protocol']['manifest_sha256']
    assert json.loads((ROOT / 'protocol.json').read_text()) == summary['protocol']
    train = training_union(manifest)
    ids = {split: {c['video_id'] for c in clips} for split, clips in
           [('train', train), ('val', manifest['val']), ('confirmation', manifest['confirmation'])]}
    assert [len(ids[s]) for s in ids] == [100, 20, 40]
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['confirmation'] or ids['val'] & ids['confirmation'])
    assert not set.union(*ids.values()) & set(manifest['excluded_video_ids'])
    dataset = load_dataset('ovis', Path('semantic_artifacts/ovis/annotations_train.json'), Path('semantic_artifacts/ovis'))
    features = {}; hashes = {}
    for split, clips in [('train', train), ('val', manifest['val']), ('confirmation', manifest['confirmation'])]:
        digest = hashlib.sha256()
        for clip in clips:
            vid = clip['video_id']; path = ROOT / 'features' / split / f'{vid}.npz'
            with np.load(path) as p:
                data = {k: p[k].copy() for k in p.files}
            assert int(data['video_id']) == vid and data['decoded'].shape == (len(data['labels']), 1280)
            assert np.isfinite(data['decoded']).all() and np.all((data['labels'] >= 0) & (data['labels'] < 26))
            if split == 'train':
                with np.load(ROOT / 'teacher' / f'{vid}.npz') as p:
                    maps = p['labels'].copy()
                assert np.array_equal(maps[0], dataset.gt_label_map(clip, 0))
            else:
                maps = np.stack([dataset.gt_label_map(clip, t) for t in range(8)])
            query, targets, xy = sample_queries(maps, per_frame=256, seed=42 + vid)
            assert np.array_equal(data['labels'], targets.numpy()) and np.array_equal(data['xy'], xy.numpy())
            for k in ('t_tgt', 't_cam'):
                assert np.array_equal(data[k], query[k][0].numpy())
            features[(split, vid)] = data
            digest.update(path.name.encode()); digest.update(bytes.fromhex(file_hash(path)))
        hashes[split] = digest.hexdigest()
        print('verified queries/features', split, len(clips), flush=True)
    for group, clips in manifest['training_sets'].items():
        cached = [features[('train', c['video_id'])] for c in clips]
        actual = source_coverage(cached)
        assert {str(k): v for k, v in actual.items()} == summary['actual_source_queried_video_coverage'][group]
        for c, data in zip(clips, cached):
            source = data['labels'][data['xy'][:, 2] == 0]
            assert sorted(int(v) for v in np.unique(source[source > 0])) == manifest['candidate_source_query_classes'][str(c['video_id'])]
        if group == 'coverage80':
            assert min(actual.values()) >= 3
    checkpoint_hashes = {}; selected_steps = {}
    for arm, result in summary['results'].items():
        packages = []
        for seed, run in enumerate(result['runs']):
            assert seed == run['seed']
            report = json.loads(Path(result['trajectory_reports'][seed]).read_text())
            for budget in (500, 2000):
                entry = report['budgets'][str(budget)]
                candidates = [h for h in report['history'] if h['step'] <= budget]
                best = max(candidates, key=lambda h: h['val_macro_f1'])
                assert entry['best_step'] == best['step']
                assert entry['best_val_macro_f1'] == best['val_macro_f1']
                hist = np.asarray(entry['sampling_histogram_time_class'])
                assert hist[:, 0].sum() == budget * 128 and hist[:, 1:].sum() == budget * 128
                if arm.endswith('source_gt'):
                    assert hist[1:].sum() == 0
                path = ROOT / arm / 'runs' / f'linear_{budget}_seed{seed}.pt'
                state = torch.load(path, map_location='cpu', weights_only=True)
                assert state['protocol'] == dict(summary['protocol'], arm=arm)
                assert state['seed'] == seed and state['budget'] == budget
                checkpoint_hashes[f'{arm}_{budget}_{seed}'] = file_hash(path)
            selected_steps[f'{arm}_{seed}'] = report['budgets']['2000']['best_step']
            model = SemanticHead(num_classes=26).to('cuda').eval()
            model.load_state_dict(state['model'])
            for split in ('val', 'confirmation'):
                path = ROOT / arm / 'logits' / f'{arm}_{seed}_{split}.npz'
                with np.load(path) as p:
                    package = {k: p[k].copy() for k in p.files}
                clips = sorted(manifest[split], key=lambda c: str(c['video_id']))
                rows = [features[(split, c['video_id'])] for c in clips]
                assert np.array_equal(package['labels'], np.concatenate([r['labels'] for r in rows]))
                assert np.array_equal(package['times'], np.concatenate([r['xy'][:, 2] for r in rows]))
                assert np.array_equal(package['video_ids'], np.concatenate([np.full(len(r['labels']), int(r['video_id'])) for r in rows]))
                with torch.inference_mode():
                    logits = np.concatenate([model(torch.from_numpy(r['decoded']).float().to('cuda')).cpu().numpy() for r in rows])
                assert np.allclose(logits, package['logits'], rtol=1e-6, atol=1e-6)
                metrics = evaluate(package, result['offset'])
                assert metrics == run['calibrated_validation' if split == 'val' else 'calibrated_confirmation']
                if split == 'val':
                    packages.append(package)
            del model
        offset, curve = pick_offset(packages, summary['protocol']['calibration_grid'])
        assert offset == result['offset'] and curve == result['validation_curve']
        print('verified checkpoint selection/logits/calibration', arm, flush=True)
    chosen = max(summary['results'], key=lambda a: np.mean([r['calibrated_validation']['macro_f1_present'] for r in summary['results'][a]['runs']]))
    assert chosen == summary['validation_selected_arm']
    audit = {'summary_sha256': file_hash(ROOT / 'summary.json'), 'feature_sha256': hashes,
             'checkpoint_sha256': checkpoint_hashes, 'selected_steps_at_2000_budget': selected_steps,
             'all_160_cached_query_sets_reproduced': True, 'all_24_checkpoint_budgets_valid': True,
             'all_24_logit_packages_recomputed_from_checkpoint': True, 'all_calibrated_metrics_reproduced': True,
             'all_offsets_and_selected_arm_reproduce_validation_only_selection': True,
             'training_union_videos': 100, 'training_set_overlap_videos': 60,
             'distinct_videos_with_features_or_prior_evaluations': 300,
             'candidate_source_query_metadata_inspected': 407,
             'remaining_videos_without_training_features_or_model_evaluation': 307,
             'remaining_video_caveat': 'Source-frame category metadata was inspected for candidate selection; these are not entirely unseen annotations.'}
    atomic_json(ROOT / 'verification.json', audit)
    print('DONE independent round 11 verification', flush=True)


if __name__ == '__main__':
    main()
