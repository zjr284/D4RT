"""Fixed-protocol background pseudo-label ablation, preserving first-round queries.

Run: python -m semantic.background_ablation
Artifacts are independent of the first round. Held-out training masks audit
pseudo labels only; they never select, correct, or filter training labels.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from semantic.teacher import propagate_one, SAM_CHECKPOINT_SHA256
from semantic.ytvis import load_dataset, load_video, sample_queries
from semantic.model import extract_query_features, SemanticHead
from semantic.run import atomic_json, read_cache, _train_head, classification_metrics
from tools.load_opend4rt import load_model, DEFAULT_VARIANT

ROOT = Path('semantic_artifacts/background_ablation')
OLD = Path('semantic_artifacts/features_ovis/sam')
PROTOCOL = {
    'version': 1, 'negative_logit_threshold': -2.0, 'boundary_pixels': 2,
    'background_queries_per_later_frame': 128, 'query_seed': 42,
    'seeds': [0, 1, 2], 'steps': 500, 'batch_foreground_background': [128, 128],
    'teacher_checkpoint_sha256': SAM_CHECKPOINT_SHA256,
    'preserve_original_train_queries': True, 'reuse_original_val_test': True,
    'audit_only_no_gt_filtering': True,
}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def teachers(dataset, manifest):
    from sam2.build_sam import build_sam2_video_predictor
    checkpoint = Path('semantic_artifacts/weights/sam2.1_hiera_tiny.pt')
    if fingerprint(checkpoint) != SAM_CHECKPOINT_SHA256:
        raise ValueError('Teacher checkpoint checksum mismatch')
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml', str(checkpoint), device='cuda')
    out = ROOT / 'teacher'; out.mkdir(parents=True, exist_ok=True)
    audit = []; total_correct = total_valid = total_pixels = 0
    for clip in manifest['train']:
        vid = clip['video_id']; target = out / f'{vid}.npz'
        if not target.exists():
            labels, metrics = propagate_one(predictor, dataset, clip, include_background=True)
            # Preserve exactly the first-round positives, not a regenerated approximation.
            with np.load(Path('semantic_artifacts/teacher_ovis') / f'{vid}.npz') as old:
                original = old['labels'].copy()
            added = (original == -1) & (labels == 0); added[0] = False
            original[added] = 0
            tmp = target.with_suffix('.partial.npz')
            np.savez_compressed(tmp, labels=original, added_background=added, metrics=json.dumps(metrics))
            tmp.replace(target)
        with np.load(target) as data:
            added = data['added_background']
        correct = valid = 0
        for t in range(1, 8):
            gt = dataset.gt_label_map(clip, t)
            good = added[t] & (gt >= 0)
            valid += int(good.sum()); correct += int((good & (gt == 0)).sum())
        count = int(added.sum())
        total_correct += correct; total_valid += valid; total_pixels += count
        audit.append({'video_id': vid, 'background_pixels': count, 'nonignored_gt_pixels': valid,
                      'correct_background_pixels': correct, 'precision': correct / valid if valid else None})
        print(f'teacher {vid}: background={count}, precision={correct / valid if valid else None}', flush=True)
    result = {'videos': audit, 'precision_on_nonignored_gt': total_correct / total_valid if total_valid else None,
              'candidate_pixels': total_pixels, 'nonignored_gt_pixels': total_valid,
              'later_frame_coverage': total_pixels / (60 * 7 * 128 * 128)}
    atomic_json(ROOT / 'teacher_audit.json', result)
    del predictor; gc.collect(); torch.cuda.empty_cache()
    return result


def caches(dataset, manifest):
    out = ROOT / 'features/sam'
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((OLD / 'metadata.json').read_text())
    metadata['background_ablation_protocol'] = PROTOCOL
    atomic_json(out / 'metadata.json', metadata)
    for split in ('val', 'test'):
        destination = out / split
        if not destination.exists(): destination.symlink_to((OLD / split).resolve(), target_is_directory=True)
        elif destination.resolve() != (OLD / split).resolve(): raise ValueError('Held-out cache must be unchanged')
    missing = [c for c in manifest['train'] if not (out / 'train' / f"{c['video_id']}.npz").exists()]
    if not missing: return out
    model = load_model(DEFAULT_VARIANT / 'opend4rt.ckpt', DEFAULT_VARIANT / 'model.yaml')
    model = model.to('cuda', dtype=torch.bfloat16).eval()
    for clip in missing:
        vid = clip['video_id']
        with np.load(OLD / 'train' / f'{vid}.npz') as package:
            original = {k: package[k].copy() for k in package.files}
        with np.load(ROOT / 'teacher' / f'{vid}.npz') as package:
            mask = package['added_background']
        destination = out / 'train' / f'{vid}.npz'; destination.parent.mkdir(exist_ok=True)
        if mask.any():
            labels = np.full(mask.shape, -1, dtype=np.int16); labels[mask] = 0
            query, targets, xy = sample_queries(labels, per_frame=256, seed=42 + int(vid))
            video = load_video(clip)[None].to('cuda', dtype=torch.bfloat16)
            memory = torch.from_numpy(original['memory']).to('cuda', dtype=torch.bfloat16)[None]
            chunks = {k: [] for k in ('decoded', 'prelast', 'rgb_patch', 'geometry')}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for start in range(0, len(targets), 64):
                    small = {k: v[:, start:start+64].to('cuda') for k, v in query.items()}
                    features = extract_query_features(model, video, small, memory)
                    for k in chunks: chunks[k].append(features[k][0].to(torch.float16).cpu().numpy())
            for k in chunks: original[k] = np.concatenate([original[k], *chunks[k]])
            original['labels'] = np.concatenate([original['labels'], targets.numpy().astype(np.int16)])
            original['xy'] = np.concatenate([original['xy'], xy.numpy().astype(np.int16)])
        tmp = destination.with_suffix('.partial.npz'); np.savez_compressed(tmp, **original); tmp.replace(destination)
        print(f'cache {vid}: {len(original["labels"])} queries', flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return out


def diagnostic(model, clips):
    predictions = []; labels = []; times = []
    with torch.inference_mode():
        for clip in clips:
            predictions.append(model(torch.from_numpy(clip['decoded']).float().to('cuda')).argmax(-1).cpu().numpy())
            labels.append(clip['labels']); times.append(clip['xy'][:, 2])
    prediction = np.concatenate(predictions); target = np.concatenate(labels); t = np.concatenate(times)
    result = classification_metrics(prediction, target, 26)
    result['predicted_class_counts'] = np.bincount(prediction, minlength=26).tolist()
    result['per_frame'] = {}
    for frame in range(8):
        selected = t == frame
        result['per_frame'][str(frame)] = classification_metrics(prediction[selected], target[selected], 26)
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = dict(PROTOCOL, manifest_sha256=fingerprint(Path('semantic_artifacts/manifest.json')))
    file = ROOT / 'protocol.json'
    if file.exists() and json.loads(file.read_text()) != protocol: raise ValueError('Protocol changed; use a fresh output directory')
    atomic_json(file, protocol)  # Freeze choices before consulting audit or test metrics.
    dataset = load_dataset('ovis', Path('semantic_artifacts/ovis/annotations_train.json'), Path('semantic_artifacts/ovis'))
    manifest = json.loads(Path('semantic_artifacts/manifest.json').read_text())
    assert [len(manifest[s]) for s in ('train', 'val', 'test')] == [60, 20, 20]
    ids = [{c['video_id'] for c in manifest[s]} for s in ('train', 'val', 'test')]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    audit = teachers(dataset, manifest)
    cache = caches(dataset, manifest)
    train = read_cache(cache, 'train'); val = read_cache(OLD, 'val'); test = read_cache(OLD, 'test')
    args = SimpleNamespace(num_classes=26, steps=500, cache=cache, out=ROOT / 'runs')
    reports = []
    for seed in PROTOCOL['seeds']:
        file = ROOT / 'runs' / f'sam_seed{seed}_result.json'
        if file.exists(): report = json.loads(file.read_text())
        else:
            report = _train_head(args, 'sam', seed, train, val, test)
            atomic_json(file, report)
        reports.append(report)
        print(f'train seed {seed}: macro F1={report["test"]["macro_f1_present"]}', flush=True)
    del train; gc.collect()
    comparisons = {'positive_only': [], 'positive_and_background': []}
    for variant, folder in [('positive_only', Path('semantic_artifacts/runs_ovis_balanced')),
                             ('positive_and_background', ROOT / 'runs')]:
        for seed in PROTOCOL['seeds']:
            checkpoint = torch.load(folder / f'sam_seed{seed}.pt', map_location='cpu', weights_only=True)
            model = SemanticHead(num_classes=26).to('cuda').eval(); model.load_state_dict(checkpoint['model'])
            comparisons[variant].append({'seed': seed, 'test': diagnostic(model, test)})
    delta = [comparisons['positive_and_background'][i]['test']['macro_f1_present'] -
             comparisons['positive_only'][i]['test']['macro_f1_present'] for i in range(3)]
    result = {'protocol': protocol, 'teacher_audit': audit, 'comparisons': comparisons,
              'macro_f1_deltas': delta, 'macro_f1_delta_mean': float(np.mean(delta)),
              'macro_f1_delta_std': float(np.std(delta, ddof=1)), 'new_training_runs': reports}
    atomic_json(ROOT / 'summary.json', result)
    print('DONE', json.dumps({'delta': delta, 'teacher_precision': audit['precision_on_nonignored_gt']}), flush=True)


if __name__ == '__main__': main()
