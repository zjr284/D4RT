"""Preregistered frozen DINOv2 patch-feature vs frozen D4RT source-GT control."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from torch.nn import functional as F
from semantic.video_coverage_intervention import ROOT as PREVIOUS, extract
from semantic.ytvis import load_dataset, load_video
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler
from semantic.head_budget_probe import score
from semantic.background_calibration import pick_offset, evaluate, predictions, PROTOCOL as CALIBRATION
from semantic.fresh_video_confirmation import bootstrap_video_differences
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json, classification_metrics
from tools.load_opend4rt import load_model, DEFAULT_VARIANT

ROOT = Path('semantic_artifacts/semantic_feature_control')
VENDOR = Path('semantic_artifacts/vendor/dinov2')
WEIGHTS = Path('semantic_artifacts/weights/dinov2_vits14_pretrain.pth')
BASELINE = 'random80_source_gt'
PROTOCOL = {'version': 1, 'confirmation_seed': 20261015, 'confirmation_videos': 40,
            'training_videos': 80, 'validation_videos': 20, 'arms': ['d4rt', 'dinov2'],
            'baseline': 'frozen round11 validation-selected random80_source_gt; no repeated training',
            'dinov2': 'official ViT-S/14, frozen, final normalized patch tokens, no pretrained head',
            'dinov2_input': 'shared 128x128 RGB -> bicubic 224x224; ImageNet mean/std; no GT masks/crops/prompts',
            'patch_sampling': 'bilinear at 128-pixel centers, align_corners=False, border padding',
            'feature_dims': {'d4rt': 1280, 'dinov2': 384}, 'feature_storage': 'float16',
            'supervision': 'source-frame GT only, identical query metadata', 'head': 'linear',
            'steps': 2000, 'seeds': [0, 1, 2], 'batch': '128 foreground + 128 background uniform queries',
            'optimizer': 'AdamW lr=0.001 weight_decay=0.01',
            'checkpoint_selection': 'raw validation macro F1 every 50 steps; ties earlier checkpoint',
            'calibration_grid': CALIBRATION['offset_grid'], 'calibration_selection': CALIBRATION['offset_selection'],
            'frozen_d4rt_offset': -1.25, 'bootstrap_seed': 20261015, 'bootstrap_repetitions': 2000,
            'confirmation_never_selects_parameters': True, 'source_category_metadata_previously_inspected': True,
            'pretraining_ovis_image_overlap': 'unknown, cannot establish disjoint pretraining data'}


def aggregate_hash(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode()); digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


def source_path(arm, split, vid):
    if arm == 'dinov2': return ROOT / 'features' / arm / split / f'{vid}.npz'
    if split == 'confirmation': return ROOT / 'features' / 'd4rt' / split / f'{vid}.npz'
    return PREVIOUS / 'features' / split / f'{vid}.npz'


def freeze():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    previous = json.loads((PREVIOUS / 'summary.json').read_text())
    verified = json.loads((PREVIOUS / 'verification.json').read_text())
    assert file_hash(PREVIOUS / 'summary.json') == verified['summary_sha256']
    assert manifest['train'] == previous['manifest']['training_sets']['random80']
    assert manifest['val'] == previous['manifest']['val']
    assert manifest['eligible_model_unevaluated'] == 307 and len(manifest['excluded_video_ids']) == 300
    assert len(manifest['confirmation']) == 40 and not {c['video_id'] for c in manifest['confirmation']} & set(manifest['excluded_video_ids'])
    checkpoints = {}
    for seed in PROTOCOL['seeds']:
        checkpoints[str(seed)] = file_hash(PREVIOUS / BASELINE / 'runs' / f'linear_2000_seed{seed}.pt')
        assert checkpoints[str(seed)] == verified['checkpoint_sha256'][f'{BASELINE}_2000_{seed}']
    assert previous['validation_selected_arm'] == BASELINE
    assert previous['results'][BASELINE]['offset'] == PROTOCOL['frozen_d4rt_offset']
    protocol = dict(PROTOCOL, manifest_sha256=file_hash(ROOT / 'manifest.json'),
                    dinov2_weight_sha256=file_hash(WEIGHTS),
                    dinov2_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip(),
                    d4rt_backbone_sha256=file_hash(DEFAULT_VARIANT / 'opend4rt.ckpt'),
                    d4rt_model_config_sha256=file_hash(DEFAULT_VARIANT / 'model.yaml'),
                    frozen_baseline_checkpoint_sha256=checkpoints,
                    reused_d4rt_features_sha256={s: aggregate_hash([source_path('d4rt', s, c['video_id']) for c in manifest[s]]) for s in ('train', 'val')},
                    annotation_sha256=file_hash('semantic_artifacts/ovis/annotations_train.json'))
    path = ROOT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Frozen protocol/input mismatch')
    atomic_json(path, protocol)
    return manifest, protocol


def save_npz(path, **data):
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix('.partial.npz')
    np.savez_compressed(temp, **data); temp.replace(path)


def cache_new_d4rt(dataset, manifest):
    missing = [c for c in manifest['confirmation'] if not source_path('d4rt', 'confirmation', c['video_id']).exists()]
    if not missing: return
    model = load_model(DEFAULT_VARIANT / 'opend4rt.ckpt', DEFAULT_VARIANT / 'model.yaml').to('cuda', dtype=torch.bfloat16).eval()
    for clip in missing:
        labels = np.stack([dataset.gt_label_map(clip, t) for t in range(8)])
        data = extract(model, clip, labels)
        assert np.isfinite(data['decoded']).all()
        save_npz(source_path('d4rt', 'confirmation', clip['video_id']), **data)
        print('cached D4RT confirmation', clip['video_id'], flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


def patch_at_queries(tokens, xy, image_size=128):
    # Pixel centers match align_corners=False resize, and patch tokens represent patch centers.
    if tokens.ndim != 4 or xy.ndim != 2 or xy.shape[1] != 3 or len(xy) == 0: raise ValueError('Expected TxDxHxW and nonempty Nx3 coordinates')
    if np.any(xy[:, :2] < 0) or np.any(xy[:, :2] >= image_size) or np.any(xy[:, 2] < 0) or np.any(xy[:, 2] >= len(tokens)):
        raise ValueError('Query outside image/time range')
    result = torch.empty((len(xy), tokens.shape[1]), device=tokens.device, dtype=torch.float32)
    for t in np.unique(xy[:, 2]):
        rows = np.flatnonzero(xy[:, 2] == t)
        grid = torch.from_numpy(xy[rows, :2]).to(tokens.device, dtype=torch.float32)
        grid = ((grid + 0.5) / image_size * 2 - 1)[None, None]
        sampled = F.grid_sample(tokens[int(t):int(t)+1].float(), grid, mode='bilinear', padding_mode='border', align_corners=False)
        result[torch.as_tensor(rows, device=tokens.device)] = sampled[0, :, 0].T
    return result


def cache_dino(manifest):
    tasks = [(s, c) for s in ('train', 'val', 'confirmation') for c in manifest[s]
             if not source_path('dinov2', s, c['video_id']).exists()]
    if not tasks: return
    os.environ['XFORMERS_DISABLED'] = '1'; sys.path.insert(0, str(VENDOR.resolve()))
    from dinov2.hub.backbones import dinov2_vits14
    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    model = model.to('cuda').eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    for split, clip in tasks:
        with np.load(source_path('d4rt', split, clip['video_id'])) as p:
            data = {k: p[k].copy() for k in ('labels', 'xy', 't_tgt', 't_cam', 'video_id')}
        if split == 'train':
            mask = data['xy'][:, 2] == 0
            for k in ('labels', 'xy', 't_tgt', 't_cam'): data[k] = data[k][mask]
        video = load_video(clip).to('cuda')
        if split == 'train': video = video[:1]
        image = (F.interpolate(video, size=(224, 224), mode='bicubic', align_corners=False) - mean) / std
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            tokens = model.forward_features(image)['x_norm_patchtokens'].reshape(len(image), 16, 16, 384).permute(0, 3, 1, 2)
        with torch.inference_mode(): data['decoded'] = patch_at_queries(tokens, data['xy']).to(torch.float16).cpu().numpy()
        assert np.isfinite(data['decoded']).all()
        save_npz(source_path('dinov2', split, clip['video_id']), **data)
        print('cached DINOv2', split, clip['video_id'], flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()


def read_features(arm, split, manifest):
    clips = []
    for clip in sorted(manifest[split], key=lambda c: str(c['video_id'])):
        with np.load(source_path(arm, split, clip['video_id'])) as p: data = {k: p[k].copy() for k in p.files}
        if split == 'train' and arm == 'd4rt':
            mask = data['xy'][:, 2] == 0
            for k in ('decoded', 'labels', 'xy', 't_tgt', 't_cam'): data[k] = data[k][mask]
        assert data['decoded'].shape == (len(data['labels']), PROTOCOL['feature_dims'][arm])
        assert np.isfinite(data['decoded']).all()
        clips.append(data)
    return clips


def train_dino(seed, train, val, protocol):
    report_path = ROOT / 'dinov2' / 'runs' / f'linear_seed{seed}.json'
    checkpoint = report_path.with_suffix('.pt')
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report['protocol'] == protocol and file_hash(checkpoint) == report['checkpoint_sha256']
        return report
    torch.manual_seed(seed); model = SemanticHead(hidden_dim=384, num_classes=26).to('cuda')
    features = torch.from_numpy(np.concatenate([c['decoded'] for c in train])).float().to('cuda')
    labels = np.concatenate([c['labels'] for c in train]); times = np.concatenate([c['xy'][:, 2] for c in train])
    sampler = Sampler(labels, times, 'baseline'); generator = torch.Generator(device='cuda').manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best = -1; state = None; history = []; hist = torch.zeros(26, dtype=torch.long, device='cuda')
    for step in range(1, 2001):
        model.train(); fg, bg = sampler.draw(generator); idx = torch.cat([fg, bg])
        hist += torch.bincount(sampler.labels[idx], minlength=26)
        optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(model(features[idx]), sampler.labels[idx]); loss.backward(); optimizer.step()
        if step % 50 == 0:
            value = score(model, val); history.append({'step': step, 'val_macro_f1': value, 'loss': float(loss.detach())})
            if value > best: best = value; best_step = step; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert hist[0] == 2000 * 128 and hist[1:].sum() == 2000 * 128
    checkpoint.parent.mkdir(parents=True, exist_ok=True); temp = checkpoint.with_suffix('.partial.pt')
    torch.save({'model': state, 'seed': seed, 'steps': 2000, 'best_step': best_step, 'protocol': protocol}, temp); temp.replace(checkpoint)
    report = {'protocol': protocol, 'seed': seed, 'history': history, 'best_step': best_step, 'best_val': best,
              'sampled_class_counts': hist.cpu().tolist(), 'parameter_count': sum(p.numel() for p in model.parameters()), 'checkpoint_sha256': file_hash(checkpoint)}
    atomic_json(report_path, report); del model, features; gc.collect(); torch.cuda.empty_cache()
    print('trained DINOv2 head', seed, best_step, best, flush=True)
    return report


def infer(arm, seed, split, clips):
    checkpoint = PREVIOUS / BASELINE / 'runs' / f'linear_2000_seed{seed}.pt' if arm == 'd4rt' else ROOT / 'dinov2' / 'runs' / f'linear_seed{seed}.pt'
    sha = file_hash(checkpoint); path = ROOT / 'logits' / f'{arm}_{seed}_{split}.npz'
    if path.exists():
        with np.load(path) as p: package = {k: p[k].copy() for k in p.files}
        assert str(package['checkpoint_sha256']) == sha
        return package
    model = SemanticHead(hidden_dim=PROTOCOL['feature_dims'][arm], num_classes=26).to('cuda').eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
    with torch.inference_mode(): values = [model(torch.from_numpy(c['decoded']).float().to('cuda')).cpu().numpy() for c in clips]
    package = {'logits': np.concatenate(values), 'labels': np.concatenate([c['labels'] for c in clips]),
               'times': np.concatenate([c['xy'][:, 2] for c in clips]),
               'video_ids': np.concatenate([np.full(len(c['labels']), int(c['video_id']), dtype=np.int32) for c in clips]), 'checkpoint_sha256': np.array(sha)}
    save_npz(path, **package); del model; return package


def main():
    manifest, protocol = freeze()
    ds = load_dataset('ovis', Path('semantic_artifacts/ovis/annotations_train.json'), Path('semantic_artifacts/ovis'))
    cache_new_d4rt(ds, manifest); cache_dino(manifest)
    hashes = {f'{arm}_{split}': aggregate_hash([source_path(arm, split, c['video_id']) for c in manifest[split]])
              for arm in PROTOCOL['arms'] for split in ('train', 'val', 'confirmation')}
    file = ROOT / 'feature_hashes.json'
    if file.exists() and json.loads(file.read_text()) != hashes: raise ValueError('Feature hashes changed')
    atomic_json(file, hashes)
    # Metadata identity is checked for all three splits before head training or confirmation inference.
    for split in ('train', 'val', 'confirmation'):
        a = read_features('d4rt', split, manifest); b = read_features('dinov2', split, manifest)
        for x, y in zip(a, b):
            for key in ('labels', 'xy', 'video_id', 't_tgt', 't_cam'): assert np.array_equal(x[key], y[key])
        del a, b
    train = read_features('dinov2', 'train', manifest); val = read_features('dinov2', 'val', manifest)
    reports = [train_dino(seed, train, val, dict(protocol, feature_sha256=hashes)) for seed in PROTOCOL['seeds']]
    del train, val
    validation = {}; curves = {}; offsets = {}
    for arm in PROTOCOL['arms']:
        clips = read_features(arm, 'val', manifest)
        validation[arm] = [infer(arm, seed, 'val', clips) for seed in PROTOCOL['seeds']]
        if arm == 'd4rt': offsets[arm] = -1.25; curves[arm] = 'reused fixed validation calibration'
        else: offsets[arm], curves[arm] = pick_offset(validation[arm], protocol['calibration_grid'])
        del clips
    selection = {'offsets': offsets, 'validation_curves': curves, 'checkpoint_sha256': {str(r['seed']): r['checkpoint_sha256'] for r in reports}}
    file = ROOT / 'frozen_selection.json'
    if file.exists() and json.loads(file.read_text()) != selection: raise ValueError('Selection changed')
    atomic_json(file, selection)  # No new confirmation logits exist until all models and offsets are fixed.
    results = {}
    previous = json.loads((PREVIOUS / 'summary.json').read_text())
    for arm in PROTOCOL['arms']:
        clips = read_features(arm, 'confirmation', manifest); runs = []
        for seed in PROTOCOL['seeds']:
            package = infer(arm, seed, 'confirmation', clips); calibrated = evaluate(package, offsets[arm])
            validation_metrics = evaluate(validation[arm][seed], offsets[arm])
            if arm == 'd4rt': assert validation_metrics == previous['results'][BASELINE]['runs'][seed]['calibrated_validation']
            pred = predictions(package['logits'], offsets[arm]); per_video = {str(c['video_id']): classification_metrics(pred[package['video_ids'] == c['video_id']], package['labels'][package['video_ids'] == c['video_id']], 26) for c in clips}
            runs.append({'seed': seed, 'raw_confirmation': evaluate(package, 0), 'calibrated_confirmation': calibrated,
                         'calibrated_validation': validation_metrics, 'per_video': per_video})
            print('confirmed', arm, seed, calibrated['macro_f1_present'], flush=True)
        results[arm] = {'offset': offsets[arm], 'runs': runs, 'parameter_count': (PROTOCOL['feature_dims'][arm] + 1) * 26}
        del clips
    contrasts = {}
    for metric in ('macro_f1_present', 'foreground_macro_f1_present', 'foreground_accuracy'):
        values = [results['dinov2']['runs'][i]['calibrated_confirmation'][metric] - results['d4rt']['runs'][i]['calibrated_confirmation'][metric] for i in range(3)]
        contrasts[metric] = {'paired_seed_differences': values, 'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1))}
    differences = {str(c['video_id']): float(np.mean([results['dinov2']['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present'] - results['d4rt']['runs'][i]['per_video'][str(c['video_id'])]['macro_f1_present'] for i in range(3)])) for c in manifest['confirmation']}
    contrasts['per_video_differences'] = differences
    contrasts['video_bootstrap'] = bootstrap_video_differences(list(differences.values()), seed=20261015)
    validation_selected = max(results, key=lambda a: np.mean([r['calibrated_validation']['macro_f1_present'] for r in results[a]['runs']]))
    current_hashes = {f'{arm}_{split}': aggregate_hash([source_path(arm, split, c['video_id']) for c in manifest[split]])
                     for arm in PROTOCOL['arms'] for split in ('train', 'val', 'confirmation')}
    assert current_hashes == hashes
    _, current_protocol = freeze(); assert current_protocol == protocol
    atomic_json(ROOT / 'summary.json', {'protocol': protocol, 'manifest': manifest, 'feature_sha256': hashes, 'results': results,
        'contrasts': contrasts, 'validation_selected_arm': validation_selected,
        'verification': {'all_query_metadata_identical': True, 'frozen_d4rt_validation_scores_reproduced': True,
                         'confirmation_excludes_previously_processed_300': True, 'confirmation_after_model_and_offset_freeze': True,
                         'all_feature_and_backbone_hashes_unchanged': True,
                         'source_category_metadata_disclosed': True}})
    print('DONE independent semantic feature control', flush=True)


if __name__ == '__main__': main()
