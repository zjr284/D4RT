"""Prepare frozen D4RT features and run point-category experiments.

Run as ``python -m semantic.run`` from the repository root. Data and outputs
live below semantic_artifacts/ (ignored by Git). The three training seeds and
the video split are fixed so comparisons cannot silently use different data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from semantic.model import RGBPatchBaseline, SemanticHead, SemanticTail, extract_query_features, geometry_vector
from semantic.ytvis import CLIP_FRAMES, IMAGE_SIZE, VideoInstanceDataset, load_dataset, load_video, sample_queries
from tools.load_opend4rt import DEFAULT_VARIANT, load_model


ROOT = Path("semantic_artifacts")
SEEDS = (0, 1, 2)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def write_manifest(args) -> None:
    dataset = load_dataset(args.dataset, args.annotation, args.frames_root)
    manifest = dataset.select_manifest()
    atomic_json(args.manifest, manifest)
    print(f"Selected 60/20/20 videos: {args.manifest}")


def _cache_clip(model, dataset: VideoInstanceDataset, clip: dict, split: str,
                teacher_dir: Path | None, destination: Path, seed: int) -> None:
    if split == "train":
        source_gt = dataset.gt_label_map(clip, 0)
        if teacher_dir is None:
            labels = np.full((CLIP_FRAMES, IMAGE_SIZE, IMAGE_SIZE), -1, dtype=np.int16)
            labels[0] = source_gt
        else:
            path = teacher_dir / f"{clip['video_id']}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Teacher labels missing: {path}")
            with np.load(path, allow_pickle=False) as package:
                labels = package["labels"].copy()
            if labels.shape != (CLIP_FRAMES, IMAGE_SIZE, IMAGE_SIZE) or not np.array_equal(labels[0], source_gt):
                raise ValueError(f"Teacher labels/source GT mismatch for {clip['video_id']}")
    else:
        labels = np.stack([dataset.gt_label_map(clip, t) for t in range(CLIP_FRAMES)])
    query, targets, xy = sample_queries(labels, per_frame=256, seed=seed + int(clip["video_id"]))
    video = load_video(clip)[None].to(device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        memory = model.encode_video(video)
        chunks: dict[str, list[np.ndarray]] = {k: [] for k in ("decoded", "prelast", "rgb_patch", "geometry")}
        alternate_decoded = []
        alternate_prelast = []
        for first in range(0, len(targets), 64):
            small = {key: value[:, first:first+64].to("cuda") for key, value in query.items()}
            feature = extract_query_features(model, video, small, memory)
            for key in chunks:
                chunks[key].append(feature[key][0].to(dtype=torch.float16).cpu().numpy())
            if split != "train":
                variant = dict(small)
                variant["t_tgt"] = (small["t_tgt"] + 1) % CLIP_FRAMES
                variant["t_cam"] = (small["t_cam"] + 1) % CLIP_FRAMES
                alt = extract_query_features(model, video, variant, memory)
                alternate_decoded.append(alt["decoded"][0].to(dtype=torch.float16).cpu().numpy())
                alternate_prelast.append(alt["prelast"][0].to(dtype=torch.float16).cpu().numpy())
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {key: np.concatenate(values) for key, values in chunks.items()}
    data.update(memory=memory[0].to(dtype=torch.float16).cpu().numpy(),
                labels=targets.numpy().astype(np.int16), xy=xy.numpy().astype(np.int16),
                video_id=np.int64(clip["video_id"]))
    if alternate_decoded:
        data["alternate_decoded"] = np.concatenate(alternate_decoded)
        data["alternate_prelast"] = np.concatenate(alternate_prelast)
    temporary = destination.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **data)
    os.replace(temporary, destination)


def prepare_cache(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen feature extraction needs CUDA")
    dataset = load_dataset(args.dataset, args.annotation, args.frames_root)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("dataset") != dataset.name or manifest.get("categories") != {str(k): v for k, v in dataset.categories.items()}:
        raise ValueError("Cache dataset/categories do not match manifest")
    mode = "sam" if args.teacher_dir is not None else "gt_only"
    out = args.cache_root / mode
    metadata = {"manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "mode": mode, "checkpoint": str(DEFAULT_VARIANT / "opend4rt.ckpt"), "frames": CLIP_FRAMES,
                "image_size": IMAGE_SIZE, "query_seed": 42, "dataset": dataset.name,
                "num_classes": dataset.num_classes, "categories": dataset.categories}
    metadata_path = out / "metadata.json"
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text())
        if previous != json.loads(json.dumps(metadata)):
            raise ValueError(f"Existing cache metadata differs from this manifest/config: {metadata_path}")
    elif any(out.glob("*/*.npz")):
        raise ValueError(f"Cache has feature files but no metadata; refuse unsafe resume: {out}")
    atomic_json(metadata_path, metadata)
    model = load_model(DEFAULT_VARIANT / "opend4rt.ckpt", DEFAULT_VARIANT / "model.yaml")
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    for split in ("train", "val", "test"):
        for clip in manifest[split][:args.max_videos]:
            target = out / split / f"{clip['video_id']}.npz"
            if target.exists():
                continue
            _cache_clip(model, dataset, clip, split, args.teacher_dir, target, seed=42)
            print(f"Cached {split} video {clip['video_id']}: {target}", flush=True)


def read_cache(folder: Path, split: str) -> list[dict[str, np.ndarray]]:
    paths = sorted((folder / split).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No cached {split} videos in {folder}")
    clips = []
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            clips.append({key: file[key].copy() for key in file.files})
    return clips


def classification_metrics(prediction: np.ndarray, targets: np.ndarray, num_classes: int) -> dict:
    if not len(targets):
        raise ValueError("No evaluation points")
    support = np.bincount(targets, minlength=num_classes)
    f1 = {}
    for category in range(num_classes):
        if support[category] == 0:
            continue
        tp = int(((prediction == category) & (targets == category)).sum())
        fp = int(((prediction == category) & (targets != category)).sum())
        fn = int(((prediction != category) & (targets == category)).sum())
        f1[category] = 2 * tp / max(2 * tp + fp + fn, 1)
    foreground = [value for key, value in f1.items() if key != 0]
    return {"accuracy": float((prediction == targets).mean()),
            "macro_f1_present": float(np.mean(list(f1.values()))),
            "foreground_macro_f1_present": float(np.mean(foreground)) if foreground else None,
            "support": {str(k): int(v) for k, v in enumerate(support) if v},
            "per_class_f1": {str(k): float(v) for k, v in f1.items()}}


def evaluate_head(model: torch.nn.Module, clips: list[dict], feature_key: str) -> dict:
    model.eval()
    predictions, targets, agreements, failures = [], [], [], []
    with torch.inference_mode():
        for clip in clips:
            inputs = torch.from_numpy(clip[feature_key]).float().to("cuda")
            logits = model(inputs)
            predicted = logits.argmax(-1).cpu().numpy()
            predictions.append(predicted)
            targets.append(clip["labels"].astype(np.int64))
            confidence = logits.softmax(-1).max(-1).values.cpu().numpy()
            for index in np.flatnonzero(predicted != clip["labels"]):
                x, y, t = (int(value) for value in clip["xy"][index])
                failures.append({"video_id": int(clip["video_id"]), "x": x, "y": y, "t_src": t,
                                 "target": int(clip["labels"][index]), "predicted": int(predicted[index]),
                                 "confidence": float(confidence[index])})
            if feature_key == "decoded" and "alternate_decoded" in clip:
                alternative = model(torch.from_numpy(clip["alternate_decoded"]).float().to("cuda"))
                agreements.append(float((logits.argmax(-1) == alternative.argmax(-1)).float().mean()))
    result = classification_metrics(np.concatenate(predictions), np.concatenate(targets),
                                    model.classifier.out_features)
    if agreements:
        result["target_camera_invariance"] = float(np.mean(agreements))
    result["high_confidence_failures"] = sorted(failures, key=lambda item: -item["confidence"])[:25]
    return result


def _train_head(args, variant: str, seed: int, train_clips: list[dict],
                val_clips: list[dict], test_clips: list[dict]) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    feature_key = "rgb_patch" if variant == "rgb" else "decoded"
    model = (RGBPatchBaseline(num_classes=args.num_classes) if variant == "rgb"
             else SemanticHead(num_classes=args.num_classes)).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    if variant == "gt":
        parts = [(clip[feature_key][clip["xy"][:, 2] == 0], clip["labels"][clip["xy"][:, 2] == 0])
                 for clip in train_clips]
    else:
        parts = [(clip[feature_key], clip["labels"]) for clip in train_clips]
    features = torch.from_numpy(np.concatenate([part[0] for part in parts])).float().to("cuda")
    labels = torch.from_numpy(np.concatenate([part[1] for part in parts]).astype(np.int64)).to("cuda")
    if len(labels) == 0:
        raise ValueError("No training labels")
    foreground = torch.nonzero(labels > 0, as_tuple=True)[0]
    background = torch.nonzero(labels == 0, as_tuple=True)[0]
    if len(foreground) == 0 or len(background) == 0:
        raise ValueError("Balanced training needs both foreground and background queries")
    rng = torch.Generator(device="cuda").manual_seed(seed)
    best = -1.0
    best_state = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        model.train()
        index = torch.cat((foreground[torch.randint(len(foreground), (128,), generator=rng, device="cuda")],
                           background[torch.randint(len(background), (128,), generator=rng, device="cuda")]))
        loss = F.cross_entropy(model(features[index]), labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            report = evaluate_head(model, val_clips, feature_key)
            if report["macro_f1_present"] > best:
                best = report["macro_f1_present"]
                best_state = {key: tensor.cpu().clone() for key, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate_head(model, test_clips, feature_key)
    target = args.out / f"{variant}_seed{seed}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": variant, "seed": seed, "model": best_state,
                "cache": str(args.cache), "num_classes": args.num_classes,
                "sampling": "foreground_background_50_50"}, target)
    return {"variant": variant, "seed": seed, "best_val_macro_f1": best,
            "test": test, "seconds": time.perf_counter() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "checkpoint": str(target)}


def _lora_predictions(tail: SemanticTail, clips: list[dict]) -> dict:
    tail.eval()
    predictions, targets, differences, agreements, failures = [], [], [], [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for clip in clips:
            memory = torch.from_numpy(clip["memory"]).to("cuda", dtype=torch.bfloat16)[None]
            outputs = []
            geometry = []
            confidences = []
            for first in range(0, len(clip["labels"]), 64):
                before = torch.from_numpy(clip["prelast"][first:first+64]).to("cuda", dtype=torch.bfloat16)[None]
                out = tail(before, memory)
                outputs.append(out["semantic_logits"][0].argmax(-1).cpu().numpy())
                confidences.append(out["semantic_logits"][0].float().softmax(-1).max(-1).values.cpu().numpy())
                geometry.append(geometry_vector(out)[0].float().cpu().numpy())
            predicted = np.concatenate(outputs)
            predictions.append(predicted)
            targets.append(clip["labels"].astype(np.int64))
            confidence = np.concatenate(confidences)
            for index in np.flatnonzero(predicted != clip["labels"]):
                x, y, t = (int(value) for value in clip["xy"][index])
                failures.append({"video_id": int(clip["video_id"]), "x": x, "y": y, "t_src": t,
                                 "target": int(clip["labels"][index]), "predicted": int(predicted[index]),
                                 "confidence": float(confidence[index])})
            differences.append(float(np.abs(np.concatenate(geometry) - clip["geometry"].astype(np.float32)).mean()))
            if "alternate_prelast" in clip:
                alt_predictions = []
                for first in range(0, len(clip["labels"]), 64):
                    before = torch.from_numpy(clip["alternate_prelast"][first:first+64]).to("cuda", dtype=torch.bfloat16)[None]
                    alt_predictions.append(tail(before, memory)["semantic_logits"][0].argmax(-1).cpu().numpy())
                agreements.append(float((np.concatenate(outputs) == np.concatenate(alt_predictions)).mean()))
    result = classification_metrics(np.concatenate(predictions), np.concatenate(targets),
                                    tail.semantic_head.classifier.out_features)
    result["geometry_mean_absolute_delta"] = float(np.mean(differences))
    if agreements:
        result["target_camera_invariance"] = float(np.mean(agreements))
    result["high_confidence_failures"] = sorted(failures, key=lambda item: -item["confidence"])[:25]
    return result


def _train_lora(args, seed: int, train_clips: list[dict], val_clips: list[dict], test_clips: list[dict]) -> dict:
    torch.manual_seed(seed)
    published = load_model(DEFAULT_VARIANT / "opend4rt.ckpt", DEFAULT_VARIANT / "model.yaml")
    tail = SemanticTail(published, args.num_classes)
    del published
    head_file = args.out / f"sam_seed{seed}.pt"
    head_checkpoint = torch.load(head_file, map_location="cpu", weights_only=True)
    if head_checkpoint.get("num_classes") != args.num_classes:
        raise ValueError("SAM head class count does not match LoRA experiment")
    head_state = head_checkpoint["model"]
    tail.semantic_head.load_state_dict(head_state)
    tail = tail.to("cuda").eval()
    optimizer = torch.optim.AdamW((p for p in tail.parameters() if p.requires_grad), lr=1e-4, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    balanced_clips = [clip for clip in train_clips if np.any(clip["labels"] > 0) and np.any(clip["labels"] == 0)]
    if not balanced_clips:
        raise ValueError("LoRA training needs clips with foreground and background queries")
    best, best_state = -1.0, None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        clip = balanced_clips[int(rng.integers(len(balanced_clips)))]
        positive = np.flatnonzero(clip["labels"] > 0)
        negative = np.flatnonzero(clip["labels"] == 0)
        chosen = np.concatenate((rng.choice(positive, size=32, replace=len(positive) < 32),
                                 rng.choice(negative, size=32, replace=len(negative) < 32)))
        rng.shuffle(chosen)
        before = torch.from_numpy(clip["prelast"][chosen]).to("cuda", dtype=torch.bfloat16)[None]
        memory = torch.from_numpy(clip["memory"]).to("cuda", dtype=torch.bfloat16)[None]
        labels = torch.from_numpy(clip["labels"][chosen].astype(np.int64)).to("cuda")[None]
        target_geo = torch.from_numpy(clip["geometry"][chosen]).float().to("cuda")[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = tail(before, memory)
            semantic_loss = F.cross_entropy(output["semantic_logits"].float().reshape(-1, args.num_classes), labels.reshape(-1))
            geometry_loss = F.smooth_l1_loss(geometry_vector(output).float(), target_geo)
            loss = semantic_loss + 0.1 * geometry_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            report = _lora_predictions(tail, val_clips)
            if report["macro_f1_present"] > best:
                best = report["macro_f1_present"]
                best_state = {name: p.detach().cpu().clone() for name, p in tail.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for name, p in tail.named_parameters():
            if name in best_state:
                p.copy_(best_state[name].to(p.device))
    test = _lora_predictions(tail, test_clips)
    target = args.out / f"lora_seed{seed}.pt"
    torch.save({"variant": "lora", "seed": seed, "trainable": best_state,
                "base_checkpoint": str(DEFAULT_VARIANT / "opend4rt.ckpt"), "num_classes": args.num_classes,
                "sampling": "foreground_background_50_50"}, target)
    return {"variant": "lora", "seed": seed, "best_val_macro_f1": best,
            "test": test, "seconds": time.perf_counter() - started,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "checkpoint": str(target)}


def run_experiment(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Training needs CUDA")
    train = read_cache(args.cache, "train")
    val = read_cache(args.cache, "val")
    test = read_cache(args.cache, "test")
    if not args.pilot and (len(train) != 60 or len(val) != 20 or len(test) != 20):
        raise ValueError("Full experiment needs exactly 60/20/20 cached videos; use tests/test_semantic.py --pipeline for a smoke test")
    ids = [{int(clip["video_id"]) for clip in clips} for clips in (train, val, test)]
    if any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Cached video IDs overlap across train/val/test")
    metadata_path = args.cache / "metadata.json"
    if not metadata_path.is_file() or json.loads(metadata_path.read_text()).get("mode") != "sam":
        raise ValueError("Full comparison requires one shared SAM teacher feature cache")
    metadata = json.loads(metadata_path.read_text())
    args.num_classes = int(metadata["num_classes"])
    if any(np.any((clip["labels"] < 0) | (clip["labels"] >= args.num_classes))
           for clips in (train, val, test) for clip in clips):
        raise ValueError("Cached labels outside dataset category range")
    reports = []
    for seed in ((0,) if args.pilot else SEEDS):
        report = (_train_lora(args, seed, train, val, test) if args.variant == "lora"
                  else _train_head(args, args.variant, seed, train, val, test))
        reports.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    atomic_json(args.out / f"{args.variant}_results.json", {"pilot": args.pilot, "runs": reports,
                "test_accuracy_mean": float(np.mean([r["test"]["accuracy"] for r in reports])),
                "test_accuracy_std": float(np.std([r["test"]["accuracy"] for r in reports], ddof=1)) if len(reports) > 1 else None,
                "test_macro_f1_mean": float(np.mean([r["test"]["macro_f1_present"] for r in reports])),
                "test_macro_f1_std": float(np.std([r["test"]["macro_f1_present"] for r in reports], ddof=1)) if len(reports) > 1 else None})


def summarize(args) -> None:
    teacher = []
    for path in sorted(args.teacher_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as file:
            teacher.append(json.loads(str(file["metrics"].item())))
    ious = [float(value) for item in teacher for value in item["held_out_instance_iou"]]
    result = {"teacher_videos": len(teacher), "teacher_instances_evaluated": len(ious),
              "teacher_mean_held_out_iou": float(np.mean(ious)) if ious else None,
              "teacher_pseudo_pixels": sum(int(item["pseudo_pixel_count"]) for item in teacher),
              "teacher_low_coverage_videos_lt_1000_pixels": sum(int(item["pseudo_pixel_count"]) < 1000 for item in teacher),
              "experiments": {}}
    if args.annotation is not None or args.frames_root is not None or args.manifest is not None:
        if any(value is None for value in (args.annotation, args.frames_root, args.manifest)):
            raise ValueError("Teacher label audit needs --annotation, --frames-root and --manifest together")
        dataset = load_dataset(args.dataset, args.annotation, args.frames_root)
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("dataset") != dataset.name:
            raise ValueError("Report dataset does not match manifest")
        correct = evaluated_positive = gt_foreground = confident_positive = total_pixels = 0
        pseudo_support = np.zeros(dataset.num_classes, dtype=np.int64)
        for clip in manifest["train"]:
            path = args.teacher_dir / f"{clip['video_id']}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Teacher result missing: {path}")
            with np.load(path, allow_pickle=False) as file:
                pseudo = file["labels"][1:]
            if pseudo.shape != (CLIP_FRAMES - 1, IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Teacher result has unexpected shape: {path}")
            truth = np.stack([dataset.gt_label_map(clip, t) for t in range(1, CLIP_FRAMES)])
            positive = pseudo > 0
            valid = positive & (truth >= 0)
            correct += int((valid & (pseudo == truth)).sum())
            evaluated_positive += int(valid.sum())
            gt_foreground += int((truth > 0).sum())
            confident_positive += int(positive.sum())
            total_pixels += int(positive.size)
            pseudo_support += np.bincount(pseudo[positive], minlength=dataset.num_classes)
        result["teacher_confident_label_audit"] = {
            "pixel_precision_on_nonignored_gt": correct / evaluated_positive if evaluated_positive else None,
            "correct_class_recall_on_gt_objects": correct / gt_foreground if gt_foreground else None,
            "positive_labeled_fraction_of_later_frames": confident_positive / total_pixels if total_pixels else None,
            "pseudo_class_support": {str(k): int(v) for k, v in enumerate(pseudo_support) if v},
        }
    for name in ("gt", "sam", "rgb", "lora"):
        path = args.runs / f"{name}_results.json"
        if path.is_file():
            result["experiments"][name] = json.loads(path.read_text())
    if "gt" in result["experiments"] and "sam" in result["experiments"]:
        gt = result["experiments"]["gt"]["runs"]
        sam = result["experiments"]["sam"]["runs"]
        deltas = [sam[i]["test"]["macro_f1_present"] - gt[i]["test"]["macro_f1_present"]
                  for i in range(3)]
        result["sam_minus_gt_test_macro_f1_by_seed"] = deltas
        result["sam_minus_gt_test_macro_f1_mean"] = float(np.mean(deltas))
        result["sam_minus_gt_test_macro_f1_std"] = float(np.std(deltas, ddof=1))
    atomic_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "experiments"}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("manifest")
    select.add_argument("--dataset", choices=("ovis", "ytvis2019"), default="ovis")
    select.add_argument("--annotation", type=Path, required=True)
    select.add_argument("--frames-root", type=Path, required=True)
    select.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    cache = sub.add_parser("cache")
    cache.add_argument("--dataset", choices=("ovis", "ytvis2019"), default="ovis")
    cache.add_argument("--annotation", type=Path, required=True)
    cache.add_argument("--frames-root", type=Path, required=True)
    cache.add_argument("--manifest", type=Path, default=ROOT / "manifest.json")
    cache.add_argument("--teacher-dir", type=Path)
    cache.add_argument("--cache-root", type=Path, default=ROOT / "features")
    cache.add_argument("--max-videos", type=int, default=100)
    train = sub.add_parser("train")
    train.add_argument("--cache", type=Path, default=ROOT / "features" / "sam")
    train.add_argument("--variant", choices=("gt", "sam", "rgb", "lora"), required=True)
    train.add_argument("--steps", type=int, default=500)
    train.add_argument("--pilot", action="store_true", help="Allow a partial cache; run seed 0 only and mark results as pilot")
    train.add_argument("--out", type=Path, default=ROOT / "runs")
    report = sub.add_parser("report")
    report.add_argument("--dataset", choices=("ovis", "ytvis2019"), default="ovis")
    report.add_argument("--annotation", type=Path)
    report.add_argument("--frames-root", type=Path)
    report.add_argument("--manifest", type=Path)
    report.add_argument("--teacher-dir", type=Path, default=ROOT / "teacher")
    report.add_argument("--runs", type=Path, default=ROOT / "runs")
    report.add_argument("--out", type=Path, default=ROOT / "summary.json")
    args = parser.parse_args()
    if args.command == "manifest":
        write_manifest(args)
    elif args.command == "cache":
        prepare_cache(args)
    elif args.command == "train":
        run_experiment(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
