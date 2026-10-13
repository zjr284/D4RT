"""Offline SAM 2.1 mask propagation from a single annotated source frame.

Only training clips are accepted. Category IDs always come from the dataset;
SAM 2 provides masks, not names. Pixels without a confident positive mask
remain ignored (-1), preventing missed objects from becoming false background.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from semantic.ytvis import CLIP_FRAMES, IMAGE_SIZE, VideoInstanceDataset, load_dataset


SAM_CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"


def confident_labels(logits: torch.Tensor, categories: list[int], size: int = IMAGE_SIZE,
                     threshold: float = 2.0, margin: float = 2.0,
                     include_background: bool = False) -> np.ndarray:
    """Convert object-mask logits to category pixels, excluding thin boundaries."""
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 3 or len(logits) != len(categories):
        raise ValueError("Expected one [H,W] logit map per prompted object")
    result = np.full((size, size), -1, dtype=np.int16)
    if not categories:
        return result
    scaled = F.interpolate(logits[:, None].float(), size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    best, winner = scaled.max(dim=0)
    if len(categories) > 1:
        second = scaled.topk(2, dim=0).values[1]
        strong = (best > threshold) & ((best - second) > margin)
    else:
        strong = best > threshold
    # Drop a two-pixel band on the positive-mask boundary.
    strong = (-F.max_pool2d(-strong.float()[None, None], kernel_size=5, stride=1, padding=2)[0, 0] > 0.5)
    if include_background:
        negative = (scaled.max(dim=0).values < -threshold).float()[None, None]
        # Pad with zeros: image edges do not provide a complete safety band.
        negative = F.pad(negative, (2, 2, 2, 2), value=0)
        safe = -F.max_pool2d(-negative, kernel_size=5, stride=1)[0, 0] > 0.5
        result[safe.cpu().numpy()] = 0
    ids = torch.as_tensor(categories, device=logits.device, dtype=torch.int16)
    result[strong.cpu().numpy()] = ids[winner[strong]].cpu().numpy()
    return result


def propagate_one(predictor, dataset: VideoInstanceDataset, clip: dict, include_background: bool = False) -> tuple[np.ndarray, dict]:
    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:
        raise RuntimeError("Install pycocotools before running the teacher") from exc
    video_id = int(clip["video_id"])
    source_index = int(clip["start"])
    prompts = []
    for ann in dataset.annotations.get(video_id, []):
        if source_index < len(ann["segmentations"]) and ann["segmentations"][source_index] is not None:
            mask = coco_mask.decode(ann["segmentations"][source_index])
            if mask.ndim == 3:
                mask = mask.any(axis=2)
            if mask.any():
                prompts.append((int(ann["id"]), int(ann["category_id"]), mask.astype(bool)))
    if not prompts:
        raise ValueError(f"Video {video_id} has no source-frame instance mask")
    labels = np.full((CLIP_FRAMES, IMAGE_SIZE, IMAGE_SIZE), -1, dtype=np.int16)
    labels[0] = dataset.gt_label_map(clip, 0)
    category_by_object = {object_id: category for object_id, category, _ in prompts}
    scores = []
    with tempfile.TemporaryDirectory(prefix="d4rt_sam2_frames_") as temporary:
        folder = Path(temporary)
        for index, source in enumerate(clip["frames"]):
            (folder / f"{index:05d}.jpg").symlink_to(Path(source).resolve())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(str(folder), offload_video_to_cpu=True, offload_state_to_cpu=True)
            try:
                for object_id, _, mask in prompts:
                    predictor.add_new_mask(state, frame_idx=0, obj_id=object_id, mask=mask)
                for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
                    if frame_idx == 0 or frame_idx >= CLIP_FRAMES:
                        continue
                    logits = mask_logits.detach()
                    if logits.ndim == 4 and logits.shape[1] == 1:
                        logits = logits[:, 0]
                    categories = [category_by_object[int(object_id)] for object_id in object_ids]
                    labels[frame_idx] = confident_labels(logits, categories, include_background=include_background)
                    # Diagnostic only: never use held-out GT to accept/reject pseudo labels.
                    for row, object_id in enumerate(object_ids):
                        ann = next(a for a in dataset.annotations[video_id] if int(a["id"]) == int(object_id))
                        raw = ann["segmentations"][source_index + frame_idx]
                        if raw is None:
                            continue
                        target = coco_mask.decode(raw)
                        if target.ndim == 3:
                            target = target.any(axis=2)
                        predicted = F.interpolate(logits[row:row+1, None].float(), size=target.shape[:2],
                                                  mode="bilinear", align_corners=False)[0, 0] > 0
                        target = torch.as_tensor(target.astype(bool), device=predicted.device)
                        union = (predicted | target).sum().item()
                        if union:
                            scores.append(float((predicted & target).sum().item() / union))
            finally:
                predictor.reset_state(state)
    return labels, {"video_id": video_id, "prompted_objects": len(prompts),
                    "held_out_instance_iou": scores, "mean_iou": float(np.mean(scores)) if scores else None,
                    "pseudo_pixel_count": int((labels[1:] > 0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ovis", "ytvis2019"), default="ovis")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("semantic_artifacts/teacher"))
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("semantic_artifacts/weights/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--max-videos", type=int, default=60)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 2 teacher requires CUDA in this experiment")
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError("Install the official facebookresearch/sam2 package") from exc
    dataset = load_dataset(args.dataset, args.annotation, args.frames_root)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("dataset") != dataset.name:
        raise ValueError("Teacher dataset does not match manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.sam_checkpoint.is_file():
        raise FileNotFoundError(f"SAM 2.1 tiny checkpoint missing: {args.sam_checkpoint}")
    digest = hashlib.sha256()
    with args.sam_checkpoint.open("rb") as file:
        while block := file.read(4 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != SAM_CHECKPOINT_SHA256:
        raise RuntimeError("SAM 2.1 tiny checkpoint failed local SHA-256 verification")
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", str(args.sam_checkpoint), device="cuda"
    )
    for clip in manifest["train"][:args.max_videos]:
        target = args.out / f"{clip['video_id']}.npz"
        if target.exists():
            continue
        labels, metrics = propagate_one(predictor, dataset, clip)
        temporary = target.with_suffix(".partial.npz")
        np.savez_compressed(temporary, labels=labels, metrics=json.dumps(metrics))
        os.replace(temporary, target)
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
