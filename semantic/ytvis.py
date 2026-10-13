"""Deterministic VIS-style clips, masks, and source-pixel queries."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


NUM_CLASSES = 41  # Legacy YouTube-VIS default; OVIS uses 26 including background.
CLIP_FRAMES = 8
IMAGE_SIZE = 128


class VideoInstanceDataset:
    name = "video instance dataset"
    object_classes = 0

    def __init__(self, annotation_path: Path, frames_root: Path):
        if not annotation_path.is_file():
            raise FileNotFoundError(f"{self.name} annotation not found: {annotation_path}")
        self.frames_root = Path(frames_root)
        payload = json.loads(Path(annotation_path).read_text())
        self.videos = {int(v["id"]): v for v in payload["videos"]}
        self.categories = {int(c["id"]): c["name"] for c in payload["categories"]}
        if set(self.categories) != set(range(1, self.object_classes + 1)):
            raise ValueError(f"Expected {self.name} category IDs 1..{self.object_classes}")
        self.num_classes = self.object_classes + 1
        self.annotations: dict[int, list[dict]] = {}
        for ann in payload["annotations"]:
            self.annotations.setdefault(int(ann["video_id"]), []).append(ann)

    def frame_path(self, name: str) -> Path:
        for prefix in (self.frames_root, self.frames_root / "train", self.frames_root / "JPEGImages",
                       self.frames_root / "train_all_frames" / "JPEGImages"):
            candidate = prefix / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Frame {name!r} not found below {self.frames_root}")

    def select_manifest(self, count: int = 100, seed: int = 42) -> dict:
        if count != 100:
            raise ValueError("This experiment fixes exactly 100 clips for a 60/20/20 split")
        eligible = []
        for video_id, video in sorted(self.videos.items()):
            names = video["file_names"]
            if len(names) < CLIP_FRAMES:
                continue
            starts = range(len(names) - CLIP_FRAMES + 1)
            start = next((i for i in starts if any(
                i < len(a["segmentations"]) and a["segmentations"][i] is not None
                for a in self.annotations.get(video_id, [])
            )), None)
            if start is None:
                continue
            try:
                files = [str(self.frame_path(name)) for name in names[start:start + CLIP_FRAMES]]
            except FileNotFoundError:
                continue
            eligible.append({"video_id": video_id, "start": start, "frames": files})
        if len(eligible) < count:
            raise ValueError(f"Only {len(eligible)} eligible clips with images found; need {count}")
        random.Random(seed).shuffle(eligible)
        selected = eligible[:count]
        groups = {"train": selected[:60], "val": selected[60:80], "test": selected[80:100]}

        def source_categories(clip: dict) -> set[int]:
            frame = int(clip["start"])
            return {int(ann["category_id"]) for ann in self.annotations.get(int(clip["video_id"]), [])
                    if frame < len(ann["segmentations"]) and ann["segmentations"][frame] is not None}

        def categories(clips: list[dict]) -> set[int]:
            return set().union(*(source_categories(clip) for clip in clips))

        # Keep the seeded 100-video sample, but avoid validation/test classes
        # that the semantic head never sees in a training source frame.
        while missing := (categories(groups["val"] + groups["test"]) - categories(groups["train"])):
            swapped = False
            for split in ("val", "test"):
                for heldout_index, heldout in enumerate(groups[split]):
                    if not (source_categories(heldout) & missing):
                        continue
                    for train_index in range(len(groups["train"]) - 1, -1, -1):
                        candidate_train = groups["train"][:train_index] + groups["train"][train_index + 1:] + [heldout]
                        candidate_heldout = groups[split][:heldout_index] + groups[split][heldout_index + 1:] + [groups["train"][train_index]]
                        other_split = "test" if split == "val" else "val"
                        remaining = categories(candidate_heldout + groups[other_split]) - categories(candidate_train)
                        if len(remaining) < len(missing):
                            groups["train"][train_index], groups[split][heldout_index] = heldout, groups["train"][train_index]
                            swapped = True
                            break
                    if swapped:
                        break
                if swapped:
                    break
            if not swapped:
                raise ValueError(f"Cannot cover held-out source-frame categories {sorted(missing)} with training videos")
        return {"dataset": self.name, "seed": seed,
                "split_strategy": "seeded_shuffle_then_source_category_coverage",
                **groups,
                "categories": self.categories}

    def gt_label_map(self, clip: dict, local_frame: int, size: int = IMAGE_SIZE) -> np.ndarray:
        try:
            from pycocotools import mask as coco_mask
        except ImportError as exc:
            raise RuntimeError("Install pycocotools to decode instance masks") from exc
        if local_frame < 0 or local_frame >= CLIP_FRAMES:
            raise IndexError(local_frame)
        video_id = int(clip["video_id"])
        frame_idx = int(clip["start"]) + local_frame
        video = self.videos[video_id]
        labels = np.zeros((size, size), dtype=np.int16)
        for ann in self.annotations.get(video_id, []):
            if frame_idx >= len(ann["segmentations"]):
                continue
            rle = ann["segmentations"][frame_idx]
            if rle is None:
                continue
            decoded = coco_mask.decode(rle)
            if decoded.ndim == 3:
                decoded = decoded.any(axis=2)
            if decoded.shape != (int(video["height"]), int(video["width"])):
                raise ValueError(f"Mask size mismatch for video {video_id}")
            mask = np.asarray(Image.fromarray(decoded.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST), dtype=bool)
            overlap = mask & (labels != 0)
            labels[mask & (labels == 0)] = int(ann["category_id"])
            labels[overlap] = -1
        return labels


class YTVIS2019(VideoInstanceDataset):
    name = "YouTube-VIS 2019 train"
    object_classes = 40


class OVIS(VideoInstanceDataset):
    name = "OVIS train"
    object_classes = 25


def load_dataset(name: str, annotation_path: Path, frames_root: Path) -> VideoInstanceDataset:
    choices = {"ytvis2019": YTVIS2019, "ovis": OVIS}
    if name not in choices:
        raise ValueError(f"Unknown dataset {name!r}; choose one of {tuple(choices)}")
    return choices[name](annotation_path, frames_root)


def load_video(clip: dict, size: int = IMAGE_SIZE) -> torch.Tensor:
    frames = []
    for path in clip["frames"]:
        with Image.open(path) as source:
            image = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
    return torch.stack(frames).float().div_(255.0)


def sample_queries(labels: np.ndarray, per_frame: int, seed: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Sample half object / half other pixels; -1 is never treated as background."""
    if labels.ndim != 3:
        raise ValueError("Expected [T,H,W] labels")
    frames, height, width = labels.shape
    rng = np.random.default_rng(seed)
    selected = []
    for t in range(frames):
        foreground = np.argwhere(labels[t] > 0)
        background = np.argwhere(labels[t] == 0)
        for pool, quota in ((foreground, per_frame // 2), (background, per_frame - per_frame // 2)):
            if len(pool):
                picks = pool[rng.choice(len(pool), size=quota, replace=len(pool) < quota)]
                selected.extend((t, int(y), int(x)) for y, x in picks)
    if not selected:
        raise ValueError("No valid query pixels")
    t_src, y, x = (np.asarray(v) for v in zip(*selected))
    query = {
        "u": torch.tensor(x / max(width - 1, 1), dtype=torch.float32)[None],
        "v": torch.tensor(y / max(height - 1, 1), dtype=torch.float32)[None],
        "t_src": torch.tensor(t_src, dtype=torch.long)[None],
        "t_tgt": torch.tensor(rng.integers(0, frames, len(x)), dtype=torch.long)[None],
        "t_cam": torch.tensor(rng.integers(0, frames, len(x)), dtype=torch.long)[None],
    }
    targets = torch.tensor(labels[t_src, y, x], dtype=torch.long)
    xy = torch.tensor(np.stack([x, y, t_src], axis=-1), dtype=torch.long)
    return query, targets, xy
