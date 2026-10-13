"""Run with .venv/bin/python tests/test_semantic.py [--integration]."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/open-d4rt"))

from semantic.model import GEOMETRY_KEYS, LoRALinear, SemanticD4RT, SemanticTail, geometry_vector
from semantic.teacher import confident_labels, propagate_one
from semantic.ytvis import OVIS, YTVIS2019, load_dataset, sample_queries


def fixture(directory: Path, video_count: int = 1, dataset_name: str = "ytvis2019") -> tuple[YTVIS2019 | OVIS, list[dict]]:
    images = directory / ("train" if dataset_name == "ovis" else "JPEGImages")
    images.mkdir()
    frame_names = []
    masks = []
    for t in range(8):
        name = f"{t:05d}.jpg"
        frame_names.append(name)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[12:48, 10 + t:45 + t] = 1
        picture = np.zeros((64, 64, 3), dtype=np.uint8)
        picture[mask.astype(bool)] = (235, 80, 30)
        Image.fromarray(picture).save(images / name)
        rle = coco_mask.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        masks.append(rle)
    videos = [{"id": i + 1, "width": 64, "height": 64, "file_names": frame_names}
              for i in range(video_count)]
    annotations = [{"id": i + 1, "video_id": i + 1, "category_id": 1,
                    "segmentations": masks} for i in range(video_count)]
    payload = {"videos": videos, "annotations": annotations,
               "categories": [{"id": i, "name": f"class_{i}"}
                              for i in range(1, 26 if dataset_name == "ovis" else 41)]}
    annotation = directory / "train.json"
    annotation.write_text(json.dumps(payload))
    dataset = load_dataset(dataset_name, annotation, directory)
    clips = [{"video_id": i + 1, "start": 0,
              "frames": [str(images / name) for name in frame_names]} for i in range(video_count)]
    return dataset, clips


def test_manifest_and_labels() -> None:
    with tempfile.TemporaryDirectory() as folder:
        data, clips = fixture(Path(folder), 100)
        manifest = data.select_manifest()
        assert [len(manifest[x]) for x in ("train", "val", "test")] == [60, 20, 20]
        ids = [int(clip["video_id"]) for split in ("train", "val", "test") for clip in manifest[split]]
        assert len(set(ids)) == 100
        label = data.gt_label_map(clips[0], 0)
        assert set(np.unique(label)) == {0, 1}
        stack = np.stack([label, np.full_like(label, -1)])
        query, targets, xy = sample_queries(stack, 64, 7)
        assert len(targets) == 64 and (targets >= 0).all()
        assert (xy[:, 2] == 0).all()
        assert query["u"].min() >= 0 and query["u"].max() <= 1
        # Category is fixed to the source pixel, irrespective of target/camera time.
        old_targets = targets.clone()
        query["t_tgt"] = (query["t_tgt"] + 1) % 8
        query["t_cam"] = (query["t_cam"] + 1) % 8
        assert torch.equal(targets, old_targets)


def test_ovis_manifest_and_categories() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        dataset, clips = fixture(root, 100, "ovis")
        manifest = dataset.select_manifest()
        assert manifest["dataset"] == "OVIS train"
        assert [len(manifest[x]) for x in ("train", "val", "test")] == [60, 20, 20]
        assert dataset.num_classes == 26
        assert dataset.frame_path("00000.jpg") == root / "train/00000.jpg"
        assert set(np.unique(dataset.gt_label_map(clips[0], 0))) == {0, 1}
        from semantic.model import SemanticHead, RGBPatchBaseline
        assert SemanticHead(num_classes=26)(torch.zeros(1, 2, 1280)).shape == (1, 2, 26)
        assert RGBPatchBaseline(num_classes=26)(torch.zeros(1, 2, 243)).shape == (1, 2, 26)


def test_heldout_category_gets_training_source_example() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        dataset, _ = fixture(root, 100, "ovis")
        original = dataset.select_manifest()
        rare_id = original["val"][0]["video_id"]
        annotation = root / "train.json"
        payload = json.loads(annotation.read_text())
        next(ann for ann in payload["annotations"] if ann["video_id"] == rare_id)["category_id"] = 2
        annotation.write_text(json.dumps(payload))
        covered = OVIS(annotation, root).select_manifest()
        assert covered == OVIS(annotation, root).select_manifest()
        assert rare_id in {clip["video_id"] for clip in covered["train"]}
        assert {clip["video_id"] for split in ("train", "val", "test") for clip in covered[split]} == set(range(1, 101))


def test_teacher_rejects_uncertainty() -> None:
    logits = torch.full((2, 128, 128), -5.0)
    logits[0, 15:60, 15:60] = 6.0
    logits[1, 65:110, 65:110] = 6.0
    labels = confident_labels(logits, [1, 2])
    assert labels[35, 35] == 1 and labels[85, 85] == 2
    assert labels[0, 0] == -1 and labels[15, 15] == -1
    background = confident_labels(logits, [1, 2], include_background=True)
    assert background[8, 8] == 0
    assert background[0, 0] == -1  # Complete safety band required at image edges.
    assert background[35, 35] == 1 and background[85, 85] == 2
    assert background[15, 15] == -1  # Uncertain object boundary stays ignored.
    ambiguous = torch.full((2, 128, 128), -5.0)
    ambiguous[1, 30:50, 30:50] = -1.0
    assert confident_labels(ambiguous, [1, 2], include_background=True)[40, 40] == -1
    assert (confident_labels(torch.empty(0, 128, 128), [], include_background=True) == -1).all()
    logits[1, 25:35, 25:35] = 5.8
    assert confident_labels(logits, [1, 2])[30, 30] == -1


def test_lora_tail_and_reload() -> None:
    from src.model.decoder import IndependentQueryDecoder
    from src.model.heads import D4RTHeads

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = IndependentQueryDecoder(hidden_dim=16, num_layers=3, num_heads=4, mlp_ratio=2)
            self.heads = D4RTHeads(hidden_dim=16)

    fake = Fake().eval()
    tail = SemanticTail(fake).eval()
    baseline = fake.decoder.out_norm(fake.decoder.blocks[-1](fake.decoder.blocks[-2](
        torch.randn(1, 5, 16), torch.randn(1, 7, 16)), torch.randn(1, 7, 16)))
    assert baseline.shape == (1, 5, 16)
    q = torch.randn(1, 5, 16)
    memory = torch.randn(1, 7, 16)
    with torch.no_grad():
        original = fake.decoder.out_norm(fake.decoder.blocks[-1](fake.decoder.blocks[-2](q, memory), memory))
        adapted = tail(q, memory)
    original_geometry = geometry_vector(fake.heads(original))
    assert torch.allclose(geometry_vector(adapted), original_geometry, atol=1e-6)
    assert adapted["semantic_logits"].shape == (1, 5, 41)
    assert SemanticTail(fake, num_classes=26)(q, memory)["semantic_logits"].shape == (1, 5, 26)
    assert set(GEOMETRY_KEYS).issubset(adapted)
    loss = tail(q, memory)["semantic_logits"].sum()
    loss.backward()
    trainable = {name: p for name, p in tail.named_parameters() if p.requires_grad}
    assert any(name.endswith(".b") and p.grad is not None and p.grad.abs().sum() > 0
               for name, p in trainable.items())
    assert all(p.grad is None for p in fake.parameters())
    saved = {name: p.detach().clone() for name, p in trainable.items()}
    other = SemanticTail(fake).eval()
    with torch.no_grad():
        for name, p in other.named_parameters():
            if name in saved:
                p.copy_(saved[name])
    assert torch.allclose(tail(q, memory)["semantic_logits"], other(q, memory)["semantic_logits"])
    assert isinstance(tail.blocks[0].ff[0], LoRALinear)
    with tempfile.TemporaryDirectory() as folder:
        adapter = SemanticD4RT(fake, variant="sam", num_classes=26)
        saved_path = Path(folder) / "ovis_adapter.pt"
        torch.save({"variant": "sam", "num_classes": 26,
                    "model": adapter.semantic_head.state_dict()}, saved_path)
        restored = SemanticD4RT(fake, variant="sam", num_classes=26)
        restored.load_adapter(saved_path)
        assert torch.equal(adapter.semantic_head.classifier.weight, restored.semantic_head.classifier.weight)
        try:
            SemanticD4RT(fake, variant="sam", num_classes=41).load_adapter(saved_path)
        except ValueError:
            pass
        else:
            raise AssertionError("OVIS adapter must reject a 41-class output head")


def test_model_extraction_integration() -> None:
    from semantic.model import extract_query_features
    from tools.load_opend4rt import DEFAULT_VARIANT, load_model

    with tempfile.TemporaryDirectory() as folder:
        dataset, clips = fixture(Path(folder))
        video = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((128, 128))).copy())
                             .permute(2, 0, 1) for p in clips[0]["frames"]]).float()[None].div(255).to("cuda", torch.bfloat16)
        labels = np.stack([dataset.gt_label_map(clips[0], t) for t in range(8)])
        query, _, _ = sample_queries(labels, 4, 7)
        model = load_model(DEFAULT_VARIANT / "opend4rt.ckpt", DEFAULT_VARIANT / "model.yaml")
        model = model.to("cuda", torch.bfloat16).eval()
        query = {key: value.to("cuda") for key, value in query.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            memory = model.encode_video(video)
            feature = extract_query_features(model, video, query, memory)
            official = model.decode_queries(video, query, memory)
        assert feature["decoded"].shape == (1, 8 * 4, 1280)
        assert feature["prelast"].shape == feature["decoded"].shape
        assert feature["rgb_patch"].shape == (1, 8 * 4, 243)
        difference = (feature["geometry"].float() - geometry_vector(official).float()).abs()
        assert difference.max().item() < 1e-3, (
            f"Maximum extraction mismatch: {difference.max().item()}; "
            f"mean={difference.mean().item()}, refmax={geometry_vector(official).abs().max().item()}"
        )
        assert all(torch.isfinite(value).all() for value in feature.values())
        adapter = SemanticD4RT(model, variant="sam").to("cuda", dtype=torch.bfloat16).eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            extended = adapter({"video": video, "query": query})
        assert extended["semantic_logits"].shape == (1, 8 * 4, 41)
        assert all(key in extended for key in GEOMETRY_KEYS)
        assert torch.allclose(geometry_vector(extended), geometry_vector(official), atol=1e-3)
        adapter_path = Path(folder) / "adapter.pt"
        torch.save({"variant": "sam", "num_classes": 41,
                    "model": adapter.semantic_head.state_dict()}, adapter_path)
        restored = SemanticD4RT(model, variant="sam").to("cuda", dtype=torch.bfloat16).eval()
        restored.load_adapter(adapter_path)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            repeat = restored({"video": video, "query": query})
        assert torch.equal(extended["semantic_logits"], repeat["semantic_logits"])


def test_sam_integration() -> None:
    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = ROOT / "semantic_artifacts/weights/sam2.1_hiera_tiny.pt"
    assert checkpoint.is_file()
    with tempfile.TemporaryDirectory() as folder:
        dataset, clips = fixture(Path(folder))
        predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", str(checkpoint))
        labels, metrics = propagate_one(predictor, dataset, clips[0])
        assert labels.shape == (8, 128, 128)
        assert np.any(labels[1:] == 1)
        assert metrics["prompted_objects"] == 1


def test_pipeline_smoke() -> None:
    """One synthetic clip checks wiring only; it is not a research result."""
    from semantic.run import _cache_clip, _train_head, _train_lora, read_cache
    from tools.load_opend4rt import DEFAULT_VARIANT, load_model

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        dataset, clips = fixture(root)
        model = load_model(DEFAULT_VARIANT / "opend4rt.ckpt", DEFAULT_VARIANT / "model.yaml")
        model = model.to("cuda", torch.bfloat16).eval()
        cache = root / "features"
        _cache_clip(model, dataset, clips[0], "train", None, cache / "train/1.npz", seed=42)
        del model
        torch.cuda.empty_cache()
        # Copying a synthetic feature file is intentional here; never do this
        # for an evaluation run, where video-level split isolation is required.
        for split in ("val", "test"):
            target = cache / split / "1.npz"
            target.parent.mkdir(parents=True)
            target.write_bytes((cache / "train/1.npz").read_bytes())
        train, val, test = [read_cache(cache, split) for split in ("train", "val", "test")]
        args = SimpleNamespace(steps=2, out=root / "runs", cache=cache, num_classes=41)
        head_report = _train_head(args, "sam", 0, train, val, test)
        assert Path(head_report["checkpoint"]).is_file()
        lora_report = _train_lora(args, 0, train, val, test)
        assert Path(lora_report["checkpoint"]).is_file()
        assert np.isfinite(lora_report["test"]["geometry_mean_absolute_delta"])


if __name__ == "__main__":
    test_manifest_and_labels()
    test_ovis_manifest_and_categories()
    test_heldout_category_gets_training_source_example()
    test_teacher_rejects_uncertainty()
    test_lora_tail_and_reload()
    print("Semantic unit tests passed")
    if "--integration" in sys.argv:
        test_model_extraction_integration()
        print("OpenD4RT integration passed")
        test_sam_integration()
        print("SAM 2 integration passed")
    if "--pipeline" in sys.argv:
        test_pipeline_smoke()
        print("Semantic training pipeline smoke passed")
