"""Non-invasive semantic adapter for the public OpenD4RT query decoder."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic.ytvis import NUM_CLASSES


GEOMETRY_KEYS = ("xyz_3d", "uv_2d", "visibility", "displacement", "normal", "confidence")


def geometry_vector(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([outputs[key].reshape(*outputs[key].shape[:2], -1) for key in GEOMETRY_KEYS], dim=-1)


class SemanticHead(nn.Module):
    def __init__(self, hidden_dim: int = 1280, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, decoded: torch.Tensor) -> torch.Tensor:
        return self.classifier(decoded.to(self.classifier.weight.dtype))


class RGBPatchBaseline(nn.Module):
    def __init__(self, patch_dim: int = 3 * 9 * 9, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.classifier = nn.Linear(patch_dim, num_classes)

    def forward(self, rgb_patch: torch.Tensor) -> torch.Tensor:
        return self.classifier(rgb_patch.to(self.classifier.weight.dtype))


class LoRALinear(nn.Module):
    """Add a rank-8 update without changing the frozen published weight."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.base(x)
        update = F.linear(F.linear(x.to(self.a.dtype), self.a), self.b) * self.scale
        return original + update.to(original.dtype)


class SemanticTail(nn.Module):
    """Last two decoder blocks, original geometry heads, and a new class head.

    The preceding video encoder and six decoder blocks are frozen and cached.
    This keeps LoRA training feasible on an 8 GB GPU while making the adapted
    decoder change the query features, not just the final classifier.
    """

    def __init__(self, published_model: nn.Module, num_classes: int = NUM_CLASSES):
        super().__init__()
        if len(published_model.decoder.blocks) < 2:
            raise ValueError("Expected at least two decoder blocks")
        self.blocks = copy.deepcopy(published_model.decoder.blocks[-2:])
        self.out_norm = copy.deepcopy(published_model.decoder.out_norm)
        self.geometry_heads = copy.deepcopy(published_model.heads)
        self.semantic_head = SemanticHead(published_model.decoder.out_norm.normalized_shape[0], num_classes)
        for parameter in self.blocks.parameters():
            parameter.requires_grad_(False)
        for module in (self.out_norm, self.geometry_heads):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        for block in self.blocks:
            for index in (0, 3):
                block.ff[index] = LoRALinear(block.ff[index])

    def forward(self, before_last_two: torch.Tensor, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        x = before_last_two
        for block in self.blocks:
            x = block(x, memory)
        decoded = self.out_norm(x)
        result = self.geometry_heads(decoded)
        result["semantic_logits"] = self.semantic_head(decoded)
        return result


class SemanticD4RT(nn.Module):
    """Public inference interface: all six original outputs plus class logits.

    Instantiate with a strictly loaded public OpenD4RT model. Only adapter
    weights are saved by the experiment; the 14 GB source checkpoint remains
    separate and is never rewritten.
    """

    def __init__(self, published_model: nn.Module, variant: str = "sam", num_classes: int = NUM_CLASSES):
        super().__init__()
        if variant not in ("gt", "sam", "lora"):
            raise ValueError(f"Unsupported adapter variant: {variant}")
        self.base = published_model
        self.variant = variant
        self.num_classes = num_classes
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.semantic_head = None if variant == "lora" else SemanticHead(
            published_model.decoder.out_norm.normalized_shape[0], num_classes
        )
        self.tail = SemanticTail(published_model, num_classes) if variant == "lora" else None

    def load_adapter(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("variant") != self.variant or checkpoint.get("num_classes") != self.num_classes:
            raise ValueError("Adapter checkpoint variant/class count mismatch")
        if self.tail is None:
            self.semantic_head.load_state_dict(checkpoint["model"], strict=True)
        else:
            trainable = checkpoint["trainable"]
            current = {name: p for name, p in self.tail.named_parameters() if p.requires_grad}
            if set(trainable) != set(current):
                raise ValueError("LoRA adapter parameter names do not match")
            with torch.no_grad():
                for name, parameter in current.items():
                    if tuple(parameter.shape) != tuple(trainable[name].shape):
                        raise ValueError(f"LoRA shape mismatch: {name}")
                    parameter.copy_(trainable[name].to(device=parameter.device, dtype=parameter.dtype))

    def forward(self, batch: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        video = batch["video"]
        query = batch["query"]
        memory = self.base.encode_video(video, aspect_ratio=batch.get("aspect_ratio"))
        u = query["u"].to(dtype=video.dtype)
        v = query["v"].to(dtype=video.dtype)
        token = self.base.query_embedder(
            video=video, u=u, v=v,
            t_src=query["t_src"], t_tgt=query["t_tgt"], t_cam=query["t_cam"],
        )
        if self.tail is None:
            decoded = self.base.decoder(token, memory)
            result = self.base.heads(decoded)
            result["semantic_logits"] = self.semantic_head(decoded)
            return result
        for block in self.base.decoder.blocks[:-2]:
            token = block(token, memory)
        return self.tail(token, memory)


@torch.inference_mode()
def extract_query_features(model: nn.Module, video: torch.Tensor,
                           query: dict[str, torch.Tensor], memory: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract frozen features and original outputs from a query batch."""
    q = model.query_embedder(
        video=video, u=query["u"].to(dtype=video.dtype), v=query["v"].to(dtype=video.dtype),
        t_src=query["t_src"], t_tgt=query["t_tgt"], t_cam=query["t_cam"],
    )
    for block in model.decoder.blocks[:-2]:
        q = block(q, memory)
    before_last_two = q
    for block in model.decoder.blocks[-2:]:
        q = block(q, memory)
    decoded = model.decoder.out_norm(q)
    rgb_patch = model.query_embedder._extract_local_patches(
        video, query["u"].to(dtype=video.dtype), query["v"].to(dtype=video.dtype), query["t_src"]
    )
    return {"decoded": decoded, "prelast": before_last_two,
            "rgb_patch": rgb_patch, "geometry": geometry_vector(model.heads(decoded))}
