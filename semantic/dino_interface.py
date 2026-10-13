"""Frozen source-image semantics beside the unmodified OpenD4RT forward."""
from __future__ import annotations
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from torch import nn
from torch.nn import functional as F
from semantic.model import GEOMETRY_KEYS, SemanticHead
from semantic.semantic_feature_control import VENDOR, WEIGHTS
from semantic.dino_sam_control import ROOT as EXPERIMENT, checkpoint
from semantic.oracle_probe import file_hash
from tools.load_opend4rt import DEFAULT_VARIANT, load_model


def cuda_bf16(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


class DinoSemanticD4RT(nn.Module):
    """Inference-only API: original geometry + calibrated 26-class semantics.

    Pass float32 RGB, or supply matching float32 ``semantic_video`` alongside
    the original geometry RGB precision. CUDA geometry uses BF16 autocast.
    The original geometry model receives the exact same batch.
    Semantic weights stay float32 and quantization matches cached features.
    """
    def __init__(self, published_model, dino_model, background_offset=-1.75, feature_size=224):
        super().__init__()
        if not torch.isfinite(torch.tensor(background_offset)): raise ValueError('Finite background offset required')
        if not isinstance(feature_size, int) or feature_size not in (224, 448): raise ValueError('Choose feature_size224 or448')
        self.base = published_model; self.dino = dino_model
        self.feature_size = feature_size
        self.semantic_head = SemanticHead(hidden_dim=384, num_classes=26)
        self.background_offset = float(background_offset)
        self.requires_grad_(False); super().train(False)

    def train(self, mode=True):
        if mode: raise ValueError('This frozen inference adapter does not support training mode')
        return super().train(False)

    def load_head(self, path, expected_sha256=None):
        if expected_sha256 is not None and file_hash(path) != expected_sha256: raise ValueError('Semantic checkpoint hash mismatch')
        state = torch.load(path, map_location='cpu', weights_only=True)
        weights = state['model']
        if tuple(weights['classifier.weight'].shape) != (26, 384): raise ValueError('Expected 26-class, 384-dimensional head')
        if not all(torch.isfinite(v).all() for v in weights.values()): raise ValueError('Nonfinite semantic weights')
        self.semantic_head.load_state_dict(weights, strict=True)
        return self

    @staticmethod
    def validate(video, query):
        if not torch.is_tensor(video) or video.ndim != 5 or tuple(video.shape[2:]) != (3, 128, 128) or video.shape[0] < 1 or video.shape[1] < 1:
            raise ValueError('Expected nonempty RGB video [B,T,3,128,128]')
        if not video.is_floating_point() or not torch.isfinite(video).all() or video.min() < 0 or video.max() > 1:
            raise ValueError('Video must be finite floating RGB in [0,1]')
        if not isinstance(query, dict) or not all(k in query for k in ('u', 'v', 't_src', 't_tgt', 't_cam')):
            raise ValueError('Missing query fields')
        if not torch.is_tensor(query['u']): raise ValueError('Query tensors required')
        shape = query['u'].shape
        if len(shape) != 2 or shape[0] != video.shape[0] or shape[1] < 1: raise ValueError('Expected nonempty query [B,Q]')
        for k in ('u', 'v', 't_src', 't_tgt', 't_cam'):
            q = query[k]
            if not torch.is_tensor(q) or q.shape != shape or q.device != video.device: raise ValueError('Query shape/device mismatch')
            if k in ('u', 'v'):
                if not q.is_floating_point() or not torch.isfinite(q).all() or q.min() < 0 or q.max() > 1:
                    raise ValueError('Finite normalized coordinates in [0,1] required')
            elif q.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) or q.min() < 0 or q.max() >= video.shape[1]:
                raise ValueError('Integer query times must be inside the video')

    @torch.inference_mode()
    def semantic_features(self, video, query):
        self.validate(video, query)
        b, t = video.shape[:2]
        mean = torch.tensor([.485, .456, .406], device=video.device)[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device=video.device)[None, :, None, None]
        all_features = []
        for batch in range(b):
            image = (F.interpolate(video[batch].float(), size=(self.feature_size, self.feature_size), mode='bicubic', align_corners=False) - mean) / std
            with cuda_bf16(video.device):
                tokens = self.dino.forward_features(image)['x_norm_patchtokens']
            grid_size = self.feature_size // 14
            if tokens.shape != (t, grid_size * grid_size, 384): raise ValueError(f'Expected {grid_size}x{grid_size} DINOv2 patch tokens with dimension384')
            tokens = tokens.reshape(t, grid_size, grid_size, 384).permute(0, 3, 1, 2).float()
            result = torch.empty((query['u'].shape[1], 384), device=video.device, dtype=torch.float32)
            for source in query['t_src'][batch].unique(sorted=True).tolist():
                rows = torch.where(query['t_src'][batch] == source)[0]
                pixel = torch.stack([query['u'][batch, rows], query['v'][batch, rows]], -1).float() * 127
                grid = ((pixel + .5) / 128 * 2 - 1)[None, None]
                # Disable any enclosing autocast, matching the float32 cache interpolation.
                with torch.autocast(video.device.type, enabled=False):
                    result[rows] = F.grid_sample(tokens[source:source+1], grid, mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0].T
            all_features.append(result.to(torch.float16).float())
        return torch.stack(all_features)

    @torch.inference_mode()
    def semantic_outputs(self, features):
        if self.semantic_head.classifier.weight.dtype != torch.float32: raise ValueError('Keep semantic head float32; do not cast the entire adapter to BF16')
        with torch.autocast(features.device.type, enabled=False): raw = self.semantic_head(features.float())
        calibrated = raw.clone(); calibrated[..., 0] += self.background_offset
        return {'semantic_raw_logits': raw, 'semantic_logits': calibrated, 'semantic_labels': calibrated.argmax(-1)}

    @torch.inference_mode()
    def forward(self, batch):
        video = batch['video']; query = batch['query']; self.validate(video, query)
        semantic_video = batch.get('semantic_video', video)
        if 'semantic_video' in batch:
            if not torch.is_tensor(semantic_video) or semantic_video.dtype != torch.float32 or semantic_video.shape != video.shape or semantic_video.device != video.device:
                raise ValueError('semantic_video must be matching float32 RGB on the geometry device')
            self.validate(semantic_video, query)
            if not torch.equal(semantic_video.to(video.dtype), video):
                raise ValueError('semantic_video must quantize exactly to the geometry RGB')
        with cuda_bf16(video.device): result = dict(self.base(batch))
        if not all(key in result for key in GEOMETRY_KEYS): raise ValueError('Original geometry outputs missing')
        result.update(self.semantic_outputs(self.semantic_features(semantic_video, query)))
        return result


def load_selected_adapter(seed=0, device='cuda'):
    if seed not in (0, 1, 2): raise ValueError('Choose a preregistered seed0/1/2')
    summary = json.loads((EXPERIMENT / 'summary.json').read_text())
    verified = json.loads((EXPERIMENT / 'verification.json').read_text())
    if file_hash(EXPERIMENT / 'summary.json') != verified['summary_sha256']: raise ValueError('Round15 summary changed')
    arm = summary['validation_selected_arm']
    fp = summary['protocol']['inherited_feature_coverage_protocol']['inherited_feature_protocol']
    if file_hash(WEIGHTS) != fp['dinov2_weight_sha256']: raise ValueError('DINOv2 backbone changed')
    if file_hash(DEFAULT_VARIANT / 'opend4rt.ckpt') != fp['d4rt_backbone_sha256']: raise ValueError('D4RT backbone changed')
    if file_hash(DEFAULT_VARIANT / 'model.yaml') != fp['d4rt_model_config_sha256']: raise ValueError('D4RT config changed')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip() != fp['dinov2_source_commit']:
        raise ValueError('DINOv2 code changed')
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    base = load_model(DEFAULT_VARIANT / 'opend4rt.ckpt', DEFAULT_VARIANT / 'model.yaml').to(device, dtype=dtype).eval()
    os.environ['XFORMERS_DISABLED'] = '1'; sys.path.insert(0, str(VENDOR.resolve()))
    from dinov2.hub.backbones import dinov2_vits14
    dino = dinov2_vits14(pretrained=False)
    dino.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    dino = dino.to(device).eval()
    adapter = DinoSemanticD4RT(base, dino, summary['results'][arm]['offset']).to(device)
    adapter.load_head(checkpoint(arm, seed), verified['checkpoint_sha256'][f'{arm}_{seed}'])
    return adapter


def load_resolution_selected_adapter(seed=0, device='cuda'):
    """Load round18 validation selection, keeping the round15 entry compatible."""
    from semantic.dino_resolution_control import ROOT, ARMS, checkpoint as resolution_checkpoint
    if seed not in (0, 1, 2): raise ValueError('Choose a preregistered seed0/1/2')
    summary = json.loads((ROOT/'summary.json').read_text())
    verified = json.loads((ROOT/'verification.json').read_text())
    selection = json.loads((ROOT/'frozen_selection.json').read_text())
    if file_hash(ROOT/'summary.json') != verified['summary_sha256']: raise ValueError('Round18 summary changed')
    if file_hash(ROOT/'frozen_selection.json') != summary['frozen_selection_sha256']: raise ValueError('Round18 selection changed')
    scores = {a: sum(r['calibrated_validation']['macro_f1_present'] for r in summary['results'][a]['runs'])/3 for a in ARMS}
    arm = max(ARMS, key=lambda a: scores[a])
    if arm != summary['validation_selected_arm'] or arm != selection['validation_selected_arm']: raise ValueError('Round18 validation selection mismatch')
    if file_hash(WEIGHTS) != summary['protocol']['input_sha256'][str(WEIGHTS)]: raise ValueError('Round18 backbone changed')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip() != summary['protocol']['dinov2_source_commit']:
        raise ValueError('Round18 DINOv2 code changed')
    if selection['offsets'][arm] != summary['results'][arm]['offset']: raise ValueError('Round18 background offset mismatch')
    sha = verified['checkpoint_sha256'][f'{arm}_{seed}']
    if sha != selection['checkpoint_sha256'][f'{arm}_{seed}']: raise ValueError('Round18 checkpoint selection mismatch')
    # Reuse the strict foundation loader; its default224 behavior remains intact.
    adapter = load_selected_adapter(seed, device)
    adapter.feature_size = 224 if arm == 'resolution224' else 448
    adapter.background_offset = float(selection['offsets'][arm])
    adapter.load_head(resolution_checkpoint(arm, seed), sha)
    return adapter
