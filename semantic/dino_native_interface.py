"""Frozen original-JPEG semantics with checked source provenance for geometry."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from semantic.dino_interface import DinoSemanticD4RT,load_selected_adapter,cuda_bf16
from semantic.model import GEOMETRY_KEYS
from semantic import dino_native_rgb_control as experiment
from semantic.oracle_probe import file_hash


def load_source_pair(frame_groups):
    """One RGB decode per file, then the two registered PIL transforms on CPU."""
    geometry=[];semantic=[]
    if not isinstance(frame_groups,(list,tuple))or not frame_groups:raise ValueError('source_frames must contain nonempty frame groups')
    for group in frame_groups:
        if not isinstance(group,(list,tuple))or not group:raise ValueError('source_frames contains invalid frame group')
        g=[];s=[]
        for path in group:
            if not isinstance(path,(str,Path)):raise ValueError('Original RGB frame paths required')
            with Image.open(path)as image:
                rgb=image.convert('RGB')
                def pixels(size):return torch.from_numpy(np.array(rgb.resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)).permute(2,0,1)/255
                g.append(pixels(128));s.append(pixels(448))
        geometry.append(torch.stack(g));semantic.append(torch.stack(s))
    return torch.stack(geometry),torch.stack(semantic)


class NativeDinoSemanticD4RT(DinoSemanticD4RT):
    """Inference-only native source RGB; geometry receives the original batch."""
    def __init__(self,published_model,dino_model,background_offset=-1.5):
        super().__init__(published_model,dino_model,background_offset,feature_size=448)

    @torch.inference_mode()
    def native_features(self,rgb,query):
        if rgb.ndim!=5 or tuple(rgb.shape[2:])!=(3,448,448)or rgb.dtype!=torch.float32:raise ValueError('Expected native448 float32 RGB')
        mean=torch.tensor([.485,.456,.406],device=rgb.device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=rgb.device)[None,:,None,None];features=[]
        for batch in range(len(rgb)):
            images=(rgb[batch]-mean)/std
            with cuda_bf16(rgb.device):tokens=self.dino.forward_features(images)['x_norm_patchtokens']
            if tokens.shape!=(rgb.shape[1],1024,384):raise ValueError('Expected native32x32 patch tokens')
            tokens=tokens.reshape(rgb.shape[1],32,32,384).permute(0,3,1,2).float()
            output=torch.empty(query['u'].shape[1],384,device=rgb.device,dtype=torch.float32)
            for source in query['t_src'][batch].unique(sorted=True).tolist():
                rows=torch.where(query['t_src'][batch]==source)[0]
                positions=torch.stack([query['u'][batch,rows],query['v'][batch,rows]],-1).float()*127
                grid=((positions+.5)/128*2-1)[None,None]
                with torch.autocast(rgb.device.type,enabled=False):output[rows]=F.grid_sample(tokens[source:source+1],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
            features.append(output.half().float())
        return torch.stack(features)

    def source_rgb(self,batch):
        video=batch['video'];DinoSemanticD4RT.validate(video,batch['query'])
        if 'semantic_video'in batch:raise ValueError('Native source_frames cannot mix the legacy semantic_video input')
        groups=batch.get('source_frames')
        if not isinstance(groups,(list,tuple))or len(groups)!=video.shape[0]:raise ValueError('One source frame group per batch video required')
        if any(not isinstance(g,(list,tuple))or len(g)!=video.shape[1]for g in groups):raise ValueError('Source frame counts must match geometry times')
        geometry,native=load_source_pair(groups)
        if not torch.equal(geometry.to(device=video.device,dtype=video.dtype),video):raise ValueError('Original source frames do not reproduce geometry RGB')
        return native.to(device=video.device)

    @torch.inference_mode()
    def forward(self,batch):
        native=self.source_rgb(batch)
        with cuda_bf16(batch['video'].device):outputs=dict(self.base(batch))
        if not all(k in outputs for k in GEOMETRY_KEYS):raise ValueError('Original geometry outputs missing')
        outputs.update(self.semantic_outputs(self.native_features(native,batch['query'])))
        return outputs


def load_native_selected_adapter(seed=0,device='cuda'):
    if seed not in (0,1,2):raise ValueError('Choose a preregistered seed0/1/2')
    summary=json.loads((experiment.ROOT/'summary.json').read_text());verified=json.loads((experiment.ROOT/'verification.json').read_text())
    selection=json.loads((experiment.ROOT/'frozen_selection.json').read_text())
    if file_hash(experiment.ROOT/'summary.json')!=verified['summary_sha256']:raise ValueError('Native experiment summary changed')
    if file_hash(experiment.ROOT/'frozen_selection.json')!=summary['frozen_selection_sha256']:raise ValueError('Native selection changed')
    scores={a:sum(r['calibrated_validation']['macro_f1_present']for r in summary['results'][a]['runs'])/3 for a in experiment.ARMS}
    arm=max(experiment.ARMS,key=lambda a:scores[a])
    if arm!=summary['validation_selected_arm']or arm!=selection['validation_selected_arm']or not arm.startswith('native_'):raise ValueError('Expected verified native validation selection')
    if selection['offsets'][arm]!=summary['results'][arm]['offset']:raise ValueError('Native offset mismatch')
    sha=verified['checkpoint_sha256'][f'{arm}_{seed}']
    if sha!=selection['checkpoint_sha256'][f'{arm}_{seed}']:raise ValueError('Native checkpoint mismatch')
    foundation=load_selected_adapter(seed,device)
    model=NativeDinoSemanticD4RT(foundation.base,foundation.dino,selection['offsets'][arm]).to(device)
    model.load_head(experiment.checkpoint(arm,seed),sha)
    return model
