"""Trainable source-query semantics beside an unchanged frozen geometry path."""
from __future__ import annotations
import torch
from torch import nn
from semantic.model import SemanticTail

class SourceQuerySemanticD4RT(nn.Module):
    def __init__(self,base:nn.Module,num_classes:int=26):
        super().__init__()
        self.base=base
        for parameter in self.base.parameters():parameter.requires_grad_(False)
        self.semantic_tail=SemanticTail(base,num_classes)
        self.train(False)

    def train(self,mode:bool=True):
        super().train(mode)
        self.base.eval()
        self.semantic_tail.eval()  # Deterministic dropout; gradients remain enabled.
        return self

    def source_prefix(self,video,query,memory):
        src=query['t_src']
        with torch.no_grad():
            x=self.base.query_embedder(video=video,u=query['u'].to(video.dtype),v=query['v'].to(video.dtype),t_src=src,t_tgt=src,t_cam=src)
            for block in self.base.decoder.blocks[:-2]:x=block(x,memory)
        return x

    def forward(self,batch):
        video=batch['video'];query=batch['query']
        with torch.no_grad():
            memory=self.base.encode_video(video,aspect_ratio=batch.get('aspect_ratio'))
            geometry=self.base.decode_queries(video,query,memory)
            prefix=self.source_prefix(video,query,memory)
        semantic=self.semantic_tail(prefix,memory)['semantic_logits']
        return {**geometry,'semantic_logits':semantic}
