"""Experimental raw/source query factor; original geometry path stays frozen."""
import torch
from semantic.source_query_adapter import SourceQuerySemanticD4RT

class QueryModeSemanticD4RT(SourceQuerySemanticD4RT):
    def __init__(self,base,num_classes=26,query_mode='source'):
        if query_mode not in ('raw','source'):raise ValueError('query_mode must be raw or source')
        super().__init__(base,num_classes)
        self.query_mode=query_mode

    def source_prefix(self,video,query,memory):
        if self.query_mode=='source':return super().source_prefix(video,query,memory)
        with torch.no_grad():
            x=self.base.query_embedder(video=video,u=query['u'].to(video.dtype),v=query['v'].to(video.dtype),t_src=query['t_src'],t_tgt=query['t_tgt'],t_cam=query['t_cam'])
            for block in self.base.decoder.blocks[:-2]:x=block(x,memory)
        return x
