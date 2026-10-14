"""Reuse exact whole-batch encode_video outputs; original decode_queries stays."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import torch
from semantic.dino_native_cache import CachedNativeDinoSemanticD4RT
from semantic.dino_interface import cuda_bf16
from semantic.model import GEOMETRY_KEYS


class CachedGeometryNativeDinoSemanticD4RT(CachedNativeDinoSemanticD4RT):
    def __init__(self,base,dino,background_offset=-1.5,max_cached_videos=2,max_geometry_batches=2):
        super().__init__(base,dino,background_offset,max_cached_videos)
        if not callable(getattr(base,'encode_video',None))or not callable(getattr(base,'decode_queries',None)):raise ValueError('Original encode_video/decode_queries API required')
        if not isinstance(max_geometry_batches,int)or max_geometry_batches<1:raise ValueError('Positive geometry cache bound required')
        self.max_geometry_batches=max_geometry_batches;self._geometry_cache=OrderedDict();self.geometry_hits=0;self.geometry_misses=0;self.geometry_evictions=0

    def clear_cache(self):
        super().clear_cache();self._geometry_cache.clear();self.geometry_hits=0;self.geometry_misses=0;self.geometry_evictions=0

    def cache_info(self):
        return dict(super().cache_info(),geometry_entries=len(self._geometry_cache),geometry_hits=self.geometry_hits,geometry_misses=self.geometry_misses,geometry_evictions=self.geometry_evictions)

    @staticmethod
    def _tensor_key(value):
        if value is None:return None
        if not torch.is_tensor(value):value=torch.as_tensor(value)
        raw=value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return tuple(value.shape),str(value.dtype),str(value.device),hashlib.sha256(raw).hexdigest()

    @torch.inference_mode()
    def geometry_memory(self,batch):
        if self.base.training:raise ValueError('Frozen geometry model must remain eval')
        signature=tuple((p._version,p.data_ptr(),str(p.device),str(p.dtype))for p in self.base.parameters())
        key=(self._tensor_key(batch['video']),self._tensor_key(batch.get('aspect_ratio')),signature)
        if key in self._geometry_cache:
            memory=self._geometry_cache[key];self._geometry_cache.move_to_end(key);self.geometry_hits+=1
        else:
            memory=self.base.encode_video(video=batch['video'],aspect_ratio=batch.get('aspect_ratio'))
            self._geometry_cache[key]=memory;self.geometry_misses+=1
            while len(self._geometry_cache)>self.max_geometry_batches:self._geometry_cache.popitem(last=False);self.geometry_evictions+=1
        return memory

    @torch.inference_mode()
    def forward(self,batch):
        features=self.cached_features(batch)
        with cuda_bf16(batch['video'].device):
            memory=self.geometry_memory(batch)
            result=dict(self.base.decode_queries(video=batch['video'],query=batch['query'],memory=memory))
        if not all(k in result for k in GEOMETRY_KEYS):raise ValueError('Original geometry fields missing')
        result.update(self.semantic_outputs(features));return result
