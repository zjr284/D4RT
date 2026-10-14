"""Bounded frozen patch-token reuse, with source content checks on every call."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from semantic.dino_native_interface import NativeDinoSemanticD4RT
from semantic.dino_interface import DinoSemanticD4RT,cuda_bf16
from semantic.model import GEOMETRY_KEYS


class CachedNativeDinoSemanticD4RT(NativeDinoSemanticD4RT):
    def __init__(self,base,dino,background_offset=-1.5,max_cached_videos=2):
        super().__init__(base,dino,background_offset)
        if not isinstance(max_cached_videos,int)or max_cached_videos<1:raise ValueError('Positive cache bound required')
        self.max_cached_videos=max_cached_videos;self._cache=OrderedDict();self.hits=0;self.misses=0;self.evictions=0

    def clear_cache(self):
        self._cache.clear();self.hits=0;self.misses=0;self.evictions=0

    def cache_info(self):
        return {'entries':len(self._cache),'hits':self.hits,'misses':self.misses,'evictions':self.evictions}

    def _backbone_signature(self):
        if self.dino.training:raise ValueError('Frozen semantic backbone must remain eval')
        return tuple((p._version,p.data_ptr(),str(p.device),str(p.dtype))for p in self.dino.parameters())

    @staticmethod
    def _decode(content):
        geometry=[];native=[]
        for data in content:
            with Image.open(io.BytesIO(data))as image:
                rgb=image.convert('RGB')
                def pixels(size):return torch.from_numpy(np.array(rgb.resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)).permute(2,0,1)/255
                geometry.append(pixels(128));native.append(pixels(448))
        return torch.stack(geometry),torch.stack(native)

    @torch.inference_mode()
    def source_tokens(self,batch):
        video=batch['video'];DinoSemanticD4RT.validate(video,batch['query'])
        if 'semantic_video'in batch:raise ValueError('Native source_frames cannot mix legacy semantic_video')
        groups=batch.get('source_frames')
        if not isinstance(groups,(list,tuple))or len(groups)!=len(video):raise ValueError('One source group per video required')
        signature=self._backbone_signature();all_tokens=[]
        mean=torch.tensor([.485,.456,.406],device=video.device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=video.device)[None,:,None,None]
        for i,group in enumerate(groups):
            if not isinstance(group,(list,tuple))or len(group)!=video.shape[1]or any(not isinstance(p,(str,Path))for p in group):raise ValueError('Source frame counts and paths must match geometry')
            content=[Path(p).read_bytes()for p in group];key=(tuple(hashlib.sha256(v).hexdigest()for v in content),signature)
            if key in self._cache:
                entry=self._cache[key]
                if not torch.equal(entry['geometry'].to(device=video.device,dtype=video.dtype),video[i]):raise ValueError('Original source frames do not reproduce geometry RGB')
                self._cache.move_to_end(key);self.hits+=1
            else:
                geometry,native=self._decode(content)
                if not torch.equal(geometry.to(device=video.device,dtype=video.dtype),video[i]):raise ValueError('Original source frames do not reproduce geometry RGB')
                images=(native.to(video.device)-mean)/std
                with cuda_bf16(video.device):patches=self.dino.forward_features(images)['x_norm_patchtokens']
                if patches.shape!=(video.shape[1],1024,384):raise ValueError('Expected native32x32 tokens')
                entry={'geometry':geometry,'tokens':patches.reshape(video.shape[1],32,32,384).permute(0,3,1,2).float()}
                self._cache[key]=entry;self.misses+=1
                while len(self._cache)>self.max_cached_videos:self._cache.popitem(last=False);self.evictions+=1
            all_tokens.append(entry['tokens'])
        return all_tokens

    @torch.inference_mode()
    def cached_features(self,batch):
        tokens=self.source_tokens(batch);query=batch['query'];result=[]
        for b,patches in enumerate(tokens):
            output=torch.empty(query['u'].shape[1],384,device=patches.device,dtype=torch.float32)
            for source in query['t_src'][b].unique(sorted=True).tolist():
                rows=torch.where(query['t_src'][b]==source)[0]
                position=torch.stack([query['u'][b,rows],query['v'][b,rows]],-1).float()*127;grid=((position+.5)/128*2-1)[None,None]
                with torch.autocast(patches.device.type,enabled=False):output[rows]=torch.nn.functional.grid_sample(patches[source:source+1],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
            result.append(output.half().float())
        return torch.stack(result)

    @torch.inference_mode()
    def forward(self,batch):
        features=self.cached_features(batch)
        with cuda_bf16(batch['video'].device):result=dict(self.base(batch))
        if not all(k in result for k in GEOMETRY_KEYS):raise ValueError('Original geometry outputs missing')
        result.update(self.semantic_outputs(features));return result
