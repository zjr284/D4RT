import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch import nn
from test_dino_interface import FakeDino
from semantic.model import GEOMETRY_KEYS
from semantic.dino_native_interface import NativeDinoSemanticD4RT,load_source_pair
from semantic.dino_geometry_cache import CachedGeometryNativeDinoSemanticD4RT


class Geometry(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(1.))
    def encode_video(self,video,aspect_ratio=None):
        memory=video.mean((1,2,3,4))[:,None,None]*self.weight
        if aspect_ratio is not None:memory=memory+torch.as_tensor(aspect_ratio).reshape(len(video),1,1)
        return memory
    def decode_queries(self,video,query,memory):
        value=(query['u']+query['v']+query['t_tgt']+2*query['t_cam'])[...,None]+memory
        return {k:value.clone()for k in GEOMETRY_KEYS}
    def forward(self,batch):return self.decode_queries(batch['video'],batch['query'],self.encode_video(batch['video'],batch.get('aspect_ratio')))


def main():
    with tempfile.TemporaryDirectory()as folder:
        paths=[]
        for i in range(2):
            p=Path(folder)/f'{i}.png';Image.fromarray(np.full((140,175,3),40+i*60,dtype=np.uint8)).save(p);paths.append(str(p))
        q={'u':torch.tensor([[0.,.5,1.]]),'v':torch.tensor([[1.,.5,0.]]),'t_src':torch.zeros(1,3,dtype=torch.long),'t_tgt':torch.zeros(1,3,dtype=torch.long),'t_cam':torch.zeros(1,3,dtype=torch.long)}
        base=Geometry();dino=FakeDino();ref=NativeDinoSemanticD4RT(base,dino);cached=CachedGeometryNativeDinoSemanticD4RT(base,dino)
        cached.semantic_head.load_state_dict(ref.semantic_head.state_dict());video=load_source_pair([paths])[0].to(torch.bfloat16)
        b={'video':video,'source_frames':[paths],'query':q}
        for call in (b,{**b,'query':{**q,'t_tgt':torch.ones_like(q['t_tgt'])}}):
            a=cached(call);r=ref(call)
            for k in r:assert torch.equal(a[k],r[k])
        assert cached.geometry_misses==1 and cached.geometry_hits==1
        for ratio in ([1.],[1.7]):
            call={**b,'aspect_ratio':ratio};a=cached(call);r=ref(call)
            for k in r:assert torch.equal(a[k],r[k])
        assert cached.geometry_misses==3 and cached.cache_info()['geometry_entries']==2
        with torch.no_grad():base.weight.add_(1)
        a=cached(b);r=ref(b)
        for k in r:assert torch.equal(a[k],r[k])
        assert cached.geometry_misses==4
        b2={'video':torch.cat([video,video]),'source_frames':[paths,paths],'query':{k:torch.cat([v,v])for k,v in q.items()}}
        a=cached(b2);r=ref(b2)
        for k in r:assert torch.equal(a[k],r[k])
        assert cached.geometry_misses==5 and cached.geometry_evictions>0
        cached.clear_cache();assert cached.geometry_misses==0 and cached.cache_info()['geometry_entries']==0
    print('Whole-batch geometry equality, query reuse, shape/aspect/weight invalidation, LRU and clear passed')


if __name__=='__main__':main()
