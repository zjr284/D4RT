import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from test_dino_interface import FakeGeometry,FakeDino
from semantic.dino_native_interface import NativeDinoSemanticD4RT,load_source_pair
from semantic.dino_native_cache import CachedNativeDinoSemanticD4RT


def main():
    with tempfile.TemporaryDirectory()as folder:
        paths=[]
        for i in range(3):
            path=Path(folder)/f'{i}.png';Image.fromarray(np.full((140,175,3),40+i*50,dtype=np.uint8)).save(path);paths.append(str(path))
        q={'u':torch.tensor([[0.,.5,1.]]),'v':torch.tensor([[1.,.5,0.]]),'t_src':torch.zeros(1,3,dtype=torch.long),'t_tgt':torch.zeros(1,3,dtype=torch.long),'t_cam':torch.zeros(1,3,dtype=torch.long)}
        base=FakeGeometry();dino=FakeDino();ref=NativeDinoSemanticD4RT(base,dino);cached=CachedNativeDinoSemanticD4RT(base,dino,max_cached_videos=2);cached.semantic_head.load_state_dict(ref.semantic_head.state_dict())
        def batch(path):return {'video':load_source_pair([[path]])[0].to(torch.bfloat16),'query':q,'source_frames':[[path]]}
        b=batch(paths[0]);a=cached(b);expect=ref(b)
        for k in expect:assert torch.equal(a[k],expect[k])
        cached(b);assert cached.misses==1 and cached.hits==1
        with torch.no_grad():cached.semantic_head.classifier.bias.add_(1);ref.semantic_head.load_state_dict(cached.semantic_head.state_dict())
        assert torch.equal(cached(b)['semantic_logits'],ref(b)['semantic_logits'])and cached.misses==1
        with torch.no_grad():dino.weight.add_(1)
        assert torch.equal(cached(b)['semantic_logits'],ref(b)['semantic_logits'])and cached.misses==2
        Image.fromarray(np.full((140,175,3),200,dtype=np.uint8)).save(paths[0])
        try:cached(b)
        except ValueError:pass
        else:raise AssertionError('Changed source with stale geometry accepted')
        b=batch(paths[0]);assert torch.equal(cached(b)['semantic_logits'],ref(b)['semantic_logits'])and cached.misses==3
        cached(batch(paths[1]));cached(batch(paths[2]));assert cached.cache_info()['entries']==2 and cached.evictions>0
        cached.clear_cache();assert cached.cache_info()=={'entries':0,'hits':0,'misses':0,'evictions':0}
    print('Token cache equality, fresh head outputs, content/weight invalidation, provenance, LRU and clear passed')


if __name__=='__main__':main()
