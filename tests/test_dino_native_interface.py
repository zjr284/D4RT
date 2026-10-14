import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from test_dino_interface import FakeGeometry,FakeDino
from semantic.dino_native_interface import NativeDinoSemanticD4RT,load_source_pair
from semantic.oracle_probe import file_hash


def main():
    with tempfile.TemporaryDirectory()as folder:
        groups=[]
        for b in range(2):
            paths=[]
            for t in range(2):
                image=np.zeros((139,173,3),dtype=np.uint8);image[:,:,0]=np.arange(173,dtype=np.uint8)[None];image[:,:,1]=40+b*60+t*50
                path=Path(folder)/f'{b}_{t}.png';Image.fromarray(image).save(path);paths.append(str(path))
            groups.append(paths)
        geometry,native=load_source_pair(groups);assert geometry.shape==(2,2,3,128,128)and native.shape==(2,2,3,448,448)
        q={'u':torch.tensor([[0.,.5,1.],[.1,.8,.9]]),'v':torch.tensor([[0.,.75,1.],[.2,.7,.9]]),
           't_src':torch.tensor([[0,0,1],[1,0,1]]),'t_tgt':torch.zeros(2,3,dtype=torch.long),'t_cam':torch.ones(2,3,dtype=torch.long)}
        batch={'video':geometry.to(torch.bfloat16),'source_frames':groups,'query':q};saved=batch['video'].clone()
        model=NativeDinoSemanticD4RT(FakeGeometry(),FakeDino());result=model(batch)
        for k,v in model.base(batch).items():assert torch.equal(result[k],v)
        changed={k:v.clone()for k,v in q.items()};changed['t_tgt']=1-q['t_tgt'];changed['t_cam']=1-q['t_cam']
        assert torch.equal(result['semantic_logits'],model({**batch,'query':changed})['semantic_logits'])
        assert result['semantic_logits'].shape==(2,3,26)and torch.equal(result['semantic_labels'],result['semantic_logits'].argmax(-1))
        source={k:v.clone()for k,v in q.items()};source['t_src']=1-q['t_src']
        assert not torch.equal(model.native_features(native,q),model.native_features(native,source))
        assert torch.equal(batch['video'],saved)and all(not p.requires_grad for p in model.parameters())
        path=Path(folder)/'head.pt';torch.save({'model':model.semantic_head.state_dict()},path)
        fresh=NativeDinoSemanticD4RT(FakeGeometry(),FakeDino());fresh.load_head(path,file_hash(path))
        assert torch.equal(result['semantic_logits'],fresh(batch)['semantic_logits'])
        for bad in [None,[],[groups[0]],[[groups[0][1],groups[0][0]],groups[1]]]:
            try:model({**batch,'source_frames':bad})
            except ValueError:pass
            else:raise AssertionError('Invalid or reordered source accepted')
        for bad in [{**batch,'video':batch['video']+.1},{**batch,'semantic_video':geometry}]:
            try:model(bad)
            except ValueError:pass
            else:raise AssertionError('Mismatched geometry or ambiguous legacy input accepted')
    print('Native provenance, B2 queries, geometry, time/source dependence, reload and rejection checks passed')


if __name__=='__main__':main()
