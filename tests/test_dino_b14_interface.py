import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from test_dino_interface import FakeGeometry
from semantic.dino_b14_interface import B14ResearchD4RT,resolve_recipe
from semantic.dino_native_interface import load_source_pair
from semantic.oracle_probe import file_hash
class FakeB14(nn.Module):
 def forward_features(self,image):return {'x_norm_patchtokens':F.avg_pool2d(image,14).flatten(2).transpose(1,2).repeat(1,1,256)}
def rejected(call):
 try:call()
 except (ValueError,RuntimeError):return
 raise AssertionError('Invalid input accepted')
def main():
 with tempfile.TemporaryDirectory()as directory:
  groups=[]
  for b in range(2):
   group=[]
   for t in range(2):
    pixels=np.zeros((145,171,3),np.uint8);pixels[:,:,0]=np.arange(171,dtype=np.uint8);pixels[:,:,1]=40+b*70+t*45;p=Path(directory)/f'{b}_{t}.png';Image.fromarray(pixels).save(p);group.append(str(p))
   groups.append(group)
  geometry,native=load_source_pair(groups);q={'u':torch.tensor([[0.,.5,1.],[.2,.8,.9]]),'v':torch.tensor([[1.,.5,0.],[.3,.1,.9]]),'t_src':torch.tensor([[0,0,1],[1,0,1]]),'t_tgt':torch.zeros(2,3,dtype=torch.long),'t_cam':torch.ones(2,3,dtype=torch.long)};batch={'video':geometry.bfloat16(),'source_frames':groups,'query':q};model=B14ResearchD4RT(FakeGeometry(),FakeB14(),-2.25);out=model(batch)
  assert out['semantic_logits'].shape==(2,3,26)and model.native_features(native,q).shape==(2,3,768)
  for k,v in model.base(batch).items():assert torch.equal(v,out[k])
  change={k:v.clone()for k,v in q.items()};change['t_tgt']=1-q['t_tgt'];change['t_cam']=1-q['t_cam'];assert torch.equal(out['semantic_logits'],model({**batch,'query':change})['semantic_logits'])
  source={**q,'t_src':1-q['t_src']};assert not torch.equal(model.native_features(native,q),model.native_features(native,source))
  path=Path(directory)/'head.pt';torch.save({'model':model.semantic_head.state_dict()},path);fresh=B14ResearchD4RT(FakeGeometry(),FakeB14(),-2.25).load_head(path,file_hash(path));assert torch.equal(out['semantic_logits'],fresh(batch)['semantic_logits'])and all(not p.requires_grad for p in model.parameters())
  rejected(lambda:model.load_head(path,'0'*64));wrong=Path(directory)/'wrong.pt';torch.save({'model':{'classifier.weight':torch.zeros(26,384),'classifier.bias':torch.zeros(26)}},wrong);rejected(lambda:model.load_head(wrong))
  for extra in ({'source_frames':[[groups[0][1],groups[0][0]],groups[1]]},{'source_frames':[]},{'semantic_video':geometry},{'video':batch['video']+.01}):rejected(lambda extra=extra:model({**batch,**extra}))
  rejected(lambda:model({**batch,'query':{**q,'u':q['u']+2}}));rejected(lambda:model({**batch,'query':{**q,'t_src':q['t_src']+2}}));rejected(lambda:model.train());rejected(lambda:resolve_recipe('oracle',0));model.semantic_head.bfloat16();rejected(lambda:model(batch))
 print('PASS B14768/B2/source-only/provenance/geometry/reload/rejections')
if __name__=='__main__':main()
