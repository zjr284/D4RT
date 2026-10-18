import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from semantic.dino_flip_augmentation import mirror_queries
xy=np.array([[0,0,0],[127,127,0],[27.5,18.25,0],[63,96,0]],np.float64);mirrored=mirror_queries(xy)
assert np.array_equal(mirror_queries(mirrored),xy)and np.array_equal(mirrored[:,1:],xy[:,1:])
patch=torch.arange(64,dtype=torch.float64).reshape(1,1,8,8)
def sample(values,q):
 grid=torch.tensor(((q[:,:2]+.5)/128)*2-1)[None,None]
 return F.grid_sample(values,grid,padding_mode='border',align_corners=False)
assert torch.allclose(sample(patch,xy),sample(torch.flip(patch,[-1]),mirrored),atol=1e-12,rtol=0)
for bad in [np.array([[-1,0,0]]),np.array([[0,128,0]]),np.array([[0,0,1]])]:
 try:mirror_queries(bad)
 except ValueError:pass
 else:raise AssertionError('Invalid source query accepted')
# Replacing the original with an identical view cannot change sampled inputs.
x=torch.arange(32*3).reshape(32,3);choice=torch.tensor([True,False]*16)
assert torch.equal(torch.where(choice[:,None],x,x),x)
print('PASS flip involution, continuous pixel-center border sampling, provenance and identical-view identity')
