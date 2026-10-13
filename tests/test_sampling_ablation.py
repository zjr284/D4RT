"""Verify intervention distributions, support handling and reproducibility."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from semantic.sampling_ablation import Sampler

labels=np.array([1]*900+[2]*100+[0]*500+[0]*500)
times=np.array([0]*900+[1]*100+[0]*500+[1]*500)
for variant in ('baseline','time','class','time_class'):
    sampler=Sampler(labels,times,variant,device='cpu')
    a=sampler.draw(torch.Generator().manual_seed(7),count=20000)
    b=sampler.draw(torch.Generator().manual_seed(7),count=20000)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    fg,bg=a
    assert (sampler.labels[fg]>0).all() and (sampler.labels[bg]==0).all()
    rare=float((sampler.labels[fg]==2).float().mean())
    if variant in ('class','time_class'):assert 0.48<rare<0.52
    else:assert 0.08<rare<0.12
    if variant in ('time','time_class'):assert torch.equal(sampler.times[fg],sampler.times[bg])
try:
    Sampler([1,0],[1,0],'time',device='cpu')
except ValueError:pass
else:raise AssertionError('Missing background support must fail explicitly')
print('Sampling intervention tests passed')
