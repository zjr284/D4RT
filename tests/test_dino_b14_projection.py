import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from semantic.dino_b14_projection_control import project

def main():
 for b,s,expected in [([1.,0.],[-1.,1.],[0.,1.]),([1.,0.],[2.,1.],[2.,1.]),([1.,0.],[0.,1.],[0.,1.]),([0.,0.],[-2.,3.],[-2.,3.])]:
  base=torch.tensor(b);supp=torch.tensor(s);saved=base.clone();out,alpha=project(base,supp);assert torch.allclose(out,torch.tensor(expected),atol=1e-7)and torch.equal(base,saved)and torch.dot(base,out)>=-1e-7
 try:project(torch.tensor([1.,0.]),torch.tensor([float('nan'),1.]))
 except ValueError:pass
 else:raise AssertionError('Nonfinite gradient accepted')
 print('PASS fixed one-way projection: conflict/aligned/orthogonal/zero/finite and unchanged base')
if __name__=='__main__':main()
