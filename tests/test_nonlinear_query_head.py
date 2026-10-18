import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from semantic.nonlinear_query_head import TwoLayerQueryHead
for seed in range(3):
 torch.manual_seed(seed);a=TwoLayerQueryHead(False)
 torch.manual_seed(seed);b=TwoLayerQueryHead(True)
 assert sum(p.numel()for p in a.parameters())==sum(p.numel()for p in b.parameters())==25466
 assert all(torch.equal(x,y)for x,y in zip(a.state_dict().values(),b.state_dict().values()))
torch.manual_seed(9191);head=TwoLayerQueryHead(False).double();w=torch.randn(26,768,dtype=torch.float64);bias=torch.randn(26,dtype=torch.float64);x=torch.randn(7,768,dtype=torch.float64)
with torch.no_grad():
 head.input.weight.zero_();head.input.bias.zero_();head.output.weight.zero_();head.output.bias.zero_();head.input.weight[:26]=w;head.input.bias[:26]=bias;head.output.weight[:,:26]=torch.eye(26,dtype=torch.float64)
assert torch.allclose(head(x),x@w.T+bias,atol=1e-10,rtol=0)
head=TwoLayerQueryHead(True).double()
with torch.no_grad():
 for p in head.parameters():p.zero_()
 head.input.weight[0,0]=1;head.output.weight[0,0]=1
x=torch.zeros(3,768,dtype=torch.float64);x[:,0]=torch.tensor([-1.,0.,1.]);x.requires_grad_(True);out=head(x)
assert torch.equal(out[:,0],torch.tensor([0.,0.,1.],dtype=torch.float64))and (out[0,0]+out[2,0]-2*out[1,0])==1
out.sum().backward();assert torch.equal(x.grad[:,0],torch.tensor([0.,0.,1.],dtype=torch.float64))
print('PASS matched initialization/parameters, full linear representability, nonlinearity and boundary gradients')
