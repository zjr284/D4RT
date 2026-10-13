import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from semantic.head_budget_probe import MLPHead

torch.manual_seed(9);model=MLPHead(input_dim=8,hidden_dim=16,num_classes=3)
x=torch.randn(7,8,requires_grad=True);out=model(x)
assert out.shape==(7,3)
out.square().sum().backward()
assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
for p in model.parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
reloaded=MLPHead(input_dim=8,hidden_dim=16,num_classes=3);reloaded.load_state_dict(model.state_dict())
assert torch.equal(reloaded(x.detach()),model(x.detach()))
a=torch.randn(4,8);b=torch.randn(4,8)
assert not torch.allclose(model((a+b)/2),(model(a)+model(b))/2,atol=1e-5)
print('Nonlinear head gradients, reload and nonlinearity tests passed')
