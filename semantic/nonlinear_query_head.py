"""Small matched-width source-query heads; no changes to the deployed adapter."""
import torch
from torch import nn

class TwoLayerQueryHead(nn.Module):
 def __init__(self,nonlinear:bool):
  super().__init__();self.input=nn.Linear(768,32);self.output=nn.Linear(32,26);self.nonlinear=bool(nonlinear)
 def forward(self,features):
  hidden=self.input(features.to(self.input.weight.dtype))
  if self.nonlinear:hidden=torch.relu(hidden)
  return self.output(hidden)
