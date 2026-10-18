"""Explicit frozen B14 research adapters; existing deployment entry is unchanged."""
import json,subprocess
from pathlib import Path
import torch
from torch.nn import functional as F
from semantic.dino_native_interface import NativeDinoSemanticD4RT
from semantic.dino_interface import cuda_bf16
from semantic.model import SemanticHead
from semantic.dino_backbone_resource import load_b14,WEIGHTS,VENDOR
from semantic import dino_backbone_training as capacity
from semantic import dino_class_video_sampling as balanced
from semantic import dino_b14_auxiliary as auxiliary
from semantic.oracle_probe import file_hash
from tools.load_opend4rt import DEFAULT_VARIANT,load_model
RECIPES=('gt32','class_video','aux_sam')

def resolve_recipe(recipe,seed):
 if recipe not in RECIPES or type(seed)is not int or seed not in (0,1,2):raise ValueError('Explicit registered B14 recipe and seed0/1/2 required')
 module,arm=(capacity,'B14_gt32')if recipe=='gt32'else(balanced,'class_video')if recipe=='class_video'else(auxiliary,'aux_sam')
 summary=json.loads((module.ROOT/'summary.json').read_text());verified=json.loads((module.ROOT/'verification.json').read_text());selection=json.loads((module.ROOT/'frozen_selection.json').read_text())
 if file_hash(module.ROOT/'summary.json')!=verified['summary_sha256']or file_hash(module.ROOT/'frozen_selection.json')!=summary['frozen_selection_sha256']:raise ValueError('Source validation record changed')
 selected=selection['validation_selected_arm']
 if selected!=arm:raise ValueError('Expected the registered source-validation recipe')
 checkpoint=module.ckpt(arm,seed);sha=selection['checkpoint_sha256'][f'{arm}_{seed}'];offset=selection['offsets'][arm]
 if file_hash(checkpoint)!=sha or offset!=summary['results'][arm]['offset']:raise ValueError('Source checkpoint or offset changed')
 return checkpoint,sha,float(offset),module.ROOT

class B14ResearchD4RT(NativeDinoSemanticD4RT):
 def __init__(self,published_model,dino_model,background_offset):
  super().__init__(published_model,dino_model,background_offset)
  self.semantic_head=SemanticHead(hidden_dim=768,num_classes=26);self.requires_grad_(False);super().train(False)
 def load_head(self,path,expected_sha256=None):
  if expected_sha256 is not None and file_hash(path)!=expected_sha256:raise ValueError('Semantic checkpoint hash mismatch')
  state=torch.load(path,map_location='cpu',weights_only=True)['model']
  if tuple(state['classifier.weight'].shape)!=(26,768)or not all(torch.isfinite(v).all()for v in state.values()):raise ValueError('Finite 26-class768-dimensional head required')
  self.semantic_head.load_state_dict(state,strict=True);return self
 @torch.inference_mode()
 def native_features(self,rgb,query):
  if rgb.ndim!=5 or tuple(rgb.shape[2:])!=(3,448,448)or rgb.dtype!=torch.float32 or not torch.isfinite(rgb).all()or rgb.min()<0 or rgb.max()>1:raise ValueError('Finite native448 float32 RGB in0..1 required')
  mean=torch.tensor([.485,.456,.406],device=rgb.device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=rgb.device)[None,:,None,None];values=[]
  for batch in range(len(rgb)):
   image=(rgb[batch]-mean)/std
   with cuda_bf16(rgb.device):tokens=self.dino.forward_features(image)['x_norm_patchtokens']
   if tokens.shape!=(rgb.shape[1],1024,768):raise ValueError('B14 requires32x32 patch768 tokens')
   tokens=tokens.reshape(rgb.shape[1],32,32,768).permute(0,3,1,2).float();out=torch.empty(query['u'].shape[1],768,device=rgb.device,dtype=torch.float32)
   for source in query['t_src'][batch].unique(sorted=True).tolist():
    rows=torch.where(query['t_src'][batch]==source)[0];position=torch.stack([query['u'][batch,rows],query['v'][batch,rows]],-1).float()*127;grid=((position+.5)/128*2-1)[None,None]
    with torch.autocast(rgb.device.type,enabled=False):out[rows]=F.grid_sample(tokens[source:source+1],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
   values.append(out.half().float())
  return torch.stack(values)

def load_b14_research_adapter(recipe,seed=0,device='cuda'):
 path,sha,offset,_=resolve_recipe(recipe,seed);device=torch.device(device)
 if device.type!='cuda':raise ValueError('Registered B14 research inference requires CUDA')
 if file_hash(WEIGHTS)!='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73':raise ValueError('B14 backbone changed')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()!='7764ea0f912e53c92e82eb78a2a1631e92725fc8':raise ValueError('B14 vendor commit changed')
 subprocess.run(['git','diff','--quiet','HEAD'],cwd=VENDOR,check=True)
 from semantic.dino_sam_control import ROOT as foundation
 fp=json.loads((foundation/'summary.json').read_text())['protocol']['inherited_feature_coverage_protocol']['inherited_feature_protocol']
 if file_hash(DEFAULT_VARIANT/'opend4rt.ckpt')!=fp['d4rt_backbone_sha256']or file_hash(DEFAULT_VARIANT/'model.yaml')!=fp['d4rt_model_config_sha256']:raise ValueError('Geometry foundation changed')
 base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to(device,dtype=torch.bfloat16).eval();dino=load_b14().to(device)
 adapter=B14ResearchD4RT(base,dino,offset).to(device);adapter.load_head(path,sha);return adapter
