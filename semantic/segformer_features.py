"""Frozen official SegFormer/MIT features, no OVIS class prompts or GT inputs."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path('semantic_artifacts/segformer_resource_control')
ARMS=('mit_encoder','ade_encoder','ade_decoder')

def load_official(alias):
 from transformers import SegformerConfig,SegformerForImageClassification,SegformerForSemanticSegmentation
 spec=json.loads((ROOT/'model_specs.json').read_text())[alias];config=SegformerConfig.from_dict(json.loads(Path(spec['config_path']).read_text()));assert config.hidden_sizes==[32,64,160,256]and config.depths==[2,2,2,2]
 cls=SegformerForImageClassification if alias=='mit'else SegformerForSemanticSegmentation;model=cls(config)
 if spec['weight_format']=='safetensors':
  from safetensors.torch import load_file
  state=load_file(spec['weight_path'],device='cpu')
 else:state=torch.load(spec['weight_path'],map_location='cpu',weights_only=True)
 assert isinstance(state,dict)and all(isinstance(k,str)and torch.is_tensor(v)for k,v in state.items());model.load_state_dict(state,strict=True);model=model.cuda().eval().requires_grad_(False);return model

@torch.inference_mode()
def feature_maps(model,alias,video):
 assert video.dtype==torch.float32 and video.ndim==4 and video.shape[1:]==(3,448,448)
 mean=torch.tensor([.485,.456,.406],device=video.device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=video.device)[None,:,None,None];image=(video-mean)/std;captured=[];handle=None
 if alias=='ade':handle=model.decode_head.classifier.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
 try:
  with torch.autocast('cuda',dtype=torch.bfloat16):
   if alias=='ade':out=model(pixel_values=image,output_hidden_states=True,return_dict=True);last=out.hidden_states[-1]
   else:last=model.segformer(pixel_values=image,output_hidden_states=True,return_dict=True).hidden_states[-1]
 finally:
  if handle is not None:handle.remove()
 if last.ndim==3:last=last.transpose(1,2).reshape(len(video),256,14,14)
 assert last.shape==(len(video),256,14,14)and torch.isfinite(last).all();result={alias+'_encoder':last.float()}
 if alias=='ade':
  assert len(captured)==1 and captured[0].shape==(len(video),256,112,112)and out.logits.shape==(len(video),150,112,112)and torch.isfinite(captured[0]).all();result['ade_decoder']=captured[0].float()
 return result

@torch.inference_mode()
def at_queries(maps,xy):
 result=torch.empty(len(xy),256,device='cuda',dtype=torch.float32)
 for t in sorted(set(map(int,xy[:,2]))):
  ix=torch.from_numpy(__import__('numpy').flatnonzero(xy[:,2]==t)).cuda();points=torch.from_numpy(xy[ix.cpu().numpy(),:2]).float().cuda();grid=(((points+.5)/128)*2-1)[None,None];result[ix]=F.grid_sample(maps[t:t+1],grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0].T
 return result
