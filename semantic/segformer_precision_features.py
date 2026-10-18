"""New explicit arithmetic controls; round81 frozen helper is left unchanged."""
import torch
from semantic.segformer_features import at_queries,load_official

@torch.inference_mode()
def feature_maps(model,alias,video,precision):
    if precision not in ('fp32','bf16'):
        raise ValueError('Explicit fp32 or bf16 arithmetic required')
    if video.dtype != torch.float32 or video.ndim != 4 or video.shape[1:] != (3,448,448):
        raise ValueError('Expected native448 float32 source RGB')
    mean=torch.tensor([.485,.456,.406],device=video.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=video.device)[None,:,None,None]
    captured=[]
    handle=None
    if alias=='ade':
        handle=model.decode_head.classifier.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            if alias=='ade':
                output=model(pixel_values=(video-mean)/std,output_hidden_states=True,return_dict=True)
                last=output.hidden_states[-1]
            else:
                last=model.segformer(pixel_values=(video-mean)/std,output_hidden_states=True,return_dict=True).hidden_states[-1]
    finally:
        if handle is not None:
            handle.remove()
    if last.ndim==3:
        last=last.transpose(1,2).reshape(len(video),256,14,14)
    if last.shape!=(len(video),256,14,14):
        raise ValueError('Unexpected encoder stage shape')
    result={alias+'_encoder':last.float()}
    if alias=='ade':
        if len(captured)!=1 or captured[0].shape!=(len(video),256,112,112):
            raise ValueError('Unexpected fused decoder shape')
        result['ade_decoder']=captured[0].float()
    if not all(torch.isfinite(value).all()for value in result.values()):
        raise ValueError('Nonfinite frozen feature map')
    return result
