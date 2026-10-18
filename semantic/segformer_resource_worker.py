"""Isolated-environment source-RGB preflight only; no semantic GT scoring."""
import gc,hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from semantic.segformer_features import ROOT,load_official,feature_maps,at_queries
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

def main():
 if(ROOT/'preflight.json').exists():raise RuntimeError('Completed SegFormer RGB preflight exists')
 torch.backends.cuda.matmul.allow_tf32=False;manifest=json.loads((ROOT/'source_clip.json').read_text());clip=manifest['clip'];assert clip['video_id']==258
 with np.load(manifest['query_cache'])as a:xy=a['xy'][::32].copy()
 frames=[]
 for path in clip['frames']:
  with Image.open(path)as image:array=np.asarray(image.convert('RGB').resize((448,448),Image.Resampling.BILINEAR),np.float32)
  frames.append(torch.from_numpy(array.copy()).permute(2,0,1)/255)
 video=torch.stack(frames).cuda();results={}
 for alias in ('mit','ade'):
  model=load_official(alias);assert all(not p.requires_grad for p in model.parameters());single=feature_maps(model,alias,video[:1]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();maps=feature_maps(model,alias,video);torch.cuda.synchronize();cost={'seconds':time.perf_counter()-t,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'parameters':sum(p.numel()for p in model.parameters())}
  for arm,value in maps.items():
   points=at_queries(value,xy);quantized=points.half().float();difference=(value[0]-single[arm][0]).abs();path=ROOT/'preflight_features'/f'{arm}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,xy=xy,decoded=quantized.half().cpu().numpy());results[arm]={'map_shape':list(value.shape),'point_shape':list(points.shape),'finite':bool(torch.isfinite(points).all()),'single_vs_T8_first_frame_max_difference':float(difference.max()),'single_vs_T8_first_frame_mean_difference':float(difference.mean()),'FP16_quantization_max_difference':float((points-quantized).abs().max()),'output_path':str(path),'output_sha256':file_hash(path),'cost':cost};assert results[arm]['finite'];print('SegFormer source-only preflight',arm,results[arm],flush=True)
  del model,maps,single;gc.collect();torch.cuda.empty_cache()
 versions={name:importlib.metadata.version(name)for name in ('transformers','huggingface-hub','tokenizers','safetensors','torch','numpy','pillow')};atomic_json(ROOT/'preflight.json',{'source_video_id':258,'query_count':64,'results':results,'versions':versions,'no_GT_labels_read_or_scored':True,'training_updates':0,'new_confirmation_groups':0});print('DONE isolated SegFormer/MIT RGB resource preflight',flush=True)
if __name__=='__main__':main()
