"""Bounded diagnostic of the cached-BF16 versus live-FP32 prefix boundary."""
from pathlib import Path
import json
import numpy as np
import torch
from semantic import source_query_factorial as work
from semantic.model import SemanticTail
from semantic.query_mode_adapter import QueryModeSemanticD4RT
from semantic.ytvis import load_video
from semantic.run import atomic_json
from tools.load_opend4rt import load_model

def main():
 root=work.ROOT;data=work.load_features('val');vid=min(data);r=data[vid];published=load_model(work.ready.CHECKPOINT,work.ready.CONFIG);torch.manual_seed(0);template=SemanticTail(published,26).eval();state=torch.load(root/'raw_GT_seed0.pt',weights_only=True,map_location='cpu');tail=work.make_tail(template,False,state);base=load_model(work.ready.CHECKPOINT,work.ready.CONFIG).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False);wrapper=QueryModeSemanticD4RT(base,26,'raw');wrapper.semantic_tail=tail;wrapper.eval();clip=next(c for c in json.loads(work.ready.MANIFEST.read_text())['val']if int(c['video_id'])==vid);video=load_video(clip)[None].cuda().to(torch.bfloat16);q={k:torch.from_numpy(r['query_'+k][:64].copy()).cuda()[None]for k in ('u','v','t_src','t_tgt','t_cam')}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  memory=base.encode_video(video);live=wrapper.source_prefix(video,q,memory);cached=r['raw'][:64][None];quantized=live.to(torch.bfloat16);z=tail(quantized,memory)['semantic_logits'][0].float().cpu().numpy()
 with np.load(root/'raw_GT_seed0_validation.npz')as a:expected=a['logits'][a['video_ids']==vid][:64]
 result={'video_id':vid,'live_prefix_dtype':str(live.dtype),'cached_training_prefix_dtype':str(cached.dtype),'memory_max_difference':float((memory-r['memory'][None]).abs().max()),'unquantized_prefix_max_difference':float((live-cached).abs().max()),'explicit_BF16_boundary_max_difference':float((quantized-cached).abs().max()),'explicit_BF16_logits_exact_saved_validation':bool(np.array_equal(z,expected)),'diagnostic_code_sha256':work.ready.sha('semantic/source_query_precision_diagnosis.py')}
 assert result['memory_max_difference']==0 and result['explicit_BF16_boundary_max_difference']==0 and result['explicit_BF16_logits_exact_saved_validation'];atomic_json(root/'precision_diagnosis.json',result);print(result,flush=True)
if __name__=='__main__':main()
