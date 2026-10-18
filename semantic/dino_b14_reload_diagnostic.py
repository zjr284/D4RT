"""Bounded post-failure diagnosis, preserving the original R73 strict result."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from semantic import dino_b14_interface_probe as work
from semantic.model import GEOMETRY_KEYS
from semantic.dino_interface_probe import delta
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from tools.load_opend4rt import DEFAULT_VARIANT,load_model
ROOT=work.ROOT

def main():
 torch.backends.cuda.matmul.allow_tf32=False
 s=json.loads((ROOT/'summary.json').read_text());r=next(r for r in s['cases']if r['kind']=='B2'and r['recipe']=='gt32'and r['seed']==0)
 with np.load(r['output_path'])as z:a={k:z[k].copy()for k in z.files}
 clips=json.loads((ROOT/'manifest.json').read_text())['val'];videos=[]
 for c in clips:
  frames=[]
  for path in c['frames']:
   with Image.open(path)as image:pixels=np.asarray(image.convert('RGB').resize((128,128),Image.Resampling.BILINEAR),np.float32)
   frames.append(torch.from_numpy(pixels.copy()).permute(2,0,1)/255)
  videos.append(torch.stack(frames))
 video=torch.stack(videos).cuda().bfloat16();query={k:torch.from_numpy(a['query_'+k]).cuda()for k in ('u','v','t_src','t_tgt','t_cam')};p={'reason':'R73 independent B2 bare reload failed strict bitwise check; B1 and same-process interface passed','original_summary_sha256':file_hash(ROOT/'summary.json'),'script_sha256':file_hash('semantic/dino_b14_reload_diagnostic.py'),'arms':['parameter_requires_grad_true','parameter_requires_grad_false','parameter_requires_grad_true_repeat'],'repetitions':2,'training':False,'no_tolerance_change':True};atomic_json(ROOT/'reload_diagnostic_protocol.json',p);base=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml').to('cuda',dtype=torch.bfloat16).eval();rows=[]
 for arm,flag in [('parameter_requires_grad_true',True),('parameter_requires_grad_false',False),('parameter_requires_grad_true_repeat',True)]:
  base.requires_grad_(flag)
  for repeat in range(2):
   for changed in (False,True):
    q={k:v.clone()for k,v in query.items()}
    if changed:q['t_tgt']=(q['t_tgt']+1)%8;q['t_cam']=(q['t_cam']+3)%8
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=base({'video':video,'query':q})
    prefix='changed_reference'if changed else'reference';row={'arm':arm,'repeat':repeat,'changed_times':changed,'fields':{k:delta(out[k],torch.from_numpy(a[prefix+'__'+k]).cuda())for k in GEOMETRY_KEYS}};rows.append(row);print('reload diagnostic',arm,repeat,changed,{k:v['max_absolute_difference']for k,v in row['fields'].items()},flush=True)
 atomic_json(ROOT/'reload_diagnostic.json',{'protocol':p,'cases':rows,'all_inputs_and_weights_fixed':True,'original_strict_failure_retained':True})
if __name__=='__main__':main()
