"""Independent reload of frozen adapters and training/validation population metrics."""
import copy,gc,json
from pathlib import Path
import numpy as np
import torch
from semantic.model import SemanticTail
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha,metrics
from tools.load_opend4rt import load_model,DEFAULT_VARIANT
ROOT=Path('semantic_artifacts/d4rt_training_fit_audit');SOURCE=Path('semantic_artifacts/d4rt_controlled_finetune');CACHE=Path('semantic_artifacts/features_ovis/sam')

def near(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:near(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):near(x,y)
 elif isinstance(a,float):assert abs(a-b)<1e-12,(a,b)
 else:assert a==b,(a,b)

def stats(z,y):
 pred=np.argmax(z,axis=-1);fg=np.where(y>0)[0];bg=np.where(y==0)[0];loss=np.logaddexp.reduce(z.astype(np.float64),axis=1)-z[np.arange(len(y)),y];margin=(z[:,0]-np.max(z[:,1:],axis=1))[fg]
 return {'queries':len(y),'true_foreground':len(fg),'true_background':len(bg),'metrics':metrics(y,pred)[0],'predicted_foreground_fraction':float(np.count_nonzero(pred)/len(y)),'foreground_correct_fraction':float(np.count_nonzero(pred[fg]==y[fg])/len(fg)),'background_false_positive_fraction':float(np.count_nonzero(pred[bg])/len(bg)),'mean_CE':float(np.mean(loss)),'true_foreground_BG_margin_mean':float(margin.mean()),'true_foreground_BG_margin_quantiles':np.percentile(margin,[0,25,50,75,100]).tolist()}

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());assert sha(ROOT/'protocol.json')==s['protocol_sha256']and p['round']==100
 for path,h in p['input_sha256'].items():assert sha(path)==h
 for path,h in s['output_sha256'].items():assert sha(path)==h
 manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text());train=[];val_time=[];val_y=[];val_ids=[]
 for c in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
  vid=int(c['video_id'])
  with np.load(CACHE/'train'/f'{vid}.npz')as a:sel=np.flatnonzero(a['xy'][:,2]==0);row={'q':torch.from_numpy(a['prelast'][sel].copy()).cuda().to(torch.bfloat16),'memory':torch.from_numpy(a['memory'].copy()).cuda().to(torch.bfloat16)[None],'y':a['labels'][sel].astype(np.int64)}
  assert len(sel)==256;train.append((vid,row))
 for c in sorted(manifest['val'],key=lambda c:int(c['video_id'])):
  vid=int(c['video_id'])
  with np.load(CACHE/'val'/f'{vid}.npz')as a:val_time.extend(a['xy'][:,2]);val_y.extend(a['labels']);val_ids.extend([vid]*len(a['labels']))
 t=np.asarray(val_time);vy=np.asarray(val_y);assert(len(t),sum(t==0),sum(t>0))==(40960,5120,35840)
 model=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml');verified=[]
 for seed in range(3):
  torch.manual_seed(seed);template=SemanticTail(model,26).eval()
  for arm in ('A','B','C','D'):
   state=torch.load(SOURCE/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);tail=copy.deepcopy(template).cuda().eval()
   with torch.no_grad():
    for n,q in tail.named_parameters():
     if n in state:q.copy_(state[n].to(q.device))
   with np.load(ROOT/f'{arm}_seed{seed}_train.npz')as a:z=a['logits'].copy();y=a['labels'].copy();ids=a['video_ids'].copy()
   offset=0
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for vid,r in train:
     for start in range(0,256,64):
      got=tail(r['q'][start:start+64][None],r['memory'])['semantic_logits'][0].float().cpu().numpy();assert np.array_equal(got,z[offset+start:offset+start+64])
     assert np.array_equal(y[offset:offset+256],r['y'])and np.all(ids[offset:offset+256]==vid);offset+=256
   assert offset==len(y)==15360
   with np.load(SOURCE/f'{arm}_seed{seed}_validation.npz')as a:vz=a['logits'].copy();assert np.array_equal(a['labels'],vy)and np.array_equal(a['video_ids'],val_ids)
   calculated={'train_source':stats(z,y),'val_source':stats(vz[t==0],vy[t==0]),'val_later':stats(vz[t>0],vy[t>0])};logged=s['results'][arm][str(seed)];near(calculated,logged['populations']);classes=sorted(map(int,(set(y)-{0})&(set(vy[t==0])-{0})));assert classes==logged['common_train_val_source_FG_classes'];means={key:float(np.mean([calculated[key]['metrics']['per_class_f1'][str(c)]for c in classes]))for key in ('train_source','val_source')};near(means,logged['common_FG_macro']);assert abs(means['train_source']-means['val_source']-logged['train_minus_val_common_FG_macro'])<1e-12;verified.append(f'{arm}_{seed}');print('Verified frozen fit audit',seed,arm,flush=True);del tail;gc.collect();torch.cuda.empty_cache()
  del template;gc.collect();torch.cuda.empty_cache()
 for path,h in p['input_sha256'].items():assert sha(path)==h
 atomic_json(ROOT/'verification.json',{'round':100,'summary_sha256':sha(ROOT/'summary.json'),'verifier_sha256':sha('semantic/verify_d4rt_training_fit_audit.py'),'all12_training_logits_recomputed_exact':verified,'training_queries_each':15360,'validation_source_queries_each':5120,'validation_later_queries_each':35840,'all36_population_metrics_and_BG_margins_rebuilt':True,'all12_common_FG_source_gaps_verified':True,'all_inputs_models_and_source_scores_unchanged':True,'new_training_or_confirmation':False});print('DONE independent training-fit audit verification',flush=True)
if __name__=='__main__':main()
