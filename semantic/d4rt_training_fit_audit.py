"""Post-score frozen-adapter training fit and foreground/background diagnosis."""
from pathlib import Path
import fcntl,gc,json
import numpy as np
import torch
from semantic.model import SemanticTail
from semantic.run import atomic_json,classification_metrics
from semantic import d4rt_finetune_readiness as ready
from semantic import d4rt_controlled_finetune as source
from tools.load_opend4rt import load_model
ROOT=Path('semantic_artifacts/d4rt_training_fit_audit')

def describe(z,y):
 pred=z.argmax(1);fg=y>0;bg=y==0;mx=z.max(1);ce=mx+np.log(np.exp(z.astype(np.float64)-mx[:,None]).sum(1))-z[np.arange(len(y)),y];margin=z[:,0]-z[:,1:].max(1)
 return {'queries':len(y),'true_foreground':int(fg.sum()),'true_background':int(bg.sum()),'metrics':classification_metrics(pred,y,26),'predicted_foreground_fraction':float(np.mean(pred>0)),'foreground_correct_fraction':float(np.mean(pred[fg]==y[fg])),'background_false_positive_fraction':float(np.mean(pred[bg]>0)),'mean_CE':float(ce.mean()),'true_foreground_BG_margin_mean':float(margin[fg].mean()),'true_foreground_BG_margin_quantiles':np.quantile(margin[fg],[0,.25,.5,.75,1]).tolist()}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists()or(ROOT/'protocol.json').exists():raise RuntimeError('Inspect existing run; do not overwrite')
  old=json.loads((source.ROOT/'summary.json').read_text());verify=json.loads((source.ROOT/'verification.json').read_text());assert verify['summary_sha256']==ready.sha(source.ROOT/'summary.json')
  pub=json.loads((source.ROOT/'publication.json').read_text());assert pub['remote_contains_research_commit']
  manifest=json.loads(ready.MANIFEST.read_text());files=[source.ROOT/'summary.json',source.ROOT/'verification.json',ready.CHECKPOINT,ready.CONFIG,ready.MANIFEST,Path('semantic/model.py'),Path('semantic/d4rt_controlled_finetune.py'),Path('semantic/d4rt_training_fit_audit.py'),Path('semantic/OVIS_D4RT_TRAINING_FIT_AUDIT_PROTOCOL.md')]
  files+=list((ready.CACHE/'train').glob('*.npz'))+list((ready.CACHE/'val').glob('*.npz'))+[source.ROOT/f'{a}_seed{i}.pt'for a in source.ARMS for i in range(3)]+[source.ROOT/f'{a}_seed{i}_validation.npz'for a in source.ARMS for i in range(3)]
  protocol={'round':100,'post_score_descriptive_only':True,'source_commit':pub['research_commit'],'input_sha256':{str(p):ready.sha(p)for p in files},'train_query_chunk':64,'populations':['train_source','val_source','val_later']};atomic_json(ROOT/'protocol.json',protocol)
  train=[]
  for c in sorted(manifest['train'],key=lambda r:int(r['video_id'])):
   vid=int(c['video_id'])
   with np.load(ready.CACHE/'train'/f'{vid}.npz')as a:idx=a['xy'][:,2]==0;row={'q':torch.from_numpy(a['prelast'][idx].copy()).cuda().to(torch.bfloat16),'memory':torch.from_numpy(a['memory'].copy()).cuda().to(torch.bfloat16)[None],'labels':a['labels'][idx].astype(np.int64)}
   train.append((vid,row))
  val_t=np.concatenate([np.load(ready.CACHE/'val'/f"{c['video_id']}.npz")['xy'][:,2]for c in sorted(manifest['val'],key=lambda r:int(r['video_id']))])
  published=load_model(ready.CHECKPOINT,ready.CONFIG);results={a:{}for a in source.ARMS}
  for seed in range(3):
   torch.manual_seed(seed);template=SemanticTail(published,26).eval()
   for arm in source.ARMS:
    state=torch.load(source.ROOT/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);model=source.make_model(template,arm,state);scores=[];labels=[];ids=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
     for vid,r in train:
      for start in range(0,len(r['labels']),64):scores.append(model(r['q'][start:start+64][None],r['memory'])['semantic_logits'][0].float().cpu().numpy())
      labels.append(r['labels']);ids.extend([vid]*len(r['labels']))
    z=np.concatenate(scores);y=np.concatenate(labels);np.savez_compressed(ROOT/f'{arm}_seed{seed}_train.npz',logits=z,labels=y,video_ids=np.asarray(ids))
    with np.load(source.ROOT/f'{arm}_seed{seed}_validation.npz')as a:vz=a['logits'].copy();vy=a['labels'].copy()
    common=sorted((set(y)-{0})&(set(vy[val_t==0])-{0}));pop={'train_source':describe(z,y),'val_source':describe(vz[val_t==0],vy[val_t==0]),'val_later':describe(vz[val_t>0],vy[val_t>0])}
    common_means={k:float(np.mean([pop[k]['metrics']['per_class_f1'][str(int(c))]for c in common]))for k in ('train_source','val_source')}
    results[arm][str(seed)]={'populations':pop,'common_train_val_source_FG_classes':list(map(int,common)),'common_FG_macro':common_means,'train_minus_val_common_FG_macro':common_means['train_source']-common_means['val_source']}
    print('fit',seed,arm,'train FG',pop['train_source']['metrics']['foreground_macro_f1_present'],'val source FG',pop['val_source']['metrics']['foreground_macro_f1_present'],flush=True);del model;gc.collect();torch.cuda.empty_cache()
   del template;gc.collect();torch.cuda.empty_cache()
  for path,h in protocol['input_sha256'].items():assert ready.sha(path)==h
  atomic_json(ROOT/'summary.json',{'round':100,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'results':results,'output_sha256':{str(p):ready.sha(p)for p in ROOT.glob('*.npz')},'new_training_or_confirmation':False,'model_or_default_changed':False});print('DONE frozen training-fit audit',flush=True)
if __name__=='__main__':main()
