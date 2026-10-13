"""Fixed GT feature probes: linear vs one-hidden-layer head, 500 vs 2000 steps."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from semantic.model import SemanticHead
from semantic.sampling_ablation import Sampler, load_split, extended_diagnostic
from semantic.oracle_probe import read_oracle, file_hash
from semantic.run import atomic_json, classification_metrics

ROOT=Path('semantic_artifacts/head_budget_probe')
TRAIN=Path('semantic_artifacts/oracle_probe/features/oracle_coverage')
PROTOCOL={'version':1,'heads':['linear','mlp'],'budgets':[500,2000],'seeds':[0,1,2],
          'mlp':'Linear(1280,256) -> GELU -> Linear(256,26); no norm or dropout',
          'optimizer':'AdamW lr=0.001 weight_decay=0.01','batch':'128 foreground + 128 background',
          'sampling':'uniform query within label group','selection':'validation macro F1 every 50 steps',
          'budget_nesting':'one trajectory per head/seed, select best checkpoint within each budget',
          'diagnostic_training_gt':True,'test_does_not_select_head_or_budget':True}

class MLPHead(nn.Module):
    def __init__(self,input_dim=1280,hidden_dim=256,num_classes=26):
        super().__init__();self.hidden=nn.Linear(input_dim,hidden_dim);self.classifier=nn.Linear(hidden_dim,num_classes)
    def forward(self,x):return self.classifier(F.gelu(self.hidden(x.to(self.hidden.weight.dtype))))


def make_head(kind):
    if kind=='linear':return SemanticHead(num_classes=26)
    if kind=='mlp':return MLPHead()
    raise ValueError(kind)


def score(model,val):
    model.eval();predictions=[];targets=[]
    with torch.inference_mode():
        for c in val:
            predictions.append(model(torch.from_numpy(c['decoded']).float().to('cuda')).argmax(-1).cpu().numpy())
            targets.append(c['labels'])
    return classification_metrics(np.concatenate(predictions),np.concatenate(targets),26)['macro_f1_present']


def matched_classes(train_result,test_result):
    categories=[int(c) for c in test_result['support'] if c!='0']
    return {'categories':categories,
            'train_macro_recall':float(np.mean([train_result['per_class_recall'][str(c)] for c in categories])),
            'test_macro_recall':float(np.mean([test_result['per_class_recall'][str(c)] for c in categories]))}


def trajectory(kind,seed,features,labels,times,train,val,test,protocol, *, output_root=None):
    output_root=ROOT if output_root is None else Path(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    report_path=output_root/'runs'/f'{kind}_seed{seed}.json'
    if report_path.exists():return json.loads(report_path.read_text())
    torch.manual_seed(seed);model=make_head(kind).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    generator=torch.Generator(device='cuda').manual_seed(seed);sampler=Sampler(labels,times,'baseline')
    best=-1;best_state=None;best_step=None;history=[];snapshots={};hist=torch.zeros(8*26,dtype=torch.long,device='cuda')
    started=time.perf_counter()
    for step in range(1,max(protocol['budgets'])+1):
        model.train();fg,bg=sampler.draw(generator);idx=torch.cat([fg,bg])
        hist+=torch.bincount(sampler.times[idx]*26+sampler.labels[idx],minlength=8*26)
        optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(model(features[idx]),sampler.labels[idx]);loss.backward();optimizer.step()
        if step%50==0:
            value=score(model,val);history.append({'step':step,'val_macro_f1':value,'batch_loss':float(loss.detach())})
            if value>best:
                best=value;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if step in protocol['budgets']:
            snapshots[step]={'state':best_state,'best_step':best_step,'val':best,
                             'sampling_histogram':hist.reshape(8,26).cpu().tolist()}
            print(kind,seed,'budget',step,'best_val',best,'best_step',best_step,flush=True)
    results={}
    for budget,snapshot in snapshots.items():
        model.load_state_dict(snapshot['state']);model.eval()
        validation=extended_diagnostic(model,val);testing=extended_diagnostic(model,test);training=extended_diagnostic(model,train)
        results[str(budget)]={'head':kind,'seed':seed,'budget':budget,'parameter_count':sum(p.numel() for p in model.parameters()),
                             'best_step':snapshot['best_step'],'best_val_macro_f1':snapshot['val'],
                             'validation':validation,'test':testing,'train':training,
                             'matched_class_diagnostic':matched_classes(training,testing),
                             'sampling_histogram_time_class':snapshot['sampling_histogram']}
        checkpoint=output_root/'runs'/f'{kind}_{budget}_seed{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
        temp=checkpoint.with_suffix('.partial.pt')
        torch.save({'model':snapshot['state'],'head':kind,'num_classes':26,'budget':budget,'seed':seed,'protocol':protocol},temp);temp.replace(checkpoint)
        print(kind,seed,budget,'test foreground F1',testing['foreground_macro_f1_present'],flush=True)
    result={'head':kind,'seed':seed,'history':history,'budgets':results,'seconds':time.perf_counter()-started}
    atomic_json(report_path,result);return result


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    # Freeze input hashes and choices before seeing any new results.
    hashes={}
    for split,folder in [('train',TRAIN),('val',Path('semantic_artifacts/features_ovis/sam/val')),
                         ('test',Path('semantic_artifacts/features_ovis/sam/test'))]:
        h=hashlib.sha256()
        for p in sorted(folder.glob('*.npz')):h.update(p.name.encode());h.update(bytes.fromhex(file_hash(p)))
        hashes[split]=h.hexdigest()
    protocol=dict(PROTOCOL,feature_hashes=hashes,manifest_sha256=file_hash('semantic_artifacts/manifest.json'))
    file=ROOT/'protocol.json'
    if file.exists() and json.loads(file.read_text())!=protocol:raise ValueError('Protocol/input changed')
    atomic_json(file,protocol)
    train=read_oracle(TRAIN);val=load_split('val');test=load_split('test')
    if [len(c) for c in (train,val,test)]!=[60,20,20]:raise ValueError('Require 60/20/20')
    manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text())
    ids=[{int(c['video_id']) for c in clips} for clips in (train,val,test)]
    if ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]:raise ValueError('Split overlap')
    for index,split in enumerate(('train','val','test')):
        if ids[index]!={c['video_id'] for c in manifest[split]}:raise ValueError('Manifest mismatch')
    x=np.concatenate([c['decoded'] for c in train]);labels=np.concatenate([c['labels'] for c in train]);times=np.concatenate([c['xy'][:,2] for c in train])
    if not np.isfinite(x).all() or np.any((labels<0)|(labels>=26)):raise ValueError('Invalid training cache')
    features=torch.from_numpy(x).float().to('cuda');del x
    runs={kind:[trajectory(kind,seed,features,labels,times,train,val,test,protocol) for seed in PROTOCOL['seeds']] for kind in PROTOCOL['heads']}
    results={f'{kind}_{budget}':[r['budgets'][str(budget)] for r in runs[kind]] for kind in PROTOCOL['heads'] for budget in PROTOCOL['budgets']}
    selected=max(results,key=lambda k:np.mean([r['best_val_macro_f1'] for r in results[k]]))
    contrasts={}
    for treatment,control in (('mlp_500','linear_500'),('mlp_2000','linear_2000'),('linear_2000','linear_500'),('mlp_2000','mlp_500')):
        contrasts[f'{treatment}_minus_{control}']={}
        for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_accuracy','foreground_to_background_rate'):
            values=[results[treatment][i]['test'][metric]-results[control][i]['test'][metric] for i in range(3)]
            contrasts[f'{treatment}_minus_{control}'][metric]={'paired_deltas':values,'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1))}
    previous=json.loads(Path('semantic_artifacts/oracle_probe/summary.json').read_text())
    for i,r in enumerate(results['linear_500']):
        for metric in ('accuracy','macro_f1_present','foreground_macro_f1_present'):
            if abs(r['test'][metric]-previous['results']['oracle_coverage'][i]['test'][metric])>1e-12:raise ValueError('500-step baseline mismatch')
    for group,runs in results.items():
        for r in runs:
            hist=np.array(r['sampling_histogram_time_class'])
            if hist[:,0].sum()!=r['budget']*128 or hist[:,1:].sum()!=r['budget']*128:raise ValueError('Sampling budget mismatch')
            if r['best_step']>r['budget']:raise ValueError('Budget checkpoint leakage')
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'results':results,'contrasts':contrasts,
                                  'validation_selected_group':selected,
                                  'verification':{'linear_500_matches_oracle_probe':True,'all_sampling_budgets_valid':True,'budget_checkpoint_limits_valid':True}})
    print('DONE validation-selected',selected,flush=True)

if __name__=='__main__':main()
