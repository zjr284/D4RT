"""2x2 time matching / foreground class balancing on fixed frozen features."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from semantic.model import SemanticHead
from semantic.run import atomic_json, evaluate_head
from semantic.background_ablation import diagnostic

ROOT = Path('semantic_artifacts/sampling_ablation')
CACHE = Path('semantic_artifacts/background_ablation/features/sam')
VARIANTS = ('baseline', 'time', 'class', 'time_class')
PROTOCOL = {
    'version': 1, 'variants': list(VARIANTS), 'seeds': [0,1,2], 'steps':500,
    'foreground_batch':128, 'background_batch':128,
    'time_matching':'background source-time counts exactly match sampled foreground in each batch',
    'class_balance':'sample present foreground class uniformly, then sample query uniformly within class',
    'optimizer':'AdamW', 'lr':0.001, 'weight_decay':0.01,
    'selection':'validation macro F1 every 50 steps; ties retain earlier checkpoint',
    'fixed_features_labels_and_test_queries':True,
}

class Sampler:
    def __init__(self, labels, times, variant, device='cuda'):
        if variant not in VARIANTS: raise ValueError(variant)
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        self.times = torch.as_tensor(times, dtype=torch.long, device=device)
        self.variant = variant
        self.foreground = torch.where(self.labels > 0)[0]
        self.background = torch.where(self.labels == 0)[0]
        self.classes = self.labels[self.foreground].unique(sorted=True)
        self.by_class = [torch.where(self.labels == c)[0] for c in self.classes]
        self.by_time = [torch.where((self.labels == 0) & (self.times == t))[0] for t in range(8)]
        if not len(self.foreground) or not len(self.background): raise ValueError('Both label groups required')
        if variant in ('time','time_class'):
            for t in self.times[self.foreground].unique().tolist():
                if not len(self.by_time[t]): raise ValueError(f'No background queries for foreground source time {t}')

    def draw(self, generator, count=128):
        device = self.labels.device
        if self.variant in ('class','time_class'):
            choices = torch.randint(len(self.classes), (count,), device=device, generator=generator)
            fg = torch.empty(count, dtype=torch.long, device=device)
            for c,pool in enumerate(self.by_class):
                positions = torch.where(choices == c)[0]
                fg[positions] = pool[torch.randint(len(pool), (len(positions),), device=device, generator=generator)]
        else:
            fg = self.foreground[torch.randint(len(self.foreground), (count,), device=device, generator=generator)]
        if self.variant in ('time','time_class'):
            bg = torch.empty_like(fg)
            source = self.times[fg]
            for t,pool in enumerate(self.by_time):
                positions = torch.where(source == t)[0]
                if len(positions):
                    bg[positions] = pool[torch.randint(len(pool), (len(positions),), device=device, generator=generator)]
        else:
            bg = self.background[torch.randint(len(self.background), (count,), device=device, generator=generator)]
        return fg,bg


def load_split(split):
    result=[]
    for path in sorted((CACHE/split).glob('*.npz')):
        with np.load(path, allow_pickle=False) as p:
            keys = ('decoded','labels','xy','video_id','alternate_decoded')
            result.append({k:p[k].copy() for k in keys if k in p.files})
    return result


def extended_diagnostic(model, clips):
    result=diagnostic(model, clips)
    counts=np.zeros((26,26),dtype=np.int64)
    foreground_only_counts=np.zeros_like(counts)
    with torch.inference_mode():
        for clip in clips:
            logits=model(torch.from_numpy(clip['decoded']).float().to('cuda'))
            predicted=logits.argmax(-1).cpu().numpy()
            foreground_prediction=logits[:,1:].argmax(-1).cpu().numpy()+1
            np.add.at(foreground_only_counts,(clip['labels'].astype(int),foreground_prediction),1)
            np.add.at(counts,(clip['labels'].astype(int),predicted),1)
    support=counts.sum(axis=1)
    result['confusion_matrix']=counts.tolist()
    result['per_class_recall']={str(c):float(counts[c,c]/support[c]) for c in range(26) if support[c]}
    result['foreground_to_background_rate']=float(counts[1:,0].sum()/support[1:].sum())
    result['foreground_only_argmax_accuracy']=float(np.trace(foreground_only_counts[1:,1:])/support[1:].sum())
    result['foreground_only_argmax_macro_recall']=float(np.mean([foreground_only_counts[c,c]/support[c] for c in range(1,26) if support[c]]))
    result['foreground_accuracy']=float(np.trace(counts[1:,1:])/support[1:].sum())
    result['foreground_macro_recall']=float(np.mean([counts[c,c]/support[c] for c in range(1,26) if support[c]]))
    return result


def train_one(variant,seed,features,labels,times,val,test, *, output_root=None, run_name=None, protocol=None):
    output_root = ROOT if output_root is None else Path(output_root)
    run_name = variant if run_name is None else run_name
    protocol = PROTOCOL if protocol is None else protocol
    torch.manual_seed(seed)
    model=SemanticHead(num_classes=26).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
    generator=torch.Generator(device='cuda').manual_seed(seed)
    sampler=Sampler(labels,times,variant)
    histogram=torch.zeros(8*26,dtype=torch.long,device='cuda')
    best=-1; state=None; best_step=None; history=[]
    started=time.perf_counter()
    for step in range(1,501):
        model.train(); fg,bg=sampler.draw(generator)
        index=torch.cat([fg,bg])
        histogram += torch.bincount(sampler.times[index]*26+sampler.labels[index],minlength=8*26)
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(features[index]),sampler.labels[index]); loss.backward();optimizer.step()
        if step%50==0:
            score=evaluate_head(model,val,'decoded')['macro_f1_present']
            history.append({'step':step,'val_macro_f1':score})
            if score>best:
                best=score;best_step=step;state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state);model.eval()
    report={'variant':run_name,'seed':seed,'best_step':best_step,'best_val_macro_f1':best,
            'validation':extended_diagnostic(model,val), 'test':extended_diagnostic(model,test),
            'sampling_histogram_time_class':histogram.reshape(8,26).cpu().tolist(),
            'history':history,'seconds':time.perf_counter()-started}
    checkpoint=output_root/'runs'/f'{run_name}_seed{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    tmp=checkpoint.with_suffix('.partial.pt')
    torch.save({'model':state,'num_classes':26,'variant':run_name,'seed':seed,'protocol':protocol},tmp);tmp.replace(checkpoint)
    atomic_json(checkpoint.with_suffix('.json'),report)
    print(run_name,seed,'val',best,'foreground F1',report['test']['foreground_macro_f1_present'],flush=True)
    return report


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol=dict(PROTOCOL,cache=str(CACHE),manifest_sha256=hashlib.sha256(Path('semantic_artifacts/manifest.json').read_bytes()).hexdigest())
    fingerprint=hashlib.sha256()
    for split in ('train','val','test'):
        for file in sorted((CACHE/split).glob('*.npz')):
            fingerprint.update(f'{split}/{file.name}'.encode())
            with file.open('rb') as f:
                while block:=f.read(1024*1024): fingerprint.update(block)
    protocol['cache_sha256']=fingerprint.hexdigest()
    file=ROOT/'protocol.json'
    if file.exists() and json.loads(file.read_text())!=protocol:raise ValueError('Protocol or data changed')
    atomic_json(file,protocol)
    train=load_split('train');val=load_split('val');test=load_split('test')
    if [len(c) for c in (train,val,test)]!=[60,20,20]:raise ValueError('Require full 60/20/20')
    ids=[{int(c['video_id']) for c in s} for s in (train,val,test)]
    if ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]:raise ValueError('Split overlap')
    labels=np.concatenate([c['labels'] for c in train]);times=np.concatenate([c['xy'][:,2] for c in train])
    inputs=np.concatenate([c['decoded'] for c in train]);del train
    if not np.isfinite(inputs).all() or np.any((labels<0)|(labels>=26)):raise ValueError('Invalid training cache')
    features=torch.from_numpy(inputs).float().to('cuda');del inputs
    histogram=np.zeros((8,26),dtype=np.int64);np.add.at(histogram,(times,labels),1)
    atomic_json(ROOT/'training_distribution.json',{'time_class_counts':histogram.tolist(),'foreground_class_counts':histogram[:,1:].sum(axis=0).tolist()})
    results={}
    for variant in VARIANTS:
        results[variant]=[]
        for seed in PROTOCOL['seeds']:
            file=ROOT/'runs'/f'{variant}_seed{seed}.json'
            if file.exists():report=json.loads(file.read_text())
            else:report=train_one(variant,seed,features,labels,times,val,test)
            if 'foreground_only_argmax_accuracy' not in report['test']:
                model=SemanticHead(num_classes=26).to('cuda').eval()
                model.load_state_dict(torch.load(ROOT/'runs'/f'{variant}_seed{seed}.pt',map_location='cpu',weights_only=True)['model'])
                report['validation']=extended_diagnostic(model,val)
                report['test']=extended_diagnostic(model,test)
                atomic_json(file,report)
            results[variant].append(report)
    chosen=max(VARIANTS,key=lambda v:np.mean([r['best_val_macro_f1'] for r in results[v]]))
    contrasts={}
    for treatment,control in (('time','baseline'),('class','baseline'),('time_class','class'),('time_class','time')):
        contrasts[f'{treatment}_minus_{control}']={}
        for metric in ('macro_f1_present','foreground_macro_f1_present','foreground_to_background_rate','foreground_macro_recall'):
            d=[results[treatment][i]['test'][metric]-results[control][i]['test'][metric] for i in range(3)]
            contrasts[f'{treatment}_minus_{control}'][metric]={'paired_deltas':d,'mean':float(np.mean(d)),'std':float(np.std(d,ddof=1))}
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'results':results,'contrasts':contrasts,
                                  'validation_selected_variant':chosen})
    print('DONE validation_selected_variant',chosen,flush=True)

if __name__=='__main__':main()
