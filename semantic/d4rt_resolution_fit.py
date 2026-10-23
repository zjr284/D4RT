"""Frozen 24-model fit/transfer and class-support description; no training."""
from pathlib import Path
import csv, fcntl, gc, json
import numpy as np
import torch
from semantic.model import SemanticTail
from semantic.run import atomic_json
from semantic.d4rt_training_fit_audit import describe
from semantic.source_query_factorial import make_tail
from semantic import d4rt_finetune_readiness as ready
from semantic.ytvis import load_dataset
from tools.load_opend4rt import load_model

ROOT=Path('semantic_artifacts/d4rt_resolution_fit')
PARENTS={128:Path('semantic_artifacts/source_query_factorial'),256:Path('semantic_artifacts/d4rt_input_resolution')}
ARMS=('source_head','source_GT','source_SAM','source_extraGT')


def details(z,y):
    out=describe(z,y);pred=z.argmax(1);cm=np.zeros((26,26),dtype=np.int64)
    np.add.at(cm,(y,pred),1);per={}
    for c in range(26):
        tp=int(cm[c,c]);fp=int(cm[:,c].sum()-tp);fn=int(cm[c,:].sum()-tp)
        per[str(c)]={'support':int(cm[c].sum()),'predicted':int(cm[:,c].sum()),'TP':tp,'FP':fp,'FN':fn,
                     'F1':float(2*tp/(2*tp+fp+fn))if cm[c].sum()else None}
    return {**out,'per_class':per,'confusion':cm.tolist()}


def identities(manifest):
    rows={};parts={k:[]for k in ('train_source','val_source','val_later')}
    for split in ('train','val'):
        for c in sorted(manifest[split],key=lambda c:int(c['video_id'])):
            vid=int(c['video_id'])
            with np.load(PARENTS[128]/'features'/split/f'{vid}.npz')as a:xy=a['xy'].copy();y=a['labels'].astype(np.int64)
            with np.load(PARENTS[256]/'features'/split/f'{vid}.npz')as a:assert np.array_equal(xy,a['xy'])and np.array_equal(y,a['labels'])
            for name,mask in ([('train_source',xy[:,2]==0)]if split=='train'else [('val_source',xy[:,2]==0),('val_later',xy[:,2]>0)]):
                parts[name].append(np.column_stack([np.full(mask.sum(),vid),xy[mask],y[mask]]))
    for name,arrays in parts.items():
        a=np.concatenate(arrays);per={}
        for cls in range(26):
            chosen=a[a[:,4]==cls]
            per[str(cls)]={'queries':len(chosen),'unique_video_frame_xy':len(np.unique(chosen[:,:4],axis=0)),
                           'videos':len(np.unique(chosen[:,0]))}
        rows[name]={'queries':len(a),'per_class':per}
    return rows


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with(ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(ROOT/'protocol.json').exists()or(ROOT/'summary.json').exists():raise RuntimeError('Inspect existing run; do not overwrite')
        parent=PARENTS[256]
        assert json.loads((parent/'verification.json').read_text())['summary_sha256']==ready.sha(parent/'summary.json')
        pub=json.loads((parent/'publication.json').read_text());assert pub['remote_contains_research_commit']
        manifest=json.loads(ready.MANIFEST.read_text());files=[ready.CHECKPOINT,ready.CONFIG,ready.MANIFEST,ready.ANN,
            Path('semantic/d4rt_resolution_fit.py'),Path('semantic/OVIS_D4RT_RESOLUTION_FIT_PROTOCOL.md'),Path('semantic/model.py'),
            Path('semantic/d4rt_training_fit_audit.py'),Path('semantic/source_query_factorial.py')]
        for root in PARENTS.values():
            files += [root/'summary.json',root/'verification.json',root/'publication.json']
            files += [root/f'{a}_seed{s}{suffix}'for a in ARMS for s in range(3)for suffix in('.pt','_validation.npz')]
            files += list((root/'features'/'train').glob('*.npz'))+list((root/'features'/'val').glob('*.npz'))
        files += [ready.TEACHER/f"{c['video_id']}.npz"for c in manifest['train']]
        p={'round':107,'parent_research_commit':pub['research_commit'],'post_score_description':True,'train_chunk_size':64,
           'input_sha256':{str(f):ready.sha(f)for f in sorted(set(files))},'populations':['train_source','val_source','val_later']}
        atomic_json(ROOT/'protocol.json',p)
        dataset=load_dataset('ovis',ready.ANN,Path('semantic_artifacts/ovis'))
        for c in manifest['train']:
            vid=int(c['video_id']);gt=dataset.gt_label_map(c,0)
            with np.load(ready.TEACHER/f'{vid}.npz')as a:assert np.array_equal(gt,a['labels'][0])
            with np.load(PARENTS[128]/'features'/'train'/f'{vid}.npz')as a:
                mask=a['xy'][:,2]==0;xy=a['xy'][mask];assert np.array_equal(gt[xy[:,1],xy[:,0]],a['labels'][mask])
        support=identities(manifest)
        assert [support[k]['queries']for k in p['populations']]==[15360,5120,35840]
        common=sorted(set.intersection(*[{int(k)for k,v in support[name]['per_class'].items()if int(k)>0 and v['queries']>0}for name in p['populations']]))
        val_frames=np.concatenate([np.load(PARENTS[128]/'features'/'val'/f"{c['video_id']}.npz")['xy'][:,2]for c in sorted(manifest['val'],key=lambda c:int(c['video_id']))])
        published=load_model(ready.CHECKPOINT,ready.CONFIG);results={}
        for size,root in PARENTS.items():
            train=[]
            for c in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
                vid=int(c['video_id'])
                with np.load(root/'features'/'train'/f'{vid}.npz')as a:
                    mask=a['xy'][:,2]==0;train.append((vid,torch.from_numpy(a['source'][mask].copy()).cuda().to(torch.bfloat16),
                        torch.from_numpy(a['memory'].copy()).cuda().to(torch.bfloat16)[None],a['labels'][mask].astype(np.int64)))
            for seed in range(3):
                torch.manual_seed(seed);template=SemanticTail(published,26).eval()
                for arm in ARMS:
                    tail=make_tail(template,arm=='source_head',torch.load(root/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True));zs=[];ys=[];ids=[]
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        for vid,x,m,y in train:
                            zs.extend(tail(x[j:j+64][None],m)['semantic_logits'][0].float().cpu().numpy()for j in range(0,len(y),64))
                            ys.append(y);ids.extend([vid]*len(y))
                    z=np.concatenate(zs);y=np.concatenate(ys);path=ROOT/f'{arm}_{size}_seed{seed}_train.npz'
                    np.savez_compressed(path,logits=z,labels=y,video_ids=np.asarray(ids))
                    with np.load(root/f'{arm}_seed{seed}_validation.npz')as a:vz=a['logits'].copy();vy=a['labels'].copy()
                    pops={'train_source':details(z,y),'val_source':details(vz[val_frames==0],vy[val_frames==0]),'val_later':details(vz[val_frames>0],vy[val_frames>0])}
                    means={name:float(np.mean([pops[name]['per_class'][str(cls)]['F1']for cls in common]))for name in pops}
                    results[f'{arm}_{size}_seed{seed}']={'populations':pops,'common_FG_classes':common,'common_FG_macro':means,
                        'train_minus_val_source_common_FG':means['train_source']-means['val_source'],
                        'train_minus_val_later_common_FG':means['train_source']-means['val_later']}
                    atomic_json(ROOT/'progress.json',results)
                    print('fit',size,seed,arm,[pops[k]['metrics']['foreground_macro_f1_present']for k in p['populations']],flush=True)
                    del tail;gc.collect();torch.cuda.empty_cache()
                del template;gc.collect();torch.cuda.empty_cache()
            del train;gc.collect();torch.cuda.empty_cache()
        with(ROOT/'per_class.csv').open('w',newline='')as f:
            fields=['model','population','category','queries','unique_video_frame_xy','videos','predicted','TP','FP','FN','F1']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for model,row in results.items():
                for pop,value in row['populations'].items():
                    for cls,stats in value['per_class'].items():
                        writer.writerow({'model':model,'population':pop,'category':cls,**support[pop]['per_class'][cls],
                            **{k:stats[k]for k in('predicted','TP','FP','FN','F1')}})
        for path,h in p['input_sha256'].items():assert ready.sha(path)==h,path
        atomic_json(ROOT/'summary.json',{'round':107,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'support':support,'results':results,
            'no_training_calibration_or_new_confirmation':True,'default_changed':False,
            'outputs_sha256':{str(f):ready.sha(f)for f in ROOT.glob('*.npz')}|{str(ROOT/'per_class.csv'):ready.sha(ROOT/'per_class.csv')}})
        print('DONE frozen resolution fit/class-support audit',flush=True)

if __name__=='__main__':main()
