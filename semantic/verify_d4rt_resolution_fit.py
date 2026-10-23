"""Independent RLE, frozen-output reload, class-support and confusion audit."""
from pathlib import Path
from collections import defaultdict
import copy,csv,gc,json
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_api
from semantic.model import SemanticTail
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha,metrics
from semantic.verify_d4rt_training_fit_audit import stats,near
from tools.load_opend4rt import load_model,DEFAULT_VARIANT
ROOT=Path('semantic_artifacts/d4rt_resolution_fit')
PARENTS={128:Path('semantic_artifacts/source_query_factorial'),256:Path('semantic_artifacts/d4rt_input_resolution')}
ARMS=('source_head','source_GT','source_SAM','source_extraGT')


def independent_stats(z,y):
    out=stats(z,y);_,cm=metrics(y,z.argmax(1));per={}
    for c in range(26):
        support=int(cm[c,:].sum());predicted=int(cm[:,c].sum());tp=int(cm[c,c])
        fp=predicted-tp;fn=support-tp
        per[str(c)]={'support':support,'predicted':predicted,'TP':tp,'FP':fp,'FN':fn,
                    'F1':float(2*tp/(support+predicted))if support>0 else None}
    return {**out,'per_class':per,'confusion':cm.tolist()}


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text())
    assert p['round']==107 and s['round']==107 and s['protocol_sha256']==sha(ROOT/'protocol.json')
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    for path,h in s['outputs_sha256'].items():assert sha(path)==h,path
    manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text())
    annotations=json.loads(Path('semantic_artifacts/ovis/annotations_train.json').read_text());by=defaultdict(list)
    for a in annotations['annotations']:by[int(a['video_id'])].append(a)
    support={name:{'queries':0,'per_class':{str(c):{'queries':0,'unique_video_frame_xy':0,'videos':0}for c in range(26)}}for name in ('train_source','val_source','val_later')}
    unique={(name,c):set()for name in support for c in range(26)};videos={(name,c):set()for name in support for c in range(26)}
    train_labels=[];train_ids=[];val_labels=[];val_ids=[];val_t=[];gt_points=0
    for split in ('train','val'):
        for clip in sorted(manifest[split],key=lambda c:int(c['video_id'])):
            vid=int(clip['video_id'])
            with np.load(PARENTS[128]/'features'/split/f'{vid}.npz')as a:xy=a['xy'].copy();y=a['labels'].astype(np.int64)
            with np.load(PARENTS[256]/'features'/split/f'{vid}.npz')as a:assert np.array_equal(xy,a['xy'])and np.array_equal(y,a['labels'])
            if split=='train':
                counts=np.zeros((128,128),dtype=np.int64);cats=np.zeros_like(counts)
                for ann in by[vid]:
                    if clip['start']>=len(ann['segmentations']):continue
                    rle=ann['segmentations'][clip['start']]
                    if rle is None:continue
                    native=mask_api.decode(rle)
                    if native.ndim==3:native=native.any(2)
                    mask=np.asarray(Image.fromarray(native.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST))>0
                    counts+=mask;cats[mask]=ann['category_id']
                expected=np.where(counts==0,0,np.where(counts==1,cats,-1));source=xy[:,2]==0
                with np.load(Path('semantic_artifacts/teacher_ovis')/f'{vid}.npz')as a:assert np.array_equal(expected,a['labels'][0])
                assert np.array_equal(expected[xy[source,1],xy[source,0]],y[source]);gt_points+=int(source.sum())
                train_labels.extend(y[source]);train_ids.extend([vid]*int(source.sum()))
            else:val_labels.extend(y);val_ids.extend([vid]*len(y));val_t.extend(xy[:,2])
            for i,(x,yy,t)in enumerate(xy):
                if split=='train'and t!=0:continue
                name='train_source'if split=='train'else('val_source'if t==0 else'val_later');cls=int(y[i]);assert 0<=cls<26
                support[name]['queries']+=1;support[name]['per_class'][str(cls)]['queries']+=1
                unique[(name,cls)].add((vid,int(t),int(x),int(yy)));videos[(name,cls)].add(vid)
    for name in support:
        for c in range(26):support[name]['per_class'][str(c)]['unique_video_frame_xy']=len(unique[(name,c)]);support[name]['per_class'][str(c)]['videos']=len(videos[(name,c)])
    near(support,s['support']);assert gt_points==15360
    common=sorted(set.intersection(*[{c for c in range(1,26)if support[name]['per_class'][str(c)]['queries']>0}for name in support]))
    train_y=np.asarray(train_labels);train_vids=np.asarray(train_ids);vy=np.asarray(val_labels);v_ids=np.asarray(val_ids);vt=np.asarray(val_t)
    published=load_model(DEFAULT_VARIANT/'opend4rt.ckpt',DEFAULT_VARIANT/'model.yaml');rebuilt={};verified=[]
    for size,root in PARENTS.items():
        train=[]
        for clip in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
            vid=int(clip['video_id'])
            with np.load(root/'features'/'train'/f'{vid}.npz')as a:
                idx=np.flatnonzero(a['xy'][:,2]==0);x=torch.from_numpy(a['source'][idx].copy()).cuda().to(torch.bfloat16);memory=torch.from_numpy(a['memory'].copy()).cuda().to(torch.bfloat16)[None]
            assert len(idx)==256;train.append((vid,x,memory))
        for seed in range(3):
            torch.manual_seed(seed);template=SemanticTail(published,26).eval()
            for arm in ARMS:
                key=f'{arm}_{size}_seed{seed}';tail=copy.deepcopy(template).cuda().eval()
                state=torch.load(root/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True)
                with torch.no_grad():
                    for n,q in tail.named_parameters():
                        if n in state:q.copy_(state[n].to(q.device))
                with np.load(ROOT/f'{key}_train.npz')as a:z=a['logits'].copy();y=a['labels'].copy();ids=a['video_ids'].copy()
                assert np.array_equal(y,train_y)and np.array_equal(ids,train_vids)
                offset=0
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for vid,x,m in train:
                        for start in range(0,256,64):
                            actual=tail(x[start:start+64][None],m)['semantic_logits'][0].float().cpu().numpy()
                            assert np.array_equal(actual,z[offset+start:offset+start+64])
                        assert np.all(ids[offset:offset+256]==vid);offset+=256
                assert offset==len(z)==15360
                with np.load(root/f'{arm}_seed{seed}_validation.npz')as a:
                    vz=a['logits'].copy();assert np.array_equal(a['labels'],vy)and np.array_equal(a['video_ids'],v_ids)
                pops={'train_source':independent_stats(z,y),'val_source':independent_stats(vz[vt==0],vy[vt==0]),'val_later':independent_stats(vz[vt>0],vy[vt>0])}
                record=s['results'][key];near(pops,record['populations']);assert record['common_FG_classes']==common
                means={name:float(np.mean([pops[name]['per_class'][str(c)]['F1']for c in common]))for name in pops}
                near(means,record['common_FG_macro'])
                assert abs(record['train_minus_val_source_common_FG']-means['train_source']+means['val_source'])<1e-12
                assert abs(record['train_minus_val_later_common_FG']-means['train_source']+means['val_later'])<1e-12
                rebuilt[key]=pops;verified.append(key);print('Verified frozen fit/classes',key,flush=True)
                del tail;gc.collect();torch.cuda.empty_cache()
            del template;gc.collect();torch.cuda.empty_cache()
        del train;gc.collect();torch.cuda.empty_cache()
    with(ROOT/'per_class.csv').open(newline='')as f:rows=list(csv.DictReader(f))
    assert len(rows)==24*3*26 and len({(r['model'],r['population'],r['category'])for r in rows})==len(rows)
    for row in rows:
        name=row['population'];c=row['category'];calculated=rebuilt[row['model']][name]['per_class'][c]
        for k in ('queries','unique_video_frame_xy','videos'):assert int(row[k])==support[name]['per_class'][c][k]
        for k in ('predicted','TP','FP','FN'):assert int(row[k])==calculated[k]
        if calculated['F1']is None:assert row['F1']==''
        else:assert abs(float(row['F1'])-calculated['F1'])<1e-12
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    atomic_json(ROOT/'verification.json',{'round':107,'summary_sha256':sha(ROOT/'summary.json'),'verifier_sha256':sha('semantic/verify_d4rt_resolution_fit.py'),
        'all60_full_first_frame_GT_maps_independent_RLE_counts_exact':True,'training_GT_query_points_verified':gt_points,
        'all24_training_logits_recomputed_exact':verified,'all72_population_metrics_confusions_and_common_class_gaps_rebuilt':True,
        'all26_classes_unique_pixel_and_video_support_and1872_CSV_rows_verified':True,'original_inputs_and_models_unchanged':True,'new_training_or_confirmation':False})
    print('DONE independent frozen resolution fit audit',flush=True)
if __name__=='__main__':main()
