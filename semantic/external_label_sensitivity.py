"""Descriptive semantic-consensus label sensitivity on already observed clips."""
from collections import defaultdict
import fcntl,gc,json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as coco
import torch
from semantic import dino_native_rgb_control as rgb
from semantic.external_coco_adapter import ANN,label_map
from semantic.external_nonperson_confirmation import nonperson_queries,ARMS,PANEL,PRIOR as FIRST
from semantic.external_semantic_confirmation import contrast
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.semantic_feature_control import infer,save_npz,patch_at_queries
from semantic.background_calibration import predictions
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json,classification_metrics
ROOT=Path('semantic_artifacts/external_label_sensitivity')
OLD=Path('semantic_artifacts/external_nonperson_confirmation')
AUDIT=Path('semantic_artifacts/external_mask_coverage')
MAPPING=Path('semantic_artifacts/external_coco_adapter/summary.json')
def consensus_map(annotations,mapping,height,width,size=128):
    active=np.zeros((26,size,size),bool);occupied=np.zeros((size,size),bool);crowd=np.zeros_like(occupied)
    for a in annotations:
        m=coco.decode(a['segmentation']);assert m.shape==(height,width)
        m=np.asarray(Image.fromarray(m.astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST),dtype=bool);occupied|=m
        if a.get('iscrowd',0):crowd|=m
        elif a['category_id']in mapping:active[mapping[a['category_id']]]|=m
    out=np.full((size,size),-1,np.int16);out[~occupied]=0;unique=(active.sum(0)==1)&~crowd
    owner=active.argmax(0);out[unique]=owner[unique]
    return out

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'summary.json').exists():raise RuntimeError('Completed label sensitivity exists')
        torch.backends.cuda.matmul.allow_tf32=False
        s=json.loads((OLD/'summary.json').read_text());p0=s['protocol'];v=json.loads((AUDIT/'verification.json').read_text());assert file_hash(AUDIT/'summary.json')==v['summary_sha256']
        assert torch.__version__==p0['torch_version']and torch.version.cuda==p0['cuda_version']and torch.cuda.get_device_name(0)==p0['device']
        inputs=dict(p0['input_sha256']);inputs.update(s['feature_hashes']);paths=[OLD/'summary.json',OLD/'verification.json',OLD/'manifest.json',OLD/'query_coverage.json',AUDIT/'summary.json',AUDIT/'verification.json',Path('semantic/external_label_sensitivity.py'),Path('semantic/OVIS_EXTERNAL_LABEL_SENSITIVITY_PROTOCOL.md')]+sorted(OLD.glob('*/logits/*.npz'));inputs.update({str(x):file_hash(x)for x in paths})
        for path,h in inputs.items():assert file_hash(path)==h
        p={'round':55,'scope':'post-score descriptive label-policy sensitivity, not new confirmation','selection':p0['selection'],'primary':['source_gt32','source_gt'],'secondary':[['native_sam','source_gt'],['lambda0','source_gt']],'bootstrap_seed':20261055,'bootstrap_repetitions':2000,'training_updates':0,'calibration_updates':0,'input_sha256':inputs};atomic_json(ROOT/'protocol.json',p)
        clips=json.loads((OLD/'manifest.json').read_text())['confirmation'];atomic_json(ROOT/'manifest.json',{'confirmation':clips})
        payload=json.loads(ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(MAPPING.read_text())['shared_category_mapping'].items()};by=defaultdict(list)
        for a in payload['annotations']:by[a['image_id']].append(a)
        model=rgb.prior.resolution.load_dino();packages=[];coverage=[];prefix={}
        for clip in clips:
            group=clip['group'];points=[];targets=[]
            for slot,im in enumerate(clip['frames']):
                new=consensus_map(by[im['id']],mapping,im['height'],im['width']);old=label_map(by[im['id']],mapping,im['height'],im['width']);valid=old>0;assert np.array_equal(new[valid],old[valid])
                xy,y=nonperson_queries(new,im['id'],slot);points.append(xy);targets.append(y)
                coverage.append({'group':group,'time':slot,'image_id':im['id'],'old_nonperson_pixels':int((old>1).sum()),'new_nonperson_pixels':int((new>1).sum()),'query_count':len(y),'distinct_query_pixels':len(np.unique(xy[:,:2],axis=0)),'query_class_counts':{str(int(c)):int(n)for c,n in zip(*np.unique(y,return_counts=True))}})
            xy=np.concatenate(points);y=np.concatenate(targets)
            with np.load(OLD/'features'/f'{group}.npz')as a:oldxy=a['xy'].copy();oldfeatures=a['decoded'].copy()
            joined=np.concatenate([oldxy,xy])
            if len(joined):
                video=rgb.load_native_video({'frames':[str(PANEL/'images'/im['file_name'])for im in clip['frames']]}).cuda();tokens=rgb.prior.resolution.extract_tokens(model,video,448)
                with torch.inference_mode():features=patch_at_queries(tokens,joined).half().cpu().numpy()
                assert np.array_equal(features[:len(oldxy)],oldfeatures);newfeatures=features[len(oldxy):];del video,tokens
            else:newfeatures=np.empty((0,384),np.float16)
            prefix[group]={'old_query_count':len(oldxy),'bitwise_equal':True};package={'decoded':newfeatures,'xy':xy,'labels':y,'video_id':np.int64(int(group))};packages.append(package);save_npz(ROOT/'features'/f'{group}.npz',**package)
            print('sensitivity features',group,len(y),flush=True)
        del model;gc.collect();torch.cuda.empty_cache();atomic_json(ROOT/'query_coverage.json',coverage);atomic_json(ROOT/'old_prefix_checks.json',prefix)
        results={}
        for arm in ARMS:
            runs=[];offset=p['selection'][arm]['offset']
            for seed in range(3):
                out=infer('dinov2',seed,'sensitivity',packages,checkpoint_path=checkpoint(arm,seed),output_root=ROOT/arm);pred=predictions(out['logits'],offset);per={}
                for c in clips:
                    rows=out['video_ids']==int(c['group']);per[c['group']]=classification_metrics(pred[rows],out['labels'][rows],26)if rows.any()else None
                runs.append({'seed':seed,'raw':classification_metrics(predictions(out['logits'],0),out['labels'],26),'calibrated':classification_metrics(pred,out['labels'],26),'per_video':per})
                print('sensitivity logits',arm,seed,runs[-1]['calibrated']['macro_f1_present'],flush=True)
            results[arm]={'offset':offset,'runs':runs}
        target=np.concatenate([a['labels']for a in packages]);counts={c['group']:len(a['labels'])for c,a in zip(clips,packages)}
        for path,h in inputs.items():assert file_hash(path)==h
        atomic_json(ROOT/'summary.json',{'protocol':p,'results':results,'primary_contrast':contrast(*p['primary'],results,clips,p),'secondary_contrasts_unadjusted':{a+'_minus_'+b:contrast(a,b,results,clips,p)for a,b in p['secondary']},'actual_support':{str(int(c)):int(n)for c,n in zip(*np.unique(target,return_counts=True))},'total_queries':len(target),'query_counts':counts,'empty_groups':[g for g,n in counts.items()if not n],'empty_frames':sum(not r['query_count']for r in coverage),'distinct_frame_query_pixels':sum(r['distinct_query_pixels']for r in coverage),'old_prefix_checks':prefix,'query_coverage_sha256':file_hash(ROOT/'query_coverage.json'),'feature_hashes':{str(ROOT/'features'/f'{c["group"]}.npz'):file_hash(ROOT/'features'/f'{c["group"]}.npz')for c in clips},'prior_scores_unchanged':True,'deployment_unchanged':'native_gt'})
        print('DONE label sensitivity',flush=True)
if __name__=='__main__':main()
