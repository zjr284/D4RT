"""Append high-confidence SAM background to old positive-only teachers, never GT-fix."""
from pathlib import Path
import argparse,fcntl,gc,json,os,tempfile,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_api
from semantic import d4rt_finetune_readiness as ready
from semantic.teacher import confident_labels,SAM_CHECKPOINT_SHA256
from semantic.ytvis import load_dataset
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/sam_confident_background')
PARENT=Path('semantic_artifacts/d4rt_resolution_fit')
CHECKPOINT=Path('semantic_artifacts/weights/sam2.1_hiera_tiny.pt')
SCHEDULE=Path('semantic_artifacts/d4rt_controlled_finetune/protocol.json')


def produce(predictor,dataset,clip):
    vid=int(clip['video_id']);start=int(clip['start']);prompts=[]
    for ann in dataset.annotations.get(vid,[]):
        if start>=len(ann['segmentations'])or ann['segmentations'][start]is None:continue
        m=mask_api.decode(ann['segmentations'][start])
        if m.ndim==3:m=m.any(2)
        if m.any():prompts.append((int(ann['id']),int(ann['category_id']),m.astype(bool)))
    assert prompts
    with np.load(ready.TEACHER/f'{vid}.npz')as a:old=a['labels'].copy()
    assert np.array_equal(old[0],dataset.gt_label_map(clip,0))
    labels=old.copy();package={};conflict=0;frames=[];by={i:c for i,c,_ in prompts}
    with tempfile.TemporaryDirectory(prefix='d4rt_sam_bg_frames_')as directory:
        folder=Path(directory)
        for i,path in enumerate(clip['frames']):(folder/f'{i:05d}.jpg').symlink_to(Path(path).resolve())
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=predictor.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=True)
            try:
                for i,_,m in prompts:predictor.add_new_mask(state,frame_idx=0,obj_id=i,mask=m)
                for t,object_ids,logits in predictor.propagate_in_video(state):
                    if t==0:continue
                    assert 1<=t<8;frames.append(int(t));cats=[by[int(i)]for i in object_ids]
                    if logits.ndim==4:assert logits.shape[1]==1;logits=logits[:,0]
                    scaled=F.interpolate(logits[:,None].float(),size=(128,128),mode='bilinear',align_corners=False)[:,0]
                    proposed=confident_labels(logits,cats,include_background=True)
                    background=proposed==0;conflict+=int(np.sum(background&(old[t]>0)))
                    labels[t][background&(old[t]==-1)]=0
                    package[f'scaled_{t}']=scaled.cpu().numpy();package[f'object_ids_{t}']=np.asarray(object_ids,dtype=np.int64)
                    package[f'categories_{t}']=np.asarray(cats,dtype=np.int16)
            finally:predictor.reset_state(state)
    assert sorted(frames)==list(range(1,8))and np.array_equal(labels[old>0],old[old>0])and np.array_equal(labels[0],old[0])
    return labels,package,{'video_id':vid,'prompted_objects':len(prompts),'new_background_per_frame':[int(np.sum((old[t]==-1)&(labels[t]==0)))for t in range(8)],
        'old_positive_conflicts_with_new_negative_retained':conflict,'source_GT_unchanged':True,'all_old_positive_labels_unchanged':True}


def schedule_pairs(manifest):
    old=json.loads(SCHEDULE.read_text())['schedules'];coords={};labels={};background={}
    for c in manifest['train']:
        vid=int(c['video_id'])
        with np.load(Path('semantic_artifacts/source_query_factorial/features/train')/f'{vid}.npz')as a:coords[vid]=a['xy'].copy();labels[vid]=a['labels'].copy()
        with np.load(ROOT/'teachers'/f'{vid}.npz')as a:new=a['labels'].copy()
        background[vid]={t:np.argwhere(new[t]==0)for t in range(1,8)}
    out={};missing=[]
    for seed in range(3):
        rng=np.random.default_rng(202610108+seed);rows=[]
        for j,row in enumerate(old[str(seed)]):
            vid=row['aux_video'];fg=coords[vid][row['SAM_indices']];bg=[];empty=[]
            for k,(_,_,t)in enumerate(fg):
                pool=background[vid][int(t)]
                if len(pool):y,x=pool[int(rng.integers(len(pool)))];bg.append([int(x),int(y),int(t)])
                else:bg.append(None);empty.append(k)
            gv=row['GT_video'];pool=np.flatnonzero((coords[gv][:,2]==0)&(labels[gv]==0));extra=rng.choice(pool,32,replace=len(pool)<32).tolist()
            rows.append({**row,'SAM_BG_xy':bg,'extra_GT_BG_indices':extra,'missing_BG_pairs':empty})
            if empty:missing.append({'seed':seed,'step':j+1,'aux_video':vid,'missing_pair_positions':empty})
        out[str(seed)]=rows
    return out,missing


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8';torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True,exist_ok=True)
    with(ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(ROOT/'summary.json').exists():raise RuntimeError('Completed run; do not regenerate')
        pub=json.loads((PARENT/'publication.json').read_text());assert pub['remote_contains_research_commit']
        assert json.loads((PARENT/'verification.json').read_text())['summary_sha256']==ready.sha(PARENT/'summary.json')
        assert ready.sha(CHECKPOINT)==SAM_CHECKPOINT_SHA256
        manifest=json.loads(ready.MANIFEST.read_text());dataset=load_dataset('ovis',ready.ANN,Path('semantic_artifacts/ovis'))
        files=[CHECKPOINT,ready.MANIFEST,ready.ANN,SCHEDULE,PARENT/'summary.json',PARENT/'verification.json',PARENT/'publication.json',
            Path('semantic/sam_confident_background.py'),Path('semantic/OVIS_SAM_CONFIDENT_BACKGROUND_PROTOCOL.md'),Path('semantic/teacher.py')]
        files +=[ready.TEACHER/f"{c['video_id']}.npz"for c in manifest['train']]
        files +=[Path('semantic_artifacts/source_query_factorial/features/train')/f"{c['video_id']}.npz"for c in manifest['train']]
        files +=[Path(f)for c in manifest['train']for f in c['frames']]
        path=ROOT/'protocol.json'
        if path.exists():
            assert args.resume,'Existing run requires explicit --resume';p=json.loads(path.read_text())
            for f,h in p['input_sha256'].items():assert ready.sha(f)==h,f
        else:
            assert not args.resume
            p={'round':108,'parent_research_commit':pub['research_commit'],'input_sha256':{str(f):ready.sha(f)for f in sorted(set(files))},
                'deterministic_algorithms':True,'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'foreground_labels_frozen':True,'new_classification_training':False}
            atomic_json(path,p)
        from sam2.build_sam import build_sam2_video_predictor
        predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(CHECKPOINT),device='cuda')
        directory=ROOT/'teachers';directory.mkdir(exist_ok=True);rows=[]
        ledger=json.loads((ROOT/'progress.json').read_text())if(ROOT/'progress.json').exists()else{}
        for c in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
            vid=int(c['video_id']);target=directory/f'{vid}.npz'
            if target.exists():
                assert args.resume and str(vid)in ledger and ledger[str(vid)]['sha256']==ready.sha(target)
                rows.append(ledger[str(vid)]);continue
            labels,package,metadata=produce(predictor,dataset,c)
            temporary=target.with_suffix('.partial.npz');np.savez_compressed(temporary,labels=labels,**package);os.replace(temporary,target)
            row={**metadata,'path':str(target),'sha256':ready.sha(target)};rows.append(row);ledger[str(vid)]=row
            atomic_json(ROOT/'progress.json',ledger);print('teacherBG',vid,row['new_background_per_frame'],flush=True)
        del predictor;gc.collect();torch.cuda.empty_cache()
        schedules,missing=schedule_pairs(manifest);atomic_json(ROOT/'paired_schedules.json',{'schedules':schedules,'missing':missing,'seed_base':202610108})
        for f,h in p['input_sha256'].items():assert ready.sha(f)==h,f
        atomic_json(ROOT/'summary.json',{'round':108,'protocol_sha256':ready.sha(path),'teachers':rows,'paired_schedules_sha256':ready.sha(ROOT/'paired_schedules.json'),
            'eligible_for_separate_matched_training_protocol':not missing,'missing_steps':len(missing),'all_old_positive_labels_preserved':True,
            'new_background_pixels':sum(sum(r['new_background_per_frame'])for r in rows),'no_classification_training_or_validation_scoring':True,
            'GT_later_used_for_generation_or_selection':False,'new_confirmation':False,'default_changed':False})
        print('DONE SAM background preparation; missing steps',len(missing),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        ROOT.mkdir(parents=True,exist_ok=True)
        atomic_json(ROOT/f'failure_{time.time_ns()}.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'no_automatic_label_or_threshold_relaxation':True})
        raise
