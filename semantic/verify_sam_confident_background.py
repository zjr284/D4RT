"""Independent SAM negative-mask erosion, fixed schedules and GT error description."""
from pathlib import Path
from collections import defaultdict
import json,os,tempfile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_api
from semantic.run import atomic_json
from semantic.verify_d4rt_controlled_finetune import sha
ROOT=Path('semantic_artifacts/sam_confident_background')
OLD=Path('semantic_artifacts/teacher_ovis')
FEATURE=Path('semantic_artifacts/source_query_factorial/features/train')


def main():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8';torch.use_deterministic_algorithms(True)
    s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text())
    assert s['round']==108 and s['protocol_sha256']==sha(ROOT/'protocol.json')
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    manifest=json.loads(Path('semantic_artifacts/manifest.json').read_text());ann=json.loads(Path('semantic_artifacts/ovis/annotations_train.json').read_text())
    by=defaultdict(list)
    for a in ann['annotations']:by[int(a['video_id'])].append(a)
    coords={};labels={};pools={};new_pixels=0;reference_clip=min(manifest['train'],key=lambda c:int(c['video_id']))
    for clip in sorted(manifest['train'],key=lambda c:int(c['video_id'])):
        vid=int(clip['video_id']);record=next(r for r in s['teachers']if r['video_id']==vid)
        assert sha(record['path'])==record['sha256']
        with np.load(OLD/f'{vid}.npz')as a:old=a['labels'].copy()
        with np.load(record['path'])as a:package={k:a[k].copy()for k in a.files}
        current=package['labels'];expected=old.copy();background={};conflicts=0
        ids={int(a['id']):int(a['category_id'])for a in by[vid]if clip['start']<len(a['segmentations'])and a['segmentations'][clip['start']]is not None and mask_api.decode(a['segmentations'][clip['start']]).any()}
        assert len(ids)==record['prompted_objects']
        for t in range(1,8):
            z=package[f'scaled_{t}'];assert z.dtype==np.float32 and z.shape[1:]==(128,128)and np.isfinite(z).all()
            objects=package[f'object_ids_{t}'];cats=package[f'categories_{t}'];assert set(map(int,objects))==set(ids)
            assert cats.tolist()==[ids[int(i)]for i in objects]
            safe=np.all(z < -2.,axis=0)
            windows=np.lib.stride_tricks.sliding_window_view(np.pad(safe,2,constant_values=False),(5,5))
            safe=windows.all(axis=(-2,-1));conflicts+=int(np.count_nonzero(safe&(old[t]>0)))
            expected[t][safe&(old[t]==-1)]=0;background[t]=np.argwhere(expected[t]==0)
        assert np.array_equal(current,expected)and np.array_equal(current[old>0],old[old>0])and np.array_equal(current[0],old[0])
        counts=[int(np.count_nonzero((old[t]==-1)&(current[t]==0)))for t in range(8)]
        assert counts==record['new_background_per_frame']and conflicts==record['old_positive_conflicts_with_new_negative_retained']
        new_pixels+=sum(counts);pools[vid]=background
        with np.load(FEATURE/f'{vid}.npz')as a:coords[vid]=a['xy'].copy();labels[vid]=a['labels'].copy()
    assert len(pools)==len(s['teachers'])==60 and new_pixels==s['new_background_pixels']
    # Independently rerun the first fixed clip from RGB/prompts for provenance.
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml','semantic_artifacts/weights/sam2.1_hiera_tiny.pt',device='cuda')
    vid=int(reference_clip['video_id']);start=reference_clip['start'];checked=[]
    with np.load(ROOT/'teachers'/f'{vid}.npz')as a:saved={k:a[k].copy()for k in a.files}
    with tempfile.TemporaryDirectory(prefix='d4rt_verify_sam_bg_')as directory:
        folder=Path(directory)
        for i,f in enumerate(reference_clip['frames']):(folder/f'{i:05d}.jpg').symlink_to(Path(f).resolve())
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=predictor.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=True)
            try:
                for a in by[vid]:
                    rle=a['segmentations'][start]if start<len(a['segmentations'])else None
                    if rle is None:continue
                    m=mask_api.decode(rle)
                    if m.ndim==3:m=m.any(2)
                    if m.any():predictor.add_new_mask(state,frame_idx=0,obj_id=int(a['id']),mask=m.astype(bool))
                for t,objects,logits in predictor.propagate_in_video(state):
                    if t==0:continue
                    if logits.ndim==4:logits=logits[:,0]
                    reduced=F.interpolate(logits.float()[:,None],size=(128,128),mode='bilinear',align_corners=False)[:,0].cpu().numpy()
                    assert np.array_equal(reduced,saved[f'scaled_{t}'])and np.array_equal(objects,saved[f'object_ids_{t}']);checked.append(int(t))
            finally:predictor.reset_state(state)
    assert sorted(checked)==list(range(1,8));del predictor;torch.cuda.empty_cache()
    paired=json.loads((ROOT/'paired_schedules.json').read_text());assert s['paired_schedules_sha256']==sha(ROOT/'paired_schedules.json')
    original=json.loads(Path('semantic_artifacts/d4rt_controlled_finetune/protocol.json').read_text())['schedules'];missing=[];draws=[];per_seed={}
    for seed in range(3):
        rng=np.random.Generator(np.random.PCG64(202610108+seed));assert len(paired['schedules'][str(seed)])==500
        for j,(row,source)in enumerate(zip(paired['schedules'][str(seed)],original[str(seed)])):
            for k,v in source.items():assert row[k]==v
            av=source['aux_video'];fg=coords[av][source['SAM_indices']];want=[];empty=[]
            for k,(_,_,t)in enumerate(fg):
                pool=pools[av][int(t)]
                if len(pool):y,x=pool[int(rng.integers(len(pool)))];want.append([int(x),int(y),int(t)]);draws.append((seed,av,int(t),int(x),int(y)))
                else:want.append(None);empty.append(k)
            gv=source['GT_video'];available=np.flatnonzero((coords[gv][:,2]==0)&(labels[gv]==0))
            additional=rng.choice(available,size=32,replace=len(available)<32).tolist()
            assert row['SAM_BG_xy']==want and row['extra_GT_BG_indices']==additional and row['missing_BG_pairs']==empty
            if empty:missing.append({'seed':seed,'step':j+1,'aux_video':av,'missing_pair_positions':empty})
        per_seed[str(seed)]={'selected_BG_draws':sum(r[0]==seed for r in draws)}
    assert missing==paired['missing']and len(missing)==s['missing_steps']and (not missing)==s['eligible_for_separate_matched_training_protocol']
    # GT is consulted only now, after all teacher labels and chosen points froze.
    selected={(v,t,x,y)for _,v,t,x,y in draws};truth={};clips={int(c['video_id']):c for c in manifest['train']}
    for v,t in sorted({(v,t)for v,t,_,_ in selected}):
        counts=np.zeros((128,128),np.int64);category=np.zeros_like(counts);frame=clips[v]['start']+t
        for a in by[v]:
            rle=a['segmentations'][frame]if frame<len(a['segmentations'])else None
            if rle is None:continue
            native=mask_api.decode(rle)
            if native.ndim==3:native=native.any(2)
            hit=np.asarray(Image.fromarray(native.astype(np.uint8)).resize((128,128),Image.Resampling.NEAREST))>0
            counts+=hit;category[hit]=a['category_id']
        truth[(v,t)]=np.where(counts==0,0,np.where(counts==1,category,-1))
    def diagnostic(points):
        values=[int(truth[(v,t)][y,x])for v,t,x,y in points];good=values.count(0);ignored=values.count(-1);wrong=len(values)-good-ignored
        return {'points':len(values),'GT_background':good,'GT_foreground':wrong,'GT_overlap_ignored':ignored,
            'BG_correct_fraction_among_nonoverlap':good/(good+wrong)if good+wrong else None,
            'wrong_foreground_by_category':{str(c):values.count(c)for c in range(1,26)if values.count(c)}}
    for seed in range(3):per_seed[str(seed)]['GT_diagnostic']=diagnostic([(v,t,x,y)for sd,v,t,x,y in draws if sd==seed])
    gt={'per_seed_draws':per_seed,'all_draws':diagnostic([(v,t,x,y)for _,v,t,x,y in draws]),'unique_selected_points':diagnostic(sorted(selected)),
        'used_to_filter_or_correct_labels':False,'natural_pixel_precision_evaluated':False}
    atomic_json(ROOT/'GT_background_diagnostic.json',gt)
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    for r in s['teachers']:assert sha(r['path'])==r['sha256']
    atomic_json(ROOT/'verification.json',{'round':108,'summary_sha256':sha(ROOT/'summary.json'),'verifier_sha256':sha('semantic/verify_sam_confident_background.py'),
        'all60_scaled_logits_negative_threshold_and_erosion_reconstructed_exact':True,'all60_old_positives_and_source_labels_preserved':True,
        'first_fixed_video_RGB_SAM_scaled_logits_all7_later_frames_recomputed_exact':vid,
        'all1500_paired_schedules_RNG_and_same_frame_points_exact':True,'all_original_inputs_unchanged':True,
        'GT_background_diagnostic_sha256':sha(ROOT/'GT_background_diagnostic.json'),'new_classifier_training_or_confirmation':False})
    print('DONE independent SAM background preparation verification',gt,flush=True)
if __name__=='__main__':main()
