"""Reuse frozen training cohorts; inspect labels/resources without model scoring."""
from pathlib import Path
import fcntl,json
import numpy as np
from semantic import d4rt_finetune_readiness as ready
from semantic.run import atomic_json
from semantic.ytvis import load_dataset,sample_queries
ROOT=Path('semantic_artifacts/d4rt_training_union')
HISTORY=Path('semantic_artifacts/video_coverage_intervention')
PARENT=Path('semantic_artifacts/sam_background_matched')
TEACHER=Path('semantic_artifacts/sam_confident_background/teachers')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with(ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if(ROOT/'protocol.json').exists()or(ROOT/'summary.json').exists():raise RuntimeError('Existing run: inspect before explicit recovery')
        pub=json.loads((PARENT/'publication.json').read_text());assert pub['remote_contains_research_commit']
        assert json.loads((PARENT/'verification.json').read_text())['summary_sha256']==ready.sha(PARENT/'summary.json')
        old=json.loads((HISTORY/'manifest.json').read_text());original=json.loads(ready.MANIFEST.read_text());extra={}
        for group in ('random80','coverage80'):
            for c in old['training_sets'][group]:
                vid=int(c['video_id'])
                if vid in extra:assert extra[vid]==c
                extra[vid]=c
        assert len(extra)==100
        current={int(c['video_id']):c for c in original['train']};heldout={int(c['video_id'])for split in ('val','test')for c in original[split]}
        assert len(current)==60 and not set(extra)&(set(current)|heldout)
        merged={**current,**extra};assert len(merged)==160
        files=[ready.MANIFEST,ready.ANN,HISTORY/'manifest.json',HISTORY/'protocol.json',HISTORY/'summary.json',HISTORY/'verification.json',
            PARENT/'summary.json',PARENT/'verification.json',PARENT/'publication.json',Path('semantic/d4rt_training_union.py'),
            Path('semantic/OVIS_D4RT_TRAINING_UNION_PROTOCOL.md'),Path('semantic/ytvis.py'),Path('semantic/teacher.py')]
        for vid,c in merged.items():
            assert len(c['frames'])==8;files.extend(Path(f)for f in c['frames'])
            files.append((TEACHER if vid in current else HISTORY/'teacher')/f'{vid}.npz')
            if vid in current:files.append(Path('semantic_artifacts/source_query_factorial/features/train')/f'{vid}.npz')
        p={'round':110,'parent_research_commit':pub['research_commit'],'baseline_video_ids':sorted(current),'additional_video_ids':sorted(extra),
            'union_video_ids':sorted(merged),'excluded_current_val_test':sorted(heldout),'new_selection_or_model_scoring':False,
            'input_sha256':{str(f):ready.sha(f)for f in sorted(set(files))}}
        atomic_json(ROOT/'protocol.json',p);dataset=load_dataset('ovis',ready.ANN,Path('semantic_artifacts/ovis'))
        output=ROOT/'queries';output.mkdir(exist_ok=True);rows=[];points={name:[]for name in ('baseline60','additional100','union160')}
        for vid,c in sorted(merged.items()):
            path=(TEACHER if vid in current else HISTORY/'teacher')/f'{vid}.npz'
            with np.load(path)as a:maps=a['labels'].copy()
            assert maps.shape==(8,128,128)and np.array_equal(maps[0],dataset.gt_label_map(c,0))
            _,targets,positions=sample_queries(maps,256,42+vid);xy=positions.numpy();y=targets.numpy()
            if vid in current:
                with np.load(Path('semantic_artifacts/source_query_factorial/features/train')/f'{vid}.npz')as a:oldxy=a['xy'].copy();oldy=a['labels'].copy()
                assert np.array_equal(xy[xy[:,2]==0],oldxy[oldxy[:,2]==0])and np.array_equal(y[xy[:,2]==0],oldy[oldxy[:,2]==0])
                xy,y=oldxy,oldy
            source=xy[:,2]==0;assert source.sum()==256 and np.sum(y[source]>0)==128 and np.sum(y[source]==0)==128
            fg=(xy[:,2]>0)&(y>0);bg_counts={t:int(np.sum(maps[t]==0))for t in range(1,8)}
            matched=int(sum(bg_counts[int(t)]>0 for t in xy[fg,2]));out=output/f'{vid}.npz'
            np.savez_compressed(out,xy=xy.astype(np.int16),labels=y.astype(np.int16))
            sourcepoints=np.column_stack([np.full(256,vid),xy[source],y[source]])
            group='baseline60'if vid in current else'additional100';points[group].append(sourcepoints);points['union160'].append(sourcepoints)
            rows.append({'video_id':vid,'start':int(c['start']),'group':group,'teacher_path':str(path),'queries_path':str(out),'queries_sha256':ready.sha(out),
                'source_FG_classes':sorted(set(map(int,y[source][y[source]>0]))),'later_SAM_FG_queries':int(fg.sum()),
                'later_SAM_FG_unique_positions':len(np.unique(xy[fg],axis=0)),'later_BG_pixels':sum(bg_counts.values()),
                'later_BG_pixels_by_frame':{str(t):n for t,n in bg_counts.items()},'later_SAM_FG_queries_with_BG_frame':matched,
                'all_sampled_later_FG_have_same_frame_BG':bool(fg.sum()>0 and matched==fg.sum())})
            print('union resource',vid,group,int(fg.sum()),matched,flush=True)
        stats={}
        for group,arrays in points.items():
            a=np.concatenate(arrays);per={}
            for cls in range(26):
                selected=a[a[:,4]==cls];per[str(cls)]={'queries':len(selected),'unique_video_frame_xy':len(np.unique(selected[:,:4],axis=0)),
                    'videos':len(np.unique(selected[:,0]))}
            stats[group]={'videos':len(np.unique(a[:,0])),'queries':len(a),'per_class':per}
        atomic_json(ROOT/'manifest.json',{'dataset':original['dataset'],'baseline_train':list(current.values()),'additional_train':list(extra.values()),
            'train':[merged[v]for v in sorted(merged)],'val':original['val'],'test':original['test']})
        for path,h in p['input_sha256'].items():assert ready.sha(path)==h,path
        atomic_json(ROOT/'summary.json',{'round':110,'protocol_sha256':ready.sha(ROOT/'protocol.json'),'manifest_sha256':ready.sha(ROOT/'manifest.json'),
            'video_resources':rows,'source_GT_support':stats,'all160_source_GT_labels_match':True,
            'all160_later_auxiliary_pools_same_frame_pairable':all(r['all_sampled_later_FG_have_same_frame_BG']for r in rows),
            'new_teacher_generation_or_model_training_or_scoring':False,'all607_videos_previously_research_used':True,'default_changed':False})
        print('DONE fixed training union preparation',flush=True)
if __name__=='__main__':main()
