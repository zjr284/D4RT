"""Metadata-fixed non-Person panel and RGB-only screening; no new model output."""
from collections import defaultdict,Counter
import fcntl
import json
import math
import os
from pathlib import Path,PurePosixPath
import sqlite3
import zipfile
os.environ.setdefault('MPLCONFIGDIR','/tmp/d4rt-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from semantic.external_coco_adapter import ANN,OVIS
from semantic.external_holdout_readiness import category_mapping
from semantic.ovis_frame_index import fingerprint
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_class_panel')
PREVIOUS=Path('semantic_artifacts/external_semantic_confirmation')
GROUPS=Path('semantic_artifacts/external_coco_adapter/groups.json')
COHORT=Path('semantic_artifacts/external_rgb_screen/manifest.json')
LOCAL=Path('semantic_artifacts/external_local_rgb')
INDEX=Path('semantic_artifacts/ovis_frame_index')
ARCHIVE=Path('semantic_artifacts/external_archive_download')
SEED=20261052


def select_panel(clips,excluded,annotations,categories,ovis_categories):
    candidates=[c for c in clips if c['group']not in excluded]
    mapping=category_mapping(categories,ovis_categories)
    class_ids=sorted(set(mapping.values())-{1})
    image_group={im['id']:c['group']for c in candidates for im in c['frames']}
    pools={cid:set()for cid in class_ids}
    for ann in annotations:
        cid=mapping.get(ann['category_id'])
        if cid in pools and ann['image_id']in image_group and not ann.get('iscrowd',0)and ann.get('area',0)>0:
            pools[cid].add(image_group[ann['image_id']])
    order=[candidates[int(i)]['group']for i in np.random.default_rng(SEED).permutation(len(candidates))]
    chosen=[];by_class={}
    for cid in class_ids:
        group=next((g for g in order if g in pools[cid]),None)
        by_class[str(cid)]={'eligible_group_count':len(pools[cid]),'selected_group':group}
        if group is not None and group not in chosen:chosen.append(group)
    indexed={c['group']:c for c in candidates}
    return {'seed':SEED,'eligible_model_unevaluated_pool':len(candidates),'excluded_original80':sorted(excluded),
        'selection_by_class':by_class,'panel':[indexed[g]for g in chosen]},mapping


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'materials.json').exists():raise RuntimeError('Panel materials already exist; review, do not recreate')
        verification=json.loads((PREVIOUS/'verification.json').read_text())
        assert file_hash(PREVIOUS/'summary.json')==verification['summary_sha256']
        selection=json.loads((PREVIOUS/'frozen_selection.json').read_text())['selection']
        archive_record=json.loads((ARCHIVE/'summary.json').read_text());av=json.loads((ARCHIVE/'verification.json').read_text())
        assert file_hash(ARCHIVE/'summary.json')==av['summary_sha256']and archive_record['source_error']is None
        source=Path(archive_record['archive_path'])
        assert file_hash(source)==archive_record['archive_sha256']
        paths=[PREVIOUS/'summary.json',PREVIOUS/'verification.json',PREVIOUS/'frozen_selection.json',GROUPS,COHORT,ANN,OVIS,
            LOCAL/'image_records.json',LOCAL/'summary.json',LOCAL/'verification.json',INDEX/'index.sqlite',INDEX/'summary.json',INDEX/'verification.json',
            ARCHIVE/'summary.json',ARCHIVE/'verification.json',source,Path('semantic/external_class_panel.py'),Path('semantic/OVIS_EXTERNAL_CLASS_PANEL_PROTOCOL.md'),
            Path('semantic/ovis_frame_index.py'),Path('semantic/external_holdout_readiness.py')]
        for arm,record in selection.items():
            for seed in range(3):
                path=checkpoint(arm,seed);assert file_hash(path)==record['checkpoint_sha256'][str(seed)];paths.append(path)
        protocol={'round':52,'new_panel_seed':SEED,'no_new_model_or_mask_evaluation':True,'selection':selection,
            'source_score_files_not_parsed':True,'input_sha256':{str(p):file_hash(p)for p in paths}}
        pp=ROOT/'protocol.json'
        if pp.exists():assert json.loads(pp.read_text())==protocol
        else:atomic_json(pp,protocol)
        clips=json.loads(GROUPS.read_text())['groups'];cohort=json.loads(COHORT.read_text())
        excluded={c['group']for role in ('candidate_confirmation','provisional_reserve')for c in cohort[role]}
        assert len(clips)==833 and len(excluded)==80
        payload=json.loads(ANN.read_bytes());ovis_categories=json.loads(OVIS.read_bytes())['categories']
        manifest,mapping=select_panel(clips,excluded,payload['annotations'],payload['categories'],ovis_categories)
        assert manifest['eligible_model_unevaluated_pool']==753
        atomic_json(ROOT/'manifest.json',manifest)
        connection=sqlite3.connect(f'file:{INDEX / "index.sqlite"}?mode=ro',uri=True)
        native=[json.loads(r[0])for r in connection.execute('SELECT record FROM frames ORDER BY path')];connection.close()
        old_external=json.loads((LOCAL/'image_records.json').read_text())
        references=[{'dataset':'OVIS','path':str(Path('semantic_artifacts/ovis/train')/r['path']),**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in native]
        references.extend({'dataset':'original_external80','path':r['path'],'group':r['group'],**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in old_external)
        hashes=np.array([int(r['dhash64'],16)for r in references],np.uint64)
        records=[];pairs=[]
        with zipfile.ZipFile(source)as z:
            member_map={}
            for name in z.namelist():
                if name.lower().endswith('.jpg'):
                    suffix='/'.join(PurePosixPath(name).parts[-2:])
                    if suffix in member_map:raise ValueError('Ambiguous JPEG suffix')
                    member_map[suffix]=name
            for clip in manifest['panel']:
                for slot,im in enumerate(clip['frames']):
                    member=member_map[im['file_name']];raw=z.read(member);f=fingerprint(raw)
                    assert (f['width'],f['height'])==(im['width'],im['height'])
                    path=ROOT/'images'/im['file_name'];path.parent.mkdir(parents=True,exist_ok=True)
                    if path.exists():assert file_hash(path)==f['file_sha256']
                    else:
                        temporary=path.with_suffix('.partial');temporary.write_bytes(raw);temporary.replace(path)
                    record={'group':clip['group'],'image_id':im['id'],'time':slot,'file_name':im['file_name'],'path':str(path),'archive_member':member,**f}
                    records.append(record)
                    distance=np.bitwise_count(hashes^np.uint64(int(f['dhash64'],16)))
                    for index in np.flatnonzero(distance<=6):
                        ref=references[int(index)];assert file_hash(ref['path'])==ref['file_sha256']
                        pairs.append({'pair_id':len(pairs),'group':record['group'],'image_id':record['image_id'],'external_path':record['path'],
                            'external_sha256':record['file_sha256'],'reference_dataset':ref['dataset'],'reference_path':ref['path'],'reference_sha256':ref['file_sha256'],
                            'reference_group':ref.get('group'),'distance':int(distance[index]),'exact_file':record['file_sha256']==ref['file_sha256'],'exact_pixels':record['pixel_sha256']==ref['pixel_sha256']})
                print('panel RGB',clip['group'],len(records),flush=True)
        for i,left in enumerate(records):
            for right in records[i+1:]:
                if left['group']==right['group']:continue
                distance=(int(left['dhash64'],16)^int(right['dhash64'],16)).bit_count()
                if distance<=6:
                    pairs.append({'pair_id':len(pairs),'group':left['group'],'image_id':left['image_id'],'external_path':left['path'],'external_sha256':left['file_sha256'],
                        'reference_dataset':'new_panel','reference_path':right['path'],'reference_sha256':right['file_sha256'],'reference_group':right['group'],
                        'distance':distance,'exact_file':left['file_sha256']==right['file_sha256'],'exact_pixels':left['pixel_sha256']==right['pixel_sha256']})
        atomic_json(ROOT/'image_records.json',records);atomic_json(ROOT/'pairs.json',pairs)
        by_group=defaultdict(list)
        for pair in pairs:by_group[pair['group']].append(pair)
        sheets=[]
        for group,group_pairs in sorted(by_group.items()):
            panels={p['external_path']:f'NEW {group} image {p["image_id"]}'for p in group_pairs}
            for p in group_pairs:panels[p['reference_path']]=p['reference_dataset']+' '+Path(p['reference_path']).parent.name+'/'+Path(p['reference_path']).name
            entries=list(panels.items())
            for page,start in enumerate(range(0,len(entries),28)):
                selected=entries[start:start+28];rows=math.ceil(len(selected)/4)
                fig,axes=plt.subplots(rows,4,figsize=(16,rows*2.7),squeeze=False)
                for axis in axes.ravel():axis.axis('off')
                for axis,(path,label)in zip(axes.ravel(),selected):
                    with Image.open(path)as im:axis.imshow(im.convert('RGB'))
                    axis.set_title(label,fontsize=8)
                fig.suptitle(f'RGB-only panel group {group} page {page}',fontsize=12);fig.tight_layout(rect=(0,0,1,.97))
                destination=ROOT/f'group_{group}_page{page}.png';fig.savefig(destination,dpi=120);plt.close(fig)
                sheets.append({'group':group,'page':page,'path':str(destination),'sha256':file_hash(destination),
                    'panels':[{'path':p,'sha256':file_hash(p)}for p,_ in selected]})
        for path,h in protocol['input_sha256'].items():assert file_hash(path)==h
        atomic_json(ROOT/'materials.json',{'manifest_sha256':file_hash(ROOT/'manifest.json'),'images':len(records),'image_records_sha256':file_hash(ROOT/'image_records.json'),
            'pair_count':len(pairs),'pairs_sha256':file_hash(ROOT/'pairs.json'),'pair_sources':dict(Counter(p['reference_dataset']for p in pairs)),
            'exact_pairs':sum(p['exact_file']or p['exact_pixels']for p in pairs),'sheets':sheets,'no_model_evaluation':True})
        print('DONE class panel preparation',len(manifest['panel']),'groups',len(records),'images',len(pairs),'pairs',len(sheets),'sheets',flush=True)


if __name__=='__main__':main()
