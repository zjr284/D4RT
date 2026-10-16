"""Local archive extraction and blind OVIS/external cross-group duplicate flags."""
from collections import Counter,defaultdict
import fcntl
import json
from pathlib import Path,PurePosixPath
import sqlite3
import zipfile
import numpy as np
from semantic.ovis_frame_index import fingerprint
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_local_rgb')
PREVIOUS=Path('semantic_artifacts/external_archive_download')
COHORT=Path('semantic_artifacts/external_rgb_screen/manifest.json')
ORIGIN=Path('semantic_artifacts/external_rgb_continuation')
INDEX=Path('semantic_artifacts/ovis_frame_index')


def cross_group_flags(records):
    hashes=np.array([int(r['dhash64'],16)for r in records],np.uint64);pairs=[]
    for i,a in enumerate(records):
        distances=np.bitwise_count(hashes^hashes[i])
        for j in np.flatnonzero(distances<=6):
            if j<=i or records[int(j)]['group']==a['group']:continue
            b=records[int(j)]
            pairs.append({'left_image_id':a['image_id'],'right_image_id':b['image_id'],'left_group':a['group'],'right_group':b['group'],
                'distance':int(distances[j]),'exact_file':a['file_sha256']==b['file_sha256'],'exact_pixels':a['pixel_sha256']==b['pixel_sha256']})
    return pairs


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'summary.json').exists():raise RuntimeError('Completed local RGB audit exists')
        old=json.loads((PREVIOUS/'summary.json').read_text());v=json.loads((PREVIOUS/'verification.json').read_text())
        assert file_hash(PREVIOUS/'summary.json')==v['summary_sha256']and old['source_error']is None and old['members_CRC_verified']==19978
        source=Path(old['archive_path']);assert file_hash(source)==old['archive_sha256']
        selection=json.loads((ORIGIN/'summary.json').read_text())['protocol']['frozen_selection']
        origin_records=sorted((ORIGIN/'image_records').glob('*.json'))
        paths=[PREVIOUS/'summary.json',PREVIOUS/'verification.json',source,COHORT,ORIGIN/'summary.json',ORIGIN/'verification.json',INDEX/'index.sqlite',INDEX/'summary.json',INDEX/'verification.json',Path('semantic/ovis_frame_index.py'),Path('semantic/external_local_rgb.py'),Path('semantic/OVIS_EXTERNAL_LOCAL_RGB_PROTOCOL.md'),*origin_records]
        for arm,choice in selection.items():
            for seed in range(3):
                path=checkpoint(arm,seed);assert file_hash(path)==choice['checkpoint_sha256'][str(seed)];paths.append(path)
        p={'round':49,'no_network_or_model_evaluation':True,'fixed_seed':20261045,'near_threshold':6,'selection':selection,'input_sha256':{str(path):file_hash(path)for path in paths}}
        pp=ROOT/'protocol.json'
        if pp.exists():assert json.loads(pp.read_text())==p
        else:atomic_json(pp,p)
        manifest=json.loads(COHORT.read_text());old_by_id={r['image_id']:r for r in (json.loads(path.read_text())for path in origin_records)}
        conn=sqlite3.connect(f'file:{INDEX / "index.sqlite"}?mode=ro',uri=True);ovis=[json.loads(row[0])for row in conn.execute('SELECT record FROM frames ORDER BY path')];conn.close()
        lookup={field:defaultdict(list)for field in ('file_sha256','pixel_sha256')}
        for row in ovis:
            for field in lookup:lookup[field][row[field]].append(row['path'])
        hashes=np.array([int(row['dhash64'],16)for row in ovis],np.uint64);records=[];old_checked=0
        with zipfile.ZipFile(source)as z:
            members={}
            for name in z.namelist():
                if name.lower().endswith('.jpg'):
                    suffix='/'.join(PurePosixPath(name).parts[-2:])
                    if suffix in members:raise ValueError('ambiguous JPEG suffix')
                    members[suffix]=name
            for role in ('candidate_confirmation','provisional_reserve'):
                for clip in manifest[role]:
                    for time,im in enumerate(clip['frames']):
                        member=members[im['file_name']];raw=z.read(member);f=fingerprint(raw)
                        assert (f['width'],f['height'])==(im['width'],im['height'])
                        if im['id']in old_by_id:
                            earlier=old_by_id[im['id']];assert all(f[key]==earlier[key]for key in f);old_checked+=1
                        path=ROOT/'images'/im['file_name'];path.parent.mkdir(parents=True,exist_ok=True)
                        if path.exists():assert file_hash(path)==f['file_sha256']
                        else:
                            temporary=path.with_suffix('.partial');temporary.write_bytes(raw);temporary.replace(path)
                        distance=np.bitwise_count(hashes^np.uint64(int(f['dhash64'],16)));near=np.flatnonzero(distance<=6)
                        order=sorted(near.tolist(),key=lambda j:(int(distance[j]),j))
                        records.append({'role':role,'group':clip['group'],'time':time,'image_id':im['id'],'file_name':im['file_name'],'path':str(path),'archive_member':member,**f,
                            'exact_file_matches':lookup['file_sha256'].get(f['file_sha256'],[]),'exact_pixel_matches':lookup['pixel_sha256'].get(f['pixel_sha256'],[]),
                            'near_match_count':len(near),'near_matches_first20':[{'path':ovis[j]['path'],'distance':int(distance[j])}for j in order[:20]]})
                    print('local RGB',role,clip['group'],len(records),'/ 640',flush=True)
        assert old_checked==32 and len(records)==640
        pairs=cross_group_flags(records);atomic_json(ROOT/'image_records.json',records);atomic_json(ROOT/'cross_group_pairs.json',pairs)
        for path,h in p['input_sha256'].items():assert file_hash(path)==h
        atomic_json(ROOT/'summary.json',{'protocol':p,'images_verified':len(records),'old_images_identical':old_checked,'role_counts':dict(Counter(r['role']for r in records)),
            'OVIS_exact_groups':sorted({r['group']for r in records if r['exact_file_matches']or r['exact_pixel_matches']}),
            'OVIS_near_groups':sorted({r['group']for r in records if r['near_match_count']}),'OVIS_near_images':sum(bool(r['near_match_count'])for r in records),
            'cross_group_near_pairs':len(pairs),'cross_group_exact_pairs':sum(r['exact_file']or r['exact_pixels']for r in pairs),
            'image_records_sha256':file_hash(ROOT/'image_records.json'),'cross_group_pairs_sha256':file_hash(ROOT/'cross_group_pairs.json'),
            'no_model_evaluation':True,'no_reselection':True,'near_flags_reviewed':False,'deployment_unchanged':'native_gt'})
        print('DONE local RGB audit',len(records),'cross near',len(pairs),flush=True)


if __name__=='__main__':main()
