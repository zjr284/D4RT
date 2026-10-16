"""Independent local RGB, prior-image identity, OVIS and cross-group screening."""
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import struct
import zipfile
from PIL import Image
import numpy as np
from semantic.external_local_rgb import ROOT,PREVIOUS,COHORT,ORIGIN,INDEX
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    records=json.loads((ROOT/'image_records.json').read_text());assert file_hash(ROOT/'image_records.json')==s['image_records_sha256']
    m=json.loads(COHORT.read_text());expected=[(role,clip['group'],time,im)for role in ('candidate_confirmation','provisional_reserve')for clip in m[role]for time,im in enumerate(clip['frames'])]
    assert len(records)==len(expected)==s['images_verified']==640
    earlier={r['image_id']:r for r in (json.loads(path.read_text())for path in (ORIGIN/'image_records').glob('*.json'))}
    conn=sqlite3.connect(f'file:{INDEX / "index.sqlite"}?mode=ro',uri=True);ovis=[json.loads(row[0])for row in conn.execute('SELECT record FROM frames ORDER BY path')];conn.close()
    lookup={field:defaultdict(list)for field in ('file_sha256','pixel_sha256')}
    for row in ovis:
        for field in lookup:lookup[field][row[field]].append(row['path'])
    ovis_hashes=np.array([int(r['dhash64'],16)for r in ovis],np.uint64);prior_count=0
    archive_path=Path(json.loads((PREVIOUS/'summary.json').read_text())['archive_path'])
    with zipfile.ZipFile(archive_path)as z:
        for index,(r,(role,group,time,im))in enumerate(zip(records,expected)):
            assert (r['role'],r['group'],r['time'],r['image_id'],r['file_name'])==(role,group,time,im['id'],im['file_name'])
            raw=Path(r['path']).read_bytes();assert raw==z.read(r['archive_member'])and hashlib.sha256(raw).hexdigest()==r['file_sha256']
            with Image.open(io.BytesIO(raw))as source:
                rgb=source.convert('RGB');rgb.load();assert rgb.size==(im['width'],im['height'])==(r['width'],r['height'])
                assert hashlib.sha256(struct.pack('<II',*rgb.size)+rgb.tobytes()).hexdigest()==r['pixel_sha256']
                gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.BILINEAR).resize((9,8),Image.Resampling.BILINEAR))
                bits=''.join('1'if gray[y,x+1]>gray[y,x]else'0'for y in range(8)for x in range(8));assert f'{int(bits,2):016x}'==r['dhash64']
            if r['image_id']in earlier:
                for field in ('file_sha256','pixel_sha256','dhash64','width','height'):assert r[field]==earlier[r['image_id']][field]
                prior_count+=1
            for field,key in [('file_sha256','exact_file_matches'),('pixel_sha256','exact_pixel_matches')]:assert lookup[field].get(r[field],[])==r[key]
            distances=np.bitwise_count(ovis_hashes^np.uint64(int(r['dhash64'],16)));near=np.flatnonzero(distances<=6)
            order=sorted(near.tolist(),key=lambda i:(int(distances[i]),i))
            assert len(near)==r['near_match_count']and [{'path':ovis[i]['path'],'distance':int(distances[i])}for i in order[:20]]==r['near_matches_first20']
            if (index+1)%128==0:print('verify local RGB',index+1,'/ 640',flush=True)
    assert prior_count==s['old_images_identical']==32
    pairs=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if a['group']==b['group']:continue
            distance=(int(a['dhash64'],16)^int(b['dhash64'],16)).bit_count()
            if distance<=6:pairs.append({'left_image_id':a['image_id'],'right_image_id':b['image_id'],'left_group':a['group'],'right_group':b['group'],
                'distance':distance,'exact_file':a['file_sha256']==b['file_sha256'],'exact_pixels':a['pixel_sha256']==b['pixel_sha256']})
    assert pairs==json.loads((ROOT/'cross_group_pairs.json').read_text())and file_hash(ROOT/'cross_group_pairs.json')==s['cross_group_pairs_sha256']
    assert len(pairs)==s['cross_group_near_pairs']and sum(p['exact_file']or p['exact_pixels']for p in pairs)==s['cross_group_exact_pairs']
    assert sorted({r['group']for r in records if r['exact_file_matches']or r['exact_pixel_matches']})==s['OVIS_exact_groups']
    assert sorted({r['group']for r in records if r['near_match_count']})==s['OVIS_near_groups']and sum(bool(r['near_match_count'])for r in records)==s['OVIS_near_images']
    assert dict(Counter(r['role']for r in records))==s['role_counts']and s['no_model_evaluation']and s['no_reselection']and not s['near_flags_reviewed']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all_images_replayed':640,'prior_images_identical':32,
        'cross_group_pairs_recomputed':len(pairs),'verification':{'all_inputs_and_frozen_models_unchanged':True,'all_members_bytes_CRC_dimensions_and_pixels_recomputed':True,
        'all_OVIS_flags_recomputed':True,'cross_group_flags_independently_bit_counted':True,'all_roles_preserved_no_model_scores':True}})
    print('DONE independent local RGB audit',len(records),len(pairs),flush=True)


if __name__=='__main__':main()
