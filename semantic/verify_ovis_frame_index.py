"""Rehash every indexed source; independently decode fixed 48 for pixel checks."""
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import struct
import numpy as np
from PIL import Image
from semantic.ovis_frame_index import ROOT,FRAMES,ANN,MANIFEST
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol']
    assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert file_hash(ROOT/'inventory.json')==s['inventory_sha256']
    assert file_hash(ROOT/'index.sqlite')==s['index_sha256']
    connection=sqlite3.connect(f'file:{ROOT / "index.sqlite"}?mode=ro',uri=True)
    assert connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    rows=[json.loads(row[0])for row in connection.execute('SELECT record FROM frames ORDER BY path')];connection.close()
    inventory=json.loads((ROOT/'inventory.json').read_text());assert len(rows)==len(inventory)==s['reference_count']
    payload=json.loads(ANN.read_bytes());expected={name:(v['id'],v['width'],v['height'])for v in payload['videos']for name in v['file_names']}
    assert len(expected)==len(rows)and len(payload['videos'])==s['video_count']
    total_bytes=0;raw_checked=0
    selected={int(round(i*(len(rows)-1)/47))for i in range(48)};pixel_checks=[]
    for index,(row,item)in enumerate(zip(rows,inventory)):
        assert all(row[k]==v for k,v in item.items())and (row['video_id'],row['width'],row['height'])==expected[row['path']]
        path=FRAMES/row['path']
        if row['status']=='missing':assert row['stat']is None and not path.exists();continue
        stat=path.stat();assert {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}==row['stat'];total_bytes+=stat.st_size
        if row['status']=='ok':assert file_hash(path)==row['file_sha256'];raw_checked+=1
        if index in selected:
            if row['status']!='ok':pixel_checks.append({'index':index,'status':row['status']});continue
            with Image.open(io.BytesIO(path.read_bytes()))as source:
                rgb=source.convert('RGB');assert rgb.size==(row['width'],row['height'])
                prefix=struct.pack('<II',*rgb.size)
                assert hashlib.sha256(prefix+np.asarray(rgb).tobytes()).hexdigest()==row['pixel_sha256']
                gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.BILINEAR).resize((9,8),Image.Resampling.BILINEAR))
                bits=''.join('1'if gray[y,x+1]>gray[y,x]else'0'for y in range(8)for x in range(8))
                assert f'{int(bits,2):016x}'==row['dhash64']
            pixel_checks.append({'index':index,'status':'ok'})
        if (index+1)%4096==0:print('verify index',index+1,'/',len(rows),flush=True)
    assert total_bytes==s['source_bytes']and dict(Counter(r['status']for r in rows))==s['status_counts']
    manifest=json.loads(MANIFEST.read_text());train={c['video_id']for c in manifest['train']};val={c['video_id']for c in manifest['val']};stats={}
    for field in ('file_sha256','pixel_sha256'):
        by_digest=defaultdict(list)
        for r in rows:
            if r['status']=='ok':by_digest[r[field]].append(r['video_id'])
        duplicates=[ids for ids in by_digest.values()if len(ids)>1]
        cross=[set(ids)for ids in duplicates if len(set(ids))>1]
        stats[field]={'duplicate_content_groups':len(duplicates),'cross_video_groups':len(cross),
            'cross_video_involved_videos':len(set().union(*cross))if cross else 0,'train_val_groups':sum(bool(ids&train)and bool(ids&val)for ids in cross)}
    assert stats==s['duplicates']and file_hash(ROOT/'duplicate_details.json')==s['duplicate_details_sha256']
    assert s['no_model_evaluation']and not s['external_duplicate_screening_complete']and s['deployment_unchanged']=='native_gt'
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'all_source_file_hashes_checked':raw_checked,
        'fixed_pixel_checks':pixel_checks,'duplicates':stats,'verification':{'frozen_inputs_unchanged':True,'complete_inventory_and_stat_replayed':True,
        'all_successful_source_bytes_rehashed':True,'fixed_pixels_and_dhash_recomputed':True,'duplicate_statistics_recomputed':True,'no_external_independence_or_performance_claim':True}})
    print('DONE full frame index independent audit',raw_checked,flush=True)


if __name__=='__main__':main()
