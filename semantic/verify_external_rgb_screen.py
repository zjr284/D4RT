"""Independently replay all acquired RGB with zipfile and recalculate flags."""
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import sqlite3
import struct
import zipfile
import numpy as np
from PIL import Image
from semantic.external_rgb_screen import ROOT,PREVIOUS,INDEX,ADAPTER,CHUNK,SEGMENTS
from semantic.verify_external_archive_bounded import SparseArchive
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    manifest=json.loads((ROOT/'manifest.json').read_text());assert file_hash(ROOT/'manifest.json')==s['manifest_sha256']
    clips=json.loads((ADAPTER/'groups.json').read_text())['groups'];order=np.random.default_rng(p['seed']).permutation(len(clips))[:80]
    for role,indices in [('candidate_confirmation',order[:40]),('provisional_reserve',order[40:])]:assert manifest[role]==[clips[int(i)]for i in indices]
    expected={im['id']:(role,clip,slot,im)for role in ('candidate_confirmation','provisional_reserve')for clip in manifest[role]for slot,im in enumerate(clip['frames'])}
    assert len(expected)==640 and len({c['group']for role in ('candidate_confirmation','provisional_reserve')for c in manifest[role]})==80
    transfer=json.loads((ROOT/'transfer_progress.json').read_text());assert file_hash(ROOT/'transfer_progress.json')==s['transfer_progress_sha256']
    forms=transfer['forms'];attempts=transfer['attempts'];ranges=transfer['records']
    for form in forms:assert Path(form['path']).stat().st_size==form['bytes']and file_hash(form['path'])==form['sha256']
    saved=[a for a in attempts if a['outcome']in ('complete_response','incomplete_prefix')];assert len(saved)==len(ranges)
    for i,a in enumerate(attempts):
        assert 1<=a['segment']<=SEGMENTS and 0<a['logical_end']-a['logical_start']+1<=CHUNK
        if a['segment']>1:
            prev=attempts[i-1];assert prev['outcome']=='incomplete_prefix'and prev['observed_end']+1==a['start']and prev['segment']+1==a['segment']
            assert (prev['logical_start'],prev['logical_end'])==(a['logical_start'],a['logical_end'])
        else:assert a['start']==a['logical_start']
        if a['outcome']!='fatal':assert (a['received_bytes']==a['end']-a['start']+1)==(a['outcome']=='complete_response')
    for a,r in zip(saved,ranges):
        assert a['status']==206 and (a['start'],a['observed_end'])==(r['start'],r['end'])
        assert r['bytes']==a['received_bytes']==Path(r['path']).stat().st_size==r['end']-r['start']+1
        assert file_hash(r['path'])==r['sha256']
    received=sum(f['bytes']for f in forms)+sum(a['received_bytes']for a in attempts)
    assert received==s['received_bytes']==transfer['received_bytes']<=p['byte_cap']
    connection=sqlite3.connect(f'file:{INDEX / "index.sqlite"}?mode=ro',uri=True)
    indexed=[json.loads(r[0])for r in connection.execute('SELECT record FROM frames ORDER BY path')];connection.close()
    lookup={field:defaultdict(list)for field in ('file_sha256','pixel_sha256')}
    for row in indexed:
        for field in lookup:lookup[field][row[field]].append(row['path'])
    hashes=np.array([int(row['dhash64'],16)for row in indexed],dtype=np.uint64)
    old_ranges=json.loads((PREVIOUS/'summary.json').read_text())['ranges'];records=[json.loads(path.read_text())for path in sorted((ROOT/'image_records').glob('*.json'))]
    assert len(records)==s['completed_images']and len({r['image_id']for r in records})==len(records)
    with zipfile.ZipFile(SparseArchive(p['archive_total'],old_ranges+ranges))as archive:
        for i,r in enumerate(records):
            role,clip,slot,im=expected[r['image_id']]
            assert (role,clip['group'],slot,im['file_name'])==(r['role'],r['group'],r['time'],r['file_name'])
            raw=Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['file_sha256']
            assert archive.read(r['archive_member'])==raw and archive.getinfo(r['archive_member']).CRC==r['archive_crc32']
            with Image.open(io.BytesIO(raw))as source:
                rgb=source.convert('RGB');rgb.load();assert rgb.size==(im['width'],im['height'])==(r['width'],r['height'])
                assert hashlib.sha256(struct.pack('<II',*rgb.size)+rgb.tobytes()).hexdigest()==r['pixel_sha256']
                gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.BILINEAR).resize((9,8),Image.Resampling.BILINEAR))
                bits=''.join('1'if gray[y,x+1]>gray[y,x]else'0'for y in range(8)for x in range(8));assert f'{int(bits,2):016x}'==r['dhash64']
            for field,key in [('file_sha256','exact_file_matches'),('pixel_sha256','exact_pixel_matches')]:assert lookup[field].get(r[field],[])==r[key]
            distances=np.bitwise_count(hashes^np.uint64(int(r['dhash64'],16)));near=np.flatnonzero(distances<=p['dhash_threshold'])
            ordered=sorted(near.tolist(),key=lambda j:(int(distances[j]),j))
            assert len(near)==r['near_match_count']
            assert [{'path':indexed[j]['path'],'distance':int(distances[j])}for j in ordered[:20]]==r['near_matches_first20']
            if (i+1)%64==0:print('verify RGB',i+1,'/',len(records),flush=True)
    assert sorted({r['group']for r in records if r['exact_file_matches']or r['exact_pixel_matches']})==s['exact_match_groups']
    assert sorted({r['group']for r in records if r['near_match_count']})==s['near_flagged_groups']
    assert sum(bool(r['near_match_count'])for r in records)==s['near_flagged_images']and dict(Counter(r['role']for r in records))==s['role_counts']
    if s['source_error']is None:assert len(records)==640
    assert s['no_model_evaluation']and not s['formal_confirmation_frozen']and not s['near_matches_reviewed']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'replayed_RGB_images':len(records),'received_bytes':received,
        'verification':{'fixed_blind_roles_and_frames_identical':True,'all_frozen_inputs_unchanged':True,'HTTP_prefix_hashes_and_consumption_replayed':True,
        'all_acquired_RGB_independently_replayed_with_zipfile_CRC':True,'all_pixel_SHA_dhash_and_duplicate_flags_recomputed':True,'no_model_scores_or_reselection':True}})
    print('DONE independent blind RGB audit',len(records),flush=True)


if __name__=='__main__':main()
