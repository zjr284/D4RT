"""Full local OVIS frame inventory for later external duplicate screening."""
from collections import Counter,defaultdict
import fcntl
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import sqlite3
import struct
from PIL import Image
import numpy as np
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/ovis_frame_index')
FRAMES=Path('semantic_artifacts/ovis/train')
ANN=Path('semantic_artifacts/ovis/annotations_train.json')
MANIFEST=Path('semantic_artifacts/dino_native_rgb_control/manifest.json')


def safe_path(name):
    p=PurePosixPath(name)
    if p.is_absolute()or'..'in p.parts or'\\'in name or len(p.parts)!=2:raise ValueError('unsafe frame path')
    path=FRAMES/str(p)
    if not path.resolve().is_relative_to(FRAMES.resolve()):raise ValueError('frame symlink escapes root')
    return path


def stat_record(path):
    if not path.exists():return None
    s=path.stat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns}


def fingerprint(raw):
    with Image.open(io.BytesIO(raw))as original:
        image=original.convert('RGB');image.load();width,height=image.size
        pixels=struct.pack('<II',width,height)+image.tobytes()
        gray=image.convert('L').resize((32,32),Image.Resampling.BILINEAR).resize((9,8),Image.Resampling.BILINEAR)
        a=np.asarray(gray);bits=np.packbits((a[:,1:]>a[:,:-1]).ravel(),bitorder='big').tobytes().hex()
    return {'file_sha256':hashlib.sha256(raw).hexdigest(),'pixel_sha256':hashlib.sha256(pixels).hexdigest(),
        'width':width,'height':height,'dhash64':bits}


def inventory():
    payload=json.loads(ANN.read_bytes());items=[];seen=set()
    for video in payload['videos']:
        for name in video['file_names']:
            if name in seen:raise ValueError('repeated frame reference')
            seen.add(name);path=safe_path(name)
            items.append({'path':name,'video_id':video['id'],'width':video['width'],'height':video['height'],'stat':stat_record(path)})
    return sorted(items,key=lambda r:r['path'])


def duplicate_stats(records,manifest):
    splits={c['video_id']:'train'for c in manifest['train']};splits.update({c['video_id']:'val'for c in manifest['val']})
    result={};details={}
    for field in ('file_sha256','pixel_sha256'):
        grouped=defaultdict(list)
        for r in records:
            if r['status']=='ok':grouped[r[field]].append(r)
        duplicates=[g for g in grouped.values()if len(g)>1]
        cross=[g for g in duplicates if len({r['video_id']for r in g})>1]
        train_val=[g for g in cross if {'train','val'}<={splits.get(r['video_id'],'other')for r in g}]
        result[field]={'duplicate_content_groups':len(duplicates),'cross_video_groups':len(cross),
            'cross_video_involved_videos':len({r['video_id']for g in cross for r in g}),'train_val_groups':len(train_val)}
        details[field]={'cross_video':[[{'path':r['path'],'video_id':r['video_id']}for r in g]for g in cross]}
    return result,details


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'summary.json').exists():raise RuntimeError('Completed frame index exists')
        prior=Path('semantic_artifacts/external_coco_adapter');s=json.loads((prior/'summary.json').read_text());v=json.loads((prior/'verification.json').read_text())
        assert file_hash(prior/'summary.json')==v['summary_sha256']
        paths=[ANN,MANIFEST,prior/'summary.json',prior/'verification.json',Path('semantic/ovis_frame_index.py'),Path('semantic/OVIS_OVIS_FRAME_INDEX_PROTOCOL.md')]
        protocol={'round':42,'analysis':'local full-frame index; no external model score','input_sha256':{str(p):file_hash(p)for p in paths}}
        pp=ROOT/'protocol.json'
        if pp.exists():assert json.loads(pp.read_text())==protocol
        else:atomic_json(pp,protocol)
        ip=ROOT/'inventory.json';items=inventory()
        if ip.exists():assert json.loads(ip.read_text())==items
        else:atomic_json(ip,items)
        connection=sqlite3.connect(ROOT/'index.sqlite');connection.execute('CREATE TABLE IF NOT EXISTS frames (path TEXT PRIMARY KEY, record TEXT NOT NULL)')
        completed={row[0]for row in connection.execute('SELECT path FROM frames')}
        assert completed<={i['path']for i in items}
        for item in items:
            if item['path']in completed:continue
            record=dict(item);path=safe_path(item['path'])
            if item['stat']is None:record['status']='missing'
            else:
                assert stat_record(path)==item['stat']
                try:
                    data=fingerprint(path.read_bytes())
                    if (data['width'],data['height'])!=(item['width'],item['height']):raise ValueError('metadata_dimension_mismatch')
                    record.update(data,status='ok')
                except Exception as exc:record.update(status='decode_error',error_type=type(exc).__name__)
                assert stat_record(path)==item['stat']
            connection.execute('INSERT INTO frames VALUES (?,?)',(item['path'],json.dumps(record,sort_keys=True)))
            completed.add(item['path'])
            if len(completed)%128==0:
                connection.commit();atomic_json(ROOT/'progress.json',{'completed':len(completed),'total':len(items)})
                print('index progress',len(completed),'/',len(items),flush=True)
        connection.commit();records=[json.loads(row[0])for row in connection.execute('SELECT record FROM frames ORDER BY path')];connection.close()
        manifest=json.loads(MANIFEST.read_text());stats,details=duplicate_stats(records,manifest);atomic_json(ROOT/'duplicate_details.json',details)
        for path,h in protocol['input_sha256'].items():assert file_hash(path)==h
        atomic_json(ROOT/'summary.json',{'protocol':protocol,'inventory_sha256':file_hash(ip),'index_sha256':file_hash(ROOT/'index.sqlite'),
            'reference_count':len(items),'video_count':len({r['video_id']for r in items}),'status_counts':dict(Counter(r['status']for r in records)),
            'source_bytes':sum(r['stat']['size']for r in records if r['stat']is not None),'duplicates':stats,
            'duplicate_details_sha256':file_hash(ROOT/'duplicate_details.json'),'no_model_evaluation':True,
            'external_duplicate_screening_complete':False,'deployment_unchanged':'native_gt'})
        print('DONE OVIS frame index',len(records),stats,flush=True)


if __name__=='__main__':main()
