"""Fixed blind candidate RGB acquisition and exact/near duplicate flags, no model."""
from collections import Counter,defaultdict
import fcntl
import hashlib
import http.cookiejar
import io
import json
from pathlib import Path,PurePosixPath
import re
import sqlite3
import ssl
import shutil
import time
import urllib.error
import urllib.request
import urllib.parse
import zlib
import numpy as np
from PIL import Image
from semantic import external_archive_probe as archive
from semantic.external_archive_resume import read_prefix
from semantic.ovis_frame_index import fingerprint
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_rgb_continuation')
ORIGIN=Path('semantic_artifacts/external_rgb_screen')
PREVIOUS=Path('semantic_artifacts/external_archive_resume')
INDEX=Path('semantic_artifacts/ovis_frame_index')
ADAPTER=Path('semantic_artifacts/external_coco_adapter')
LIMIT=256*2**20
CHUNK=64*2**10
SEGMENTS=4
SEED=20261045


def fixed_manifest(clips):
    order=np.random.default_rng(SEED).permutation(len(clips))[:80]
    return {'seed':SEED,'candidate_confirmation':[clips[int(i)]for i in order[:40]],'provisional_reserve':[clips[int(i)]for i in order[40:]]}


class ChunkedRanges:
    def __init__(self,opener,url,total,version):
        self.opener=opener;self.url=url;self.total=total;self.version=tuple(version)
        state=ROOT/'transfer_progress.json'
        old=json.loads(state.read_text())if state.exists()else{}
        self.received=old.get('received_bytes',0);self.records=old.get('records',[]);self.attempts=old.get('attempts',[])
        self.failures=[a for a in self.attempts if a['outcome']!='complete_response'];self.forms=old.get('forms',[])
    def account(self,n):
        self.received+=n
        if self.received>LIMIT:raise ValueError('byte_budget_exhausted')
    def piece(self,start,end):
        offset=start;parts=[]
        for segment in range(1,SEGMENTS+1):
            if end-offset+1>LIMIT-self.received:raise ValueError('insufficient_byte_budget')
            before=self.received
            record={'logical_start':start,'logical_end':end,'start':offset,'end':end,'segment':segment}
            try:
                request=urllib.request.Request(self.url,headers={'Range':f'bytes={offset}-{end}',
                    'Accept-Encoding':'identity','User-Agent':'D4RT academic finite suffix range/1.0'})
                with self.opener.open(request,timeout=20)as response:
                    record.update(status=response.status,content_length=response.headers.get('Content-Length'),
                        content_encoding=response.headers.get('Content-Encoding'),transfer_encoding=response.headers.get('Transfer-Encoding'))
                    if response.status!=206:raise ValueError('non206_status')
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not match or tuple(map(int,match.groups()))!=(offset,end,self.total):raise ValueError('content_range_mismatch')
                    if (response.headers.get('ETag'),response.headers.get('Last-Modified'))!=self.version:raise ValueError('archive_version_mismatch')
                    if response.headers.get('Content-Encoding','identity').lower()!='identity':raise ValueError('unexpected_content_encoding')
                    length=response.headers.get('Content-Length')
                    if length is not None and int(length)!=end-offset+1:raise ValueError('declared_length_mismatch')
                    data,complete=read_prefix(response,end-offset+1,self.account)
                if not data:raise ValueError('zero_progress_prefix')
                path=ROOT/'ranges'/f'{len(self.records):03d}.bin';path.parent.mkdir(exist_ok=True);path.write_bytes(data)
                self.records.append({'start':offset,'end':offset+len(data)-1,'bytes':len(data),'path':str(path),'sha256':file_hash(path)})
                record.update(received_bytes=self.received-before,observed_end=offset+len(data)-1,outcome='complete_response'if complete else'incomplete_prefix')
                self.attempts.append(record)
                if not complete:self.failures.append(dict(record))
                atomic_json(ROOT/'transfer_progress.json',{'attempts':self.attempts,'received_bytes':self.received,'records':self.records,'forms':self.forms})
                parts.append(data);offset+=len(data)
                if offset==end+1:return b''.join(parts)
            except Exception as exc:
                record.update(received_bytes=self.received-before,outcome='fatal',type=type(exc).__name__,reason=str(exc)if isinstance(exc,ValueError)else'transport_error')
                self.attempts.append(record);self.failures.append(dict(record))
                atomic_json(ROOT/'transfer_progress.json',{'attempts':self.attempts,'received_bytes':self.received,'records':self.records,'forms':self.forms});raise
        raise ValueError('suffix_segment_limit')
    def read(self,start,end):
        if start<0 or end<start or end>=self.total:raise ValueError('invalid_zip_range')
        return b''.join(self.piece(a,min(end,a+CHUNK-1))for a in range(start,end+1,CHUNK))


class HeaderRetryOpener:
    def __init__(self,opener):
        self.opener=opener;self.path=ROOT/'header_retries.json'
        self.records=json.loads(self.path.read_text())if self.path.exists()else[]
    def open(self,request,timeout):
        for attempt in range(1,4):
            try:
                response=self.opener.open(request,timeout=60)
                if attempt>1:
                    self.records.append({'range':request.get_header('Range'),'attempt':attempt,'outcome':'headers_received_after_retry','status':response.status})
                    atomic_json(self.path,self.records)
                return response
            except Exception as exc:
                retryable=isinstance(exc,TimeoutError)or(isinstance(exc,urllib.error.URLError)and isinstance(exc.reason,TimeoutError))
                if not retryable:raise
                self.records.append({'range':request.get_header('Range'),'attempt':attempt,'outcome':'header_timeout','timeout_seconds':60,'received_body_bytes':0})
                atomic_json(self.path,self.records)
                if attempt==3:raise
                time.sleep(attempt)
        raise AssertionError('unreachable')


def seed_previous():
    prior=json.loads((ORIGIN/'summary.json').read_text());verify=json.loads((ORIGIN/'verification.json').read_text())
    assert file_hash(ORIGIN/'summary.json')==verify['summary_sha256']and prior['completed_images']==28 and prior['received_bytes']==2639191
    marker=ROOT/'seed_manifest.json'
    if marker.exists():
        seed=json.loads(marker.read_text())
        for path,h in seed['source_sha256'].items():assert file_hash(path)==h
        return
    sources=[ORIGIN/'summary.json',ORIGIN/'verification.json',ORIGIN/'transfer_progress.json',*sorted((ORIGIN/'image_records').glob('*.json'))]
    destination=ROOT/'image_records';destination.mkdir(exist_ok=True)
    for record in (ORIGIN/'image_records').glob('*.json'):shutil.copyfile(record,destination/record.name)
    shutil.copyfile(ORIGIN/'transfer_progress.json',ROOT/'transfer_progress.json')
    atomic_json(marker,{'completed_images':28,'carried_received_bytes':2639191,'source_sha256':{str(p):file_hash(p)for p in sources}})


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'summary.json').exists():raise RuntimeError('Completed RGB continuation exists; inspect instead of rerunning')
        seed_previous()
        old=json.loads((PREVIOUS/'summary.json').read_text());verify=json.loads((PREVIOUS/'verification.json').read_text())
        assert file_hash(PREVIOUS/'summary.json')==verify['summary_sha256']and old['source_error']is None
        idxs=json.loads((INDEX/'summary.json').read_text());idxv=json.loads((INDEX/'verification.json').read_text())
        assert file_hash(INDEX/'summary.json')==idxv['summary_sha256']and file_hash(INDEX/'index.sqlite')==idxs['index_sha256']
        paths=[Path('semantic/external_archive_resume.py'),Path('semantic/ovis_frame_index.py'),PREVIOUS/'summary.json',PREVIOUS/'verification.json',PREVIOUS/'directory.json',INDEX/'summary.json',INDEX/'verification.json',INDEX/'index.sqlite',ADAPTER/'groups.json',ORIGIN/'summary.json',ORIGIN/'verification.json',ORIGIN/'manifest.json',ORIGIN/'transfer_progress.json',ROOT/'seed_manifest.json',Path('semantic/external_rgb_continuation.py'),Path('semantic/OVIS_EXTERNAL_RGB_CONTINUATION_PROTOCOL.md')]
        for arm,choice in old['protocol']['frozen_selection'].items():
            for seed in range(3):
                path=checkpoint(arm,seed);assert file_hash(path)==choice['checkpoint_sha256'][str(seed)];paths.append(path)
        protocol={'round':46,'seed':SEED,'timeout_seconds':60,'max_header_timeout_retries':2,'byte_cap':LIMIT,'max_range_bytes':CHUNK,'max_suffix_segments':SEGMENTS,'dhash_threshold':6,
            'frozen_selection':old['protocol']['frozen_selection'],'archive_total':old['protocol']['fixed_archive_total'],'archive_version':old['protocol']['fixed_archive_version'],
            'no_model_evaluation':True,'input_sha256':{str(p):file_hash(p)for p in paths}}
        pp=ROOT/'protocol.json'
        if pp.exists():assert json.loads(pp.read_text())==protocol
        else:atomic_json(pp,protocol)
        manifest=fixed_manifest(json.loads((ADAPTER/'groups.json').read_text())['groups']);mp=ROOT/'manifest.json'
        if mp.exists():assert json.loads(mp.read_text())==manifest
        else:atomic_json(mp,manifest)
        directory=json.loads((PREVIOUS/'directory.json').read_text());members={}
        for name in directory:
            if name.lower().endswith('.jpg'):
                suffix='/'.join(PurePosixPath(name).parts[-2:])
                if suffix in members:raise ValueError('ambiguous member suffix')
                members[suffix]=name
        connection=sqlite3.connect(f'file:{INDEX / "index.sqlite"}?mode=ro',uri=True)
        indexed=[json.loads(row[0])for row in connection.execute('SELECT record FROM frames ORDER BY path')];connection.close()
        assert len(indexed)==42149 and all(r['status']=='ok'for r in indexed)
        exact={field:defaultdict(list)for field in ('file_sha256','pixel_sha256')}
        for row in indexed:
            for field in exact:exact[field][row[field]].append(row['path'])
        hashes=np.array([int(row['dhash64'],16)for row in indexed],dtype=np.uint64)
        opener=urllib.request.build_opener(archive.previous.previous.HTTPSOnlyRedirects(),urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        opener=HeaderRetryOpener(opener)
        ranges=None;error=None;records=[];form_bytes=0
        try:
            state=ROOT/'transfer_progress.json';forms=json.loads(state.read_text()).get('forms',[])if state.exists()else[]
            form_path=ROOT/f'confirmation_{len(forms):03d}.html'
            ranges=ChunkedRanges(opener,'',protocol['archive_total'],protocol['archive_version'])
            try:
                with opener.open(urllib.request.Request(archive.SOURCE,headers={'User-Agent':'D4RT blind RGB audit/1.0'}),timeout=20)as response,form_path.open('wb')as stream:
                    while True:
                        remaining=min(LIMIT-ranges.received,4*2**20-form_bytes)
                        if remaining<=0:raise ValueError('form_budget_exhausted')
                        chunk=response.read(min(16384,remaining))
                        if not chunk:break
                        ranges.account(len(chunk));form_bytes+=len(chunk);stream.write(chunk)
            finally:
                if form_path.exists():ranges.forms.append({'path':str(form_path),'bytes':form_bytes,'sha256':file_hash(form_path)})
                atomic_json(ROOT/'transfer_progress.json',{'attempts':ranges.attempts,'received_bytes':ranges.received,'records':ranges.records,'forms':ranges.forms})
            file_id=urllib.parse.parse_qs(urllib.parse.urlsplit(archive.SOURCE).query)['id'][0]
            form=archive.previous.validated_form(form_path.read_text(),file_id);url=form['action']+'?'+urllib.parse.urlencode(form['fields'])
            ranges.url=url
            reader=archive.RangeZIP(ranges.read,ranges.total);reader.entries=directory
            for role in ('candidate_confirmation','provisional_reserve'):
                for clip in manifest[role]:
                    for time,image in enumerate(clip['frames']):
                        rp=ROOT/'image_records'/f"{image['id']:06d}.json";rp.parent.mkdir(exist_ok=True)
                        if rp.exists():
                            record=json.loads(rp.read_text());assert file_hash(record['path'])==record['file_sha256'];records.append(record);continue
                        member=members[image['file_name']];raw=reader.member(member);features=fingerprint(raw)
                        if (features['width'],features['height'])!=(image['width'],image['height']):raise ValueError('image_dimensions_mismatch')
                        destination=ROOT/'images'/image['file_name'];destination.parent.mkdir(parents=True,exist_ok=True)
                        partial=destination.with_suffix('.partial');partial.write_bytes(raw);partial.replace(destination)
                        distances=np.bitwise_count(hashes^np.uint64(int(features['dhash64'],16)))
                        order=np.argsort(distances,kind='stable');near=order[distances[order]<=protocol['dhash_threshold']]
                        record={'role':role,'group':clip['group'],'time':time,'image_id':image['id'],'file_name':image['file_name'],
                            'archive_member':member,'archive_crc32':directory[member]['crc32'],'path':str(destination),**features,
                            'exact_file_matches':exact['file_sha256'].get(features['file_sha256'],[]),'exact_pixel_matches':exact['pixel_sha256'].get(features['pixel_sha256'],[]),
                            'near_match_count':len(near),'near_matches_first20':[{'path':indexed[int(i)]['path'],'distance':int(distances[i])}for i in near[:20]]}
                        atomic_json(rp,record);records.append(record)
                        atomic_json(ROOT/'progress.json',{'completed_images':len(records),'total_images':640,'received_bytes':ranges.received})
                    print('RGB acquired',role,clip['group'],len(records),'/ 640',flush=True)
        except Exception as exc:
            error={'type':type(exc).__name__,'reason':str(exc)if isinstance(exc,ValueError)else'transport_or_decode_error','completed_images':len(records)}
        for path,h in protocol['input_sha256'].items():assert file_hash(path)==h
        exact_groups=sorted({r['group']for r in records if r['exact_file_matches']or r['exact_pixel_matches']})
        near_groups=sorted({r['group']for r in records if r['near_match_count']})
        atomic_json(ROOT/'summary.json',{'protocol':protocol,'manifest_sha256':file_hash(mp),'source_error':error,'completed_images':len(records),'planned_images':640,
            'received_bytes':ranges.received if ranges else form_bytes,'exact_match_groups':exact_groups,'near_flagged_groups':near_groups,
            'near_flagged_images':sum(bool(r['near_match_count'])for r in records),'role_counts':dict(Counter(r['role']for r in records)),
            'header_retries_sha256':file_hash(ROOT/'header_retries.json')if(ROOT/'header_retries.json').exists()else None,
            'transfer_progress_sha256':file_hash(ROOT/'transfer_progress.json')if(ROOT/'transfer_progress.json').exists()else None,
            'no_model_evaluation':True,'formal_confirmation_frozen':False,'near_matches_reviewed':False,'deployment_unchanged':'native_gt'})
        print('DONE blind RGB acquisition',len(records),'error',error,'exact groups',len(exact_groups),'near groups',len(near_groups),flush=True)


if __name__=='__main__':main()
