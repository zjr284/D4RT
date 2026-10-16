"""Few-request bounded archive transfer; exact frozen directory and all CRCs."""
import fcntl
import http.client
import http.cookiejar
import json
import os
from pathlib import Path,PurePosixPath
import ssl
import time
import urllib.request
import urllib.parse
import urllib.error
import zipfile
from semantic import external_archive_probe as archive
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_archive_download')
PRIOR=Path('semantic_artifacts/external_http_diagnosis')
REFERENCE=Path('semantic_artifacts/external_archive_resume')
LIMIT=2*2**30
TOTAL=1805445389
MODIFIED='Mon, 07 Aug 2023 16:46:58 GMT'


def audit_zip(path,expected,progress=None):
    with zipfile.ZipFile(path)as z:
        infos=z.infolist()
        if len(infos)!=len(expected)or sum(i.file_size for i in infos)>8*2**30:raise ValueError('directory_count_or_uncompressed_limit')
        for i,info in enumerate(infos):
            name=str(PurePosixPath(info.filename));e=expected[name]
            if (info.CRC,info.compress_size,info.file_size,info.header_offset,info.compress_type)!=(e['crc32'],e['compressed_bytes'],e['uncompressed_bytes'],e['local_offset'],e['method']):raise ValueError('frozen_directory_mismatch')
            with z.open(info)as stream:
                while stream.read(2**20):pass
            if progress and (i+1)%512==0:progress(i+1,len(infos))
    return len(infos)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'summary.json').exists():raise RuntimeError('Completed archive control exists')
        previous=json.loads((PRIOR/'summary.json').read_text());v=json.loads((PRIOR/'verification.json').read_text())
        assert file_hash(PRIOR/'summary.json')==v['summary_sha256']and previous['response']['status']==206
        paths=[PRIOR/'summary.json',PRIOR/'verification.json',REFERENCE/'summary.json',REFERENCE/'verification.json',REFERENCE/'directory.json',Path('semantic_artifacts/external_rgb_screen/manifest.json'),Path('semantic/external_archive_download.py'),Path('semantic/OVIS_EXTERNAL_ARCHIVE_DOWNLOAD_PROTOCOL.md')]
        p={'round':48,'new_body_byte_cap':LIMIT,'archive_bytes':TOTAL,'modified':MODIFIED,'max_body_requests':3,'socket_timeout_seconds':60,
            'prior_RGB_and_diagnosis_bytes_preserved':4481879+2485,'input_sha256':{str(p):file_hash(p)for p in paths}}
        pp=ROOT/'protocol.json'
        if pp.exists():assert json.loads(pp.read_text())==p
        else:atomic_json(pp,p)
        state_path=ROOT/'download_state.json';state=json.loads(state_path.read_text())if state_path.exists()else{'attempts':[],'forms':[],'received_bytes':0}
        partial=ROOT/'archive.zip.partial';completed=ROOT/'archive.zip';error=None;crc_count=0;form_bytes=0
        offset=completed.stat().st_size if completed.exists()else partial.stat().st_size if partial.exists()else 0
        accounted=sum(a.get('received_bytes',0)for a in state['attempts'])
        if offset>accounted:
            assert state['attempts']
            state['attempts'][-1]['received_bytes']+=offset-accounted
            state['attempts'][-1].update(end_offset=offset,outcome='interrupted_prefix_reconciled')
        state['received_bytes']=max(state['received_bytes'],offset+sum(f['bytes']for f in state['forms']))
        opener=urllib.request.build_opener(archive.previous.previous.HTTPSOnlyRedirects(),urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        try:
            if offset<TOTAL:
                form_path=ROOT/f'confirmation_{len(state["forms"]):03d}.html'
                try:
                    with opener.open(urllib.request.Request(archive.SOURCE,headers={'User-Agent':'D4RT few-request archive/1.0'}),timeout=60)as response,form_path.open('wb')as stream:
                        while True:
                            remaining=min(LIMIT-state['received_bytes'],2**20-form_bytes)
                            if remaining<=0:raise ValueError('form_budget_exhausted')
                            data=response.read(min(16384,remaining))
                            if not data:break
                            state['received_bytes']+=len(data);form_bytes+=len(data);stream.write(data)
                finally:
                    if form_path.exists():state['forms'].append({'path':str(form_path),'bytes':form_bytes,'sha256':file_hash(form_path)})
                    atomic_json(state_path,state)
                file_id=urllib.parse.parse_qs(urllib.parse.urlsplit(archive.SOURCE).query)['id'][0]
                form=archive.previous.validated_form(form_path.read_text(),file_id);url=form['action']+'?'+urllib.parse.urlencode(form['fields'])
                while offset<TOTAL and len(state['attempts'])<3:
                    start=offset;record={'attempt':len(state['attempts'])+1,'start':start,'received_bytes':0};state['attempts'].append(record);atomic_json(state_path,state)
                    headers={'Accept-Encoding':'identity','User-Agent':'D4RT few-request archive/1.0'}
                    if start:headers['Range']=f'bytes={start}-{TOTAL-1}'
                    try:
                        with opener.open(urllib.request.Request(url,headers=headers),timeout=60)as response:
                            record.update(status=response.status,content_length=response.headers.get('Content-Length'),content_range=response.headers.get('Content-Range'),modified=response.headers.get('Last-Modified'),content_encoding=response.headers.get('Content-Encoding'))
                            if response.status!=(206 if start else 200):raise ValueError('unexpected_full_or_suffix_status')
                            if start and record['content_range']!=f'bytes {start}-{TOTAL-1}/{TOTAL}':raise ValueError('suffix_range_mismatch')
                            if record['modified']!=MODIFIED or record['content_encoding']not in (None,'identity'):raise ValueError('version_or_encoding_mismatch')
                            if record['content_length']is not None and int(record['content_length'])!=TOTAL-start:raise ValueError('declared_length_mismatch')
                            with partial.open('ab')as stream:
                                while offset<TOTAL:
                                    remaining=min(65536,TOTAL-offset,LIMIT-state['received_bytes'])
                                    if remaining<=0:raise ValueError('body_budget_exhausted')
                                    try:data=response.read(remaining)
                                    except http.client.IncompleteRead as exc:data=exc.partial
                                    if not data:record['outcome']='early_EOF';break
                                    if len(data)>remaining:raise ValueError('overlong_body')
                                    stream.write(data);offset+=len(data);record['received_bytes']+=len(data);state['received_bytes']+=len(data)
                                    if offset//(8*2**20)!=(offset-len(data))//(8*2**20):
                                        stream.flush();atomic_json(state_path,state);print('archive bytes',offset,'/',TOTAL,flush=True)
                                if offset==TOTAL:
                                    extra=response.read(1)
                                    if extra:state['received_bytes']+=1;raise ValueError('extra_archive_byte')
                                    record['outcome']='complete'
                                stream.flush();os.fsync(stream.fileno())
                    except urllib.error.HTTPError as exc:
                        record.update(status=exc.code,retry_after=exc.headers.get('Retry-After'),outcome='HTTPError')
                        if exc.code not in (500,502,503,504):raise
                    except (TimeoutError,urllib.error.URLError)as exc:
                        if not isinstance(exc,TimeoutError)and not isinstance(exc.reason,TimeoutError):raise
                        record['outcome']='timeout_preserved_prefix'
                    finally:record['end_offset']=offset;atomic_json(state_path,state)
                    if offset<TOTAL:
                        delay=record.get('retry_after');wait=int(delay)if delay and delay.isdigit()else len(state['attempts'])
                        if wait>60:raise ValueError('retry_after_exceeds_bounded_wait')
                        if len(state['attempts'])<3:time.sleep(wait)
                if offset!=TOTAL:raise ValueError('bounded_transfer_incomplete')
            if partial.exists():partial.replace(completed)
            if completed.stat().st_size!=TOTAL:raise ValueError('archive_size_mismatch')
            expected=json.loads((REFERENCE/'directory.json').read_text())
            crc_count=audit_zip(completed,expected,lambda n,t:print('CRC audit',n,'/',t,flush=True))
        except Exception as exc:error={'type':type(exc).__name__,'status':getattr(exc,'code',None),'reason':str(exc)if isinstance(exc,ValueError)else'transport_or_CRC_error'}
        for path,h in p['input_sha256'].items():assert file_hash(path)==h
        saved=completed if completed.exists()else partial
        atomic_json(ROOT/'summary.json',{'protocol':p,'source_error':error,'saved_bytes':saved.stat().st_size if saved.exists()else 0,
            'archive_path':str(saved),'archive_sha256':file_hash(saved)if saved.exists()else None,'new_received_bytes':state['received_bytes'],
            'state_sha256':file_hash(state_path),'body_requests':len(state['attempts']),'members_CRC_verified':crc_count,
            'no_model_evaluation':True,'cohort_unchanged':True,'deployment_unchanged':'native_gt'})
        print('DONE full archive control','error',error,'saved',saved.stat().st_size if saved.exists()else 0,'CRC',crc_count,flush=True)


if __name__=='__main__':main()
