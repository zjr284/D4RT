"""Finite suffix continuation, retaining the exact provenance of partial bodies."""
import http.cookiejar
import http.client
import io
import json
from pathlib import Path,PurePosixPath
import re
import ssl
import urllib.request
import urllib.parse
from PIL import Image
from semantic import external_archive_probe as prior
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_archive_resume')
LIMIT=16*2**20
CHUNK=64*2**10
SEGMENTS=4
class ShortResponse(Exception):pass


def read_prefix(stream,length,account):
    parts=[];received=0
    while received<length:
        try:data=stream.read(min(16384,length-received))
        except http.client.IncompleteRead as exc:
            data=exc.partial;account(len(data));parts.append(data);received+=len(data)
            if received>length:raise ValueError('overlong_incomplete_body')
            return b''.join(parts),received==length
        if not data:return b''.join(parts),False
        account(len(data));parts.append(data);received+=len(data)
        if received>length:raise ValueError('overlong_body')
    extra=stream.read(1)
    if extra:account(len(extra));raise ValueError('extra_body_byte')
    return b''.join(parts),True


class ChunkedRanges:
    def __init__(self,opener,url,total,version):
        self.opener=opener;self.url=url;self.total=total;self.version=tuple(version)
        self.received=0;self.records=[];self.failures=[];self.attempts=[]
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
                atomic_json(ROOT/'transfer_progress.json',{'attempts':self.attempts,'received_bytes':self.received})
                parts.append(data);offset+=len(data)
                if offset==end+1:return b''.join(parts)
            except Exception as exc:
                record.update(received_bytes=self.received-before,outcome='fatal',type=type(exc).__name__,reason=str(exc)if isinstance(exc,ValueError)else'transport_error')
                self.attempts.append(record);self.failures.append(dict(record))
                atomic_json(ROOT/'transfer_progress.json',{'attempts':self.attempts,'received_bytes':self.received});raise
        raise ValueError('suffix_segment_limit')
    def read(self,start,end):
        if start<0 or end<start or end>=self.total:raise ValueError('invalid_zip_range')
        return b''.join(self.piece(a,min(end,a+CHUNK-1))for a in range(start,end+1,CHUNK))


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed retry exists')
    old=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text())
    assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256']
    ann_summary=json.loads((prior.previous.ROOT/'summary.json').read_text());annotation=Path(ann_summary['downloads']['lvvis_validation_oracle']['path'])
    payload=json.loads(annotation.read_bytes());pilot=payload['images'][0]
    paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',annotation,Path('semantic/external_archive_resume.py'),Path('semantic/OVIS_EXTERNAL_ARCHIVE_RESUME_PROTOCOL.md')]
    choice=ann_summary['protocol']['frozen_selection']
    feasibility=Path('semantic_artifacts/external_category_feasibility');fv=json.loads((feasibility/'verification.json').read_text())
    assert file_hash(feasibility/'summary.json')==fv['summary_sha256']
    paths.extend([feasibility/'summary.json',feasibility/'verification.json'])
    for a,r in choice.items():
        for seed in range(3):
            path=checkpoint(a,seed);assert file_hash(path)==r['checkpoint_sha256'][str(seed)];paths.append(path)
    p={'round':44,'source':prior.SOURCE,'byte_cap':LIMIT,'max_range_bytes':CHUNK,'max_segments_per_piece':SEGMENTS,'accept_encoding':'identity','fixed_archive_total':old['archive_total_bytes'],
       'fixed_archive_version':old['archive_version'],'pilot':pilot,'excluded_pilot_video':'00000','frozen_selection':choice,
       'input_sha256':{str(p):file_hash(p)for p in paths},'model_evaluation_updates':0}
    pp=ROOT/'protocol.json'
    if pp.exists()and json.loads(pp.read_text())!=p:raise ValueError('Retry protocol changed')
    atomic_json(pp,p);opener=urllib.request.build_opener(prior.previous.previous.HTTPSOnlyRedirects(),
       urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    ranges=None;entries={};pilot_check=None;error=None;form_bytes=0
    try:
        with opener.open(urllib.request.Request(prior.SOURCE,headers={'User-Agent':'D4RT academic chunked range/1.0'}),timeout=20)as response:
            form_path=ROOT/'confirmation.html';form_bytes=prior.previous.read_capped(response,4*2**20,form_path)
        file_id=urllib.parse.parse_qs(urllib.parse.urlsplit(prior.SOURCE).query)['id'][0]
        form=prior.previous.validated_form(form_path.read_text(),file_id);url=form['action']+'?'+urllib.parse.urlencode(form['fields'])
        ranges=ChunkedRanges(opener,url,p['fixed_archive_total'],p['fixed_archive_version']);ranges.account(form_bytes)
        assert ranges.read(0,3)==b'PK\x03\x04'
        reader=prior.RangeZIP(ranges.read,ranges.total);entries=reader.directory()
        expected=PurePosixPath(pilot['file_name']).parts;matching=[name for name in entries if PurePosixPath(name).parts[-len(expected):]==expected]
        if len(matching)!=1:raise ValueError('Pilot member missing/ambiguous')
        raw=reader.member(matching[0]);path=ROOT/'pilot.jpg';path.write_bytes(raw)
        with Image.open(io.BytesIO(raw))as image:
            if image.size!=(pilot['width'],pilot['height']):raise ValueError('GT/pilot dimensions mismatch')
            image.convert('RGB').load()
        suffix_counts={}
        for name in entries:
            key=PurePosixPath(name).parts[-2:];suffix_counts[key]=suffix_counts.get(key,0)+1
        covered=sum(suffix_counts.get(PurePosixPath(im['file_name']).parts[-2:],0)==1 for im in payload['images'])
        pilot_check={'image_id':pilot['id'],'file_name':pilot['file_name'],'archive_member':matching[0],'path':str(path),'sha256':file_hash(path),
            'dimensions_match_GT':True,'GT_image_references_unique_in_archive':covered}
        atomic_json(ROOT/'directory.json',entries)
    except Exception as exc:error={'type':type(exc).__name__,'reason':str(exc) if isinstance(exc,(ValueError,ShortResponse)) else 'network_error','failed_without_padding_or_full_download':True}
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'source_error':error,'ranges':ranges.records if ranges else[],
        'range_failures':ranges.failures if ranges else[],'attempts':ranges.attempts if ranges else[], 'received_bytes':ranges.received if ranges else form_bytes,
        'directory_entries':len(entries),'directory_sha256':file_hash(ROOT/'directory.json')if(ROOT/'directory.json').exists()else None,
        'pilot_check':pilot_check,'no_model_evaluation':True,'no_complete_archive_download':True,'deployment_unchanged':'native_gt'})
    print('DONE prefix suffix archive transfer','error',error,'entries',len(entries),'pilot',pilot_check,flush=True)


if __name__=='__main__':main()
