"""Bounded HTTP-range ZIP directory and one fixed image, no model scoring."""
import io
import json
import http.cookiejar
from pathlib import Path,PurePosixPath
import re
import ssl
import struct
import urllib.request
import urllib.parse
import zlib
from PIL import Image
from semantic import external_annotation_acquisition as previous
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_archive_probe')
SOURCE='https://drive.google.com/uc?export=download&id=1vTYUz_XLOBnYb9e7upJsZM-nQz2S6wDn'
LIMIT=16*2**20


def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute()or '..'in p.parts or'\\'in name:raise ValueError('Unsafe archive path')
    return str(p)


class RangeZIP:
    def __init__(self,reader,total):self.reader=reader;self.total=total;self.entries={}
    def directory(self):
        start=max(0,self.total-131072);tail=self.reader(start,self.total-1);pos=tail.rfind(b'PK\x05\x06')
        if pos<0 or pos+22>len(tail):raise ValueError('ZIP end record missing')
        x=struct.unpack_from('<4s4H2LH',tail,pos)
        if x[1]or x[2]or x[3]!=x[4]:raise ValueError('Multipart ZIP unsupported')
        if pos+22+x[7]!=len(tail):raise ValueError('ZIP end comment/size mismatch')
        count,size,offset=x[4],x[5],x[6]
        if count==65535 or size==0xffffffff or offset==0xffffffff:
            if pos<20:raise ValueError('ZIP64 locator missing')
            loc=struct.unpack_from('<4sLQL',tail,pos-20)
            if loc[0]!=b'PK\x06\x07'or loc[1]!=0 or loc[3]!=1:raise ValueError('Invalid ZIP64 locator')
            record=self.reader(loc[2],loc[2]+55);v=struct.unpack('<4sQ2H2L4Q',record)
            if v[0]!=b'PK\x06\x06'or v[4]or v[5]or v[6]!=v[7]:raise ValueError('Invalid ZIP64 end record')
            count,size,offset=v[7],v[8],v[9]
        if size>8*2**20 or offset+size>self.total:raise ValueError('Directory exceeds probe bounds')
        data=self.reader(offset,offset+size-1);at=0
        for _ in range(count):
            if at+46>len(data):raise ValueError('Truncated directory')
            v=struct.unpack_from('<4s6H3L5H2L',data,at)
            if v[0]!=b'PK\x01\x02'or v[13]!=0:raise ValueError('Bad directory entry')
            n,e,c=v[10:13];raw=data[at+46:at+46+n];extra=data[at+46+n:at+46+n+e]
            name=safe_name(raw.decode('utf8'if v[3]&2048 else'cp437'));usize,csize,local=v[9],v[8],v[16]
            if 0xffffffff in (usize,csize,local):
                ptr=0;values=None
                while ptr+4<=len(extra):
                    tag,length=struct.unpack_from('<2H',extra,ptr);block=extra[ptr+4:ptr+4+length];ptr+=4+length
                    if tag==1:values=block
                if values is None:raise ValueError('ZIP64 entry values missing')
                q=0
                sizes=[usize,csize,local]
                for j in range(3):
                    if sizes[j]==0xffffffff:sizes[j]=struct.unpack_from('<Q',values,q)[0];q+=8
                usize,csize,local=sizes
            if name in self.entries:raise ValueError('Duplicate archive member name')
            self.entries[name]={'name':name,'flags':v[3],'method':v[4],'crc32':v[7],'compressed_bytes':csize,'uncompressed_bytes':usize,'local_offset':local}
            at+=46+n+e+c
        if at!=len(data):raise ValueError('Directory trailing data not supported')
        return self.entries
    def member(self,name):
        e=self.entries[name]
        if e['flags']&1 or e['method']not in (0,8):raise ValueError('Encrypted/unsupported member')
        if max(e['compressed_bytes'],e['uncompressed_bytes'])>4*2**20:raise ValueError('Pilot member exceeds size cap')
        header=self.reader(e['local_offset'],e['local_offset']+29);v=struct.unpack('<4s5H3L2H',header)
        if v[0]!=b'PK\x03\x04':raise ValueError('Invalid local header')
        names=self.reader(e['local_offset']+30,e['local_offset']+30+v[9]+v[10]-1)
        if names[:v[9]].decode('utf8'if e['flags']&2048 else'cp437')!=name:raise ValueError('Local/central member name mismatch')
        offset=e['local_offset']+30+v[9]+v[10];data=self.reader(offset,offset+e['compressed_bytes']-1)
        if e['method']==8:
            decoder=zlib.decompressobj(-15);raw=decoder.decompress(data,e['uncompressed_bytes']+1)
            if not decoder.eof or decoder.unused_data:raise ValueError('Invalid deflate stream')
        else:raw=data
        if len(raw)!=e['uncompressed_bytes']or zlib.crc32(raw)&0xffffffff!=e['crc32']:raise ValueError('Member CRC/size mismatch')
        return raw


class RemoteRanges:
    def __init__(self,opener,url):self.opener=opener;self.url=url;self.received=0;self.total=None;self.version=None;self.records=[]
    def read(self,start,end):
        if start<0 or end<start:raise ValueError('Invalid range')
        if end-start+1>LIMIT-self.received:raise ValueError('Probe byte budget exceeded')
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','User-Agent':'D4RT academic range probe/1.0'})
        with self.opener.open(request,timeout=20)as response:
            if response.status!=206:raise ValueError('Server ignored Range; refuse full-body download')
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if not match or tuple(map(int,match.groups()[:2]))!=(start,end):raise ValueError('Unexpected Content-Range')
            total=int(match[3]);version=(response.headers.get('ETag'),response.headers.get('Last-Modified'))
            if self.total is not None and(total!=self.total or version!=self.version):raise ValueError('Archive version changed across ranges')
            self.total=total;self.version=version;data=response.read(end-start+2);self.received+=len(data)
            if len(data)!=end-start+1:raise ValueError('Short/overlong range response')
        path=ROOT/'ranges'/f'{len(self.records):03d}.bin';path.parent.mkdir(exist_ok=True);path.write_bytes(data)
        self.records.append({'start':start,'end':end,'bytes':len(data),'path':str(path),'sha256':file_hash(path)})
        return data


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed archive probe exists')
    old=json.loads((previous.ROOT/'summary.json').read_text());v=json.loads((previous.ROOT/'verification.json').read_text())
    assert file_hash(previous.ROOT/'summary.json')==v['summary_sha256']
    annotation=Path(old['downloads']['lvvis_validation_oracle']['path']);payload=json.loads(annotation.read_bytes());pilot=payload['images'][0]
    paths=[previous.ROOT/'summary.json',previous.ROOT/'verification.json',annotation,Path('semantic/external_archive_probe.py'),Path('semantic/OVIS_EXTERNAL_ARCHIVE_PROTOCOL.md')]
    p={'round':38,'source':SOURCE,'byte_cap':LIMIT,'pilot_image':pilot,'pilot_video_excluded_from_future_confirmation':str(PurePosixPath(pilot['file_name']).parent),
       'model_evaluation_updates':0,'input_sha256':{str(p):file_hash(p)for p in paths}}
    pp=ROOT/'protocol.json'
    if pp.exists()and json.loads(pp.read_text())!=p:raise ValueError('Archive probe protocol changed')
    atomic_json(pp,p);opener=urllib.request.build_opener(previous.previous.HTTPSOnlyRedirects(),
       urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    ranges=None;entry_records={};pilot_check=None;error=None;form_bytes=0
    try:
        with opener.open(urllib.request.Request(SOURCE,headers={'User-Agent':'D4RT academic range probe/1.0'}),timeout=20)as response:
            form_path=ROOT/'confirmation.html';form_bytes=previous.read_capped(response,4*2**20,form_path)
        file_id=urllib.parse.parse_qs(urllib.parse.urlsplit(SOURCE).query)['id'][0]
        form=previous.validated_form(form_path.read_text(),file_id);url=form['action']+'?'+urllib.parse.urlencode(form['fields'])
        ranges=RemoteRanges(opener,url);ranges.received=form_bytes
        ranges.read(0,3);reader=RangeZIP(ranges.read,ranges.total);entry_records=reader.directory()
        expected=PurePosixPath(pilot['file_name']).parts
        matching=[name for name in entry_records if PurePosixPath(name).parts[-len(expected):]==expected]
        if len(matching)!=1:raise ValueError('Pilot archive reference missing/ambiguous')
        raw=reader.member(matching[0]);path=ROOT/'pilot.jpg';path.write_bytes(raw)
        with Image.open(io.BytesIO(raw))as image:
            if image.size!=(pilot['width'],pilot['height']):raise ValueError('Pilot image/GT dimensions mismatch')
            image.convert('RGB').load()
        archive_suffixes={str(PurePosixPath(name).parts[-2:]):name for name in entry_records if name.endswith('.jpg')}
        covered=sum(str(PurePosixPath(image['file_name']).parts[-2:])in archive_suffixes for image in payload['images'])
        pilot_check={'metadata_image_id':pilot['id'],'file_name':pilot['file_name'],'archive_member':matching[0],
            'path':str(path),'sha256':file_hash(path),'GT_and_RGB_dimensions_match':True,'archive_GT_image_reference_matches':covered}
        atomic_json(ROOT/'directory.json',entry_records)
    except Exception as exc:error={'type':type(exc).__name__,'public_range_or_schema_probe_failed':True}
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'source_access_error':error,'ranges':ranges.records if ranges else[],
       'received_bytes':ranges.received if ranges else form_bytes,'archive_total_bytes':ranges.total if ranges else None,
       'archive_version':list(ranges.version)if ranges and ranges.version else None,'directory_entries':len(entry_records),'pilot_check':pilot_check,
       'no_model_evaluation':True,'no_complete_archive_download':True,'deployment_unchanged':'native_gt',
       'pending':['COCO clip adapter and blind video grouping','OVIS duplicate screening','Fixed confirmation/reserve and preprocessing verification']})
    print('DONE archive range probe','error',error,'entries',len(entry_records),'pilot',pilot_check,flush=True)


if __name__=='__main__':main()
