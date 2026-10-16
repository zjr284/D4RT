"""Single exact failed-range replay with safe HTTP error telemetry."""
import http.cookiejar
import json
from pathlib import Path
import ssl
import urllib.request
import urllib.parse
import urllib.error
from semantic import external_archive_probe as archive
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_http_diagnosis')
PREVIOUS=Path('semantic_artifacts/external_rgb_continuation')
LIMIT=2**20
FIELDS=('Content-Range','Content-Length','Content-Encoding','Content-Type','Last-Modified','ETag','Retry-After')


def error_flags(raw):
    text=raw.decode('utf8',errors='replace').lower()
    return {'quota_text':any(s in text for s in ('quota','too many users','too many requests')),
        'login_text':any(s in text for s in ('servicelogin','sign in','authentication required'))}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed diagnosis exists')
    old=json.loads((PREVIOUS/'summary.json').read_text());v=json.loads((PREVIOUS/'verification.json').read_text())
    assert file_hash(PREVIOUS/'summary.json')==v['summary_sha256']
    progress=json.loads((PREVIOUS/'transfer_progress.json').read_text());failure=progress['attempts'][-1]
    start,end=failure['start'],failure['end'];assert end-start+1==30 and failure['type']=='HTTPError'
    paths=[PREVIOUS/'summary.json',PREVIOUS/'verification.json',PREVIOUS/'transfer_progress.json',Path('semantic/external_http_diagnosis.py'),Path('semantic/OVIS_EXTERNAL_HTTP_DIAGNOSIS_PROTOCOL.md')]
    p={'round':47,'start':start,'end':end,'byte_cap':LIMIT,'range_requests':1,'archive_total':old['protocol']['archive_total'],
        'archive_version':old['protocol']['archive_version'],'input_sha256':{str(path):file_hash(path)for path in paths}}
    atomic_json(ROOT/'protocol.json',p);received=0;response_meta=None;error=None
    opener=urllib.request.build_opener(archive.previous.previous.HTTPSOnlyRedirects(),urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    try:
        with opener.open(urllib.request.Request(archive.SOURCE,headers={'User-Agent':'D4RT single status diagnosis/1.0'}),timeout=60)as form_response,(ROOT/'confirmation.html').open('wb')as stream:
            while True:
                remaining=LIMIT-received
                if remaining<=0:raise ValueError('diagnosis_budget_exhausted')
                chunk=form_response.read(min(16384,remaining))
                if not chunk:break
                received+=len(chunk);stream.write(chunk)
        file_id=urllib.parse.parse_qs(urllib.parse.urlsplit(archive.SOURCE).query)['id'][0]
        form=archive.previous.validated_form((ROOT/'confirmation.html').read_text(),file_id)
        url=form['action']+'?'+urllib.parse.urlencode(form['fields'])
        request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'D4RT single status diagnosis/1.0'})
        try:response=opener.open(request,timeout=60)
        except urllib.error.HTTPError as exc:response=exc
        with response:
            status=response.status;headers={field:response.headers.get(field)for field in FIELDS}
            response_meta={'status':status,'headers':headers}
            if status==206:
                expected=f'bytes {start}-{end}/{p["archive_total"]}'
                if headers['Content-Range']!=expected or(headers['ETag'],headers['Last-Modified'])!=tuple(p['archive_version']):raise ValueError('range_or_version_mismatch')
                body=response.read(min(end-start+2,LIMIT-received));received+=len(body)
                (ROOT/'response_body.bin').write_bytes(body)
                response_meta.update(body_bytes=len(body),body_sha256=file_hash(ROOT/'response_body.bin'),local_ZIP_header=(len(body)==30 and body[:4]==b'PK\x03\x04'))
                if len(body)!=30 or not response_meta['local_ZIP_header']:raise ValueError('invalid_current_local_header')
            elif status>=400:
                body=response.read(min(4096,LIMIT-received));received+=len(body);(ROOT/'response_body.bin').write_bytes(body)
                response_meta.update(body_bytes=len(body),body_sha256=file_hash(ROOT/'response_body.bin'),**error_flags(body))
            else:response_meta.update(body_bytes=0,rejected_non206_body=True)
    except Exception as exc:error={'type':type(exc).__name__,'reason':str(exc)if isinstance(exc,ValueError)else'network_or_form_error'}
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'response':response_meta,'source_error':error,'received_bytes':received,
        'no_image_or_model_evaluation':True,'deployment_unchanged':'native_gt','no_retries':True})
    print('DONE single HTTP diagnosis','status',response_meta['status']if response_meta else None,'error',error,flush=True)


if __name__=='__main__':main()
