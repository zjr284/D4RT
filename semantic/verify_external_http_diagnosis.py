"""Verify the one safe status probe without further requests."""
import json
from pathlib import Path
import struct
from semantic.external_http_diagnosis import ROOT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert p['range_requests']==1 and s['no_retries']and s['no_image_or_model_evaluation']
    form=(ROOT/'confirmation.html').stat().st_size if(ROOT/'confirmation.html').exists()else 0
    r=s['response'];size=0
    if r and 'body_sha256'in r:
        body=(ROOT/'response_body.bin').read_bytes();size=len(body);assert size==r['body_bytes']and file_hash(ROOT/'response_body.bin')==r['body_sha256']
        if r['status']==206:
            assert size==30 and struct.unpack_from('<4s',body)[0]==b'PK\x03\x04'and r['local_ZIP_header']
            assert r['headers']['Content-Range']==f'bytes {p["start"]}-{p["end"]}/{p["archive_total"]}'
            assert [r['headers']['ETag'],r['headers']['Last-Modified']]==p['archive_version']
        else:
            assert r['status']>=400 and size<=4096
            text=body.decode('utf8',errors='replace').lower()
            assert r['quota_text']==any(t in text for t in ('quota','too many users','too many requests'))
            assert r['login_text']==any(t in text for t in ('servicelogin','sign in','authentication required'))
    assert form+size==s['received_bytes']<=p['byte_cap']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'status':r['status']if r else None,
        'verification':{'all_input_hashes_unchanged':True,'one_request_no_retry_protocol_preserved':True,'safe_status_body_and_budget_replayed':True,'no_historical_status_or_model_performance_inferred':True}})
    print('DONE HTTP diagnosis independent audit',flush=True)


if __name__=='__main__':main()
