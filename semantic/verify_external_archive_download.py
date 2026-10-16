"""Verify contiguous transfer history, full frozen directory and independent CRC."""
import json
from pathlib import Path,PurePosixPath
import zipfile
from semantic.external_archive_download import ROOT,REFERENCE
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    state=json.loads((ROOT/'download_state.json').read_text());assert file_hash(ROOT/'download_state.json')==s['state_sha256']
    assert len(state['attempts'])==s['body_requests']<=p['max_body_requests']
    for form in state['forms']:assert Path(form['path']).stat().st_size==form['bytes']and file_hash(form['path'])==form['sha256']
    offset=0
    for i,a in enumerate(state['attempts']):
        assert a['attempt']==i+1 and a['start']==offset
        assert a['end_offset']==a['start']+a['received_bytes']<=p['archive_bytes'];offset=a['end_offset']
        if a.get('status')in (200,206):
            assert a['status']==(206 if a['start']else 200)and a['modified']==p['modified']and a['content_encoding']in (None,'identity')
            if a['content_length']is not None:assert int(a['content_length'])==p['archive_bytes']-a['start']
            if a['start']:assert a['content_range']==f'bytes {a["start"]}-{p["archive_bytes"]-1}/{p["archive_bytes"]}'
        if i+1<len(state['attempts'])and a['outcome']=='HTTPError':assert a['status']in (500,502,503,504)
    received=sum(f['bytes']for f in state['forms'])+sum(a['received_bytes']for a in state['attempts'])
    extra=1 if s['source_error']and s['source_error']['reason']=='extra_archive_byte'else 0
    assert received+extra==state['received_bytes']==s['new_received_bytes']<=p['new_body_byte_cap']
    saved=Path(s['archive_path']);assert saved.stat().st_size==s['saved_bytes']==offset
    assert file_hash(saved)==s['archive_sha256']
    independently_verified=0
    if s['source_error']is None:
        assert offset==p['archive_bytes']
        expected=json.loads((REFERENCE/'directory.json').read_text())
        with zipfile.ZipFile(saved)as archive:
            infos=archive.infolist();names=[str(PurePosixPath(i.filename))for i in infos]
            assert len(names)==len(set(names))==len(expected)and set(names)==set(expected)
            for info,name in zip(infos,names):
                e=expected[name]
                assert (info.CRC,info.file_size,info.compress_size,info.header_offset,info.compress_type,info.flag_bits)==(e['crc32'],e['uncompressed_bytes'],e['compressed_bytes'],e['local_offset'],e['method'],e['flags'])
            assert archive.testzip()is None
            independently_verified=len(infos)
        assert independently_verified==s['members_CRC_verified']
    assert s['no_model_evaluation']and s['cohort_unchanged']and s['deployment_unchanged']=='native_gt'
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'saved_bytes':offset,'new_received_bytes':received+extra,
        'prior_RGB_and_diagnosis_bytes_separate':p['prior_RGB_and_diagnosis_bytes_preserved'],'independent_full_member_CRC_count':independently_verified,
        'verification':{'all_frozen_inputs_unchanged':True,'finite_contiguous_transfer_and_budget_recomputed':True,
        'whole_saved_file_SHA_recomputed':True,'successful_archive_unique_directory_and_all_CRC_rechecked':True,'no_cohort_or_model_change':True}})
    print('DONE independent full archive verification',offset,independently_verified,flush=True)


if __name__=='__main__':main()
