"""Replay exact prefix provenance and logical completeness, then independent ZIP."""
import io
import json
from pathlib import Path,PurePosixPath
import zipfile
from PIL import Image
from semantic.external_archive_resume import ROOT,CHUNK,SEGMENTS
from semantic.verify_external_archive_bounded import SparseArchive
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol']
    assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert s['no_model_evaluation']and s['no_complete_archive_download']and s['deployment_unchanged']=='native_gt'
    attempts=s['attempts'];accepted=[a for a in attempts if a['outcome']in ('complete_response','incomplete_prefix')]
    rejected=[a for a in attempts if a['outcome']!='complete_response'];assert rejected==s['range_failures']
    for i,a in enumerate(attempts):
        assert 1<=a['segment']<=SEGMENTS and 0<a['logical_end']-a['logical_start']+1<=CHUNK
        assert a['end']==a['logical_end']and a['logical_start']<=a['start']<=a['end']
        if a['segment']>1:
            prev=attempts[i-1]
            assert prev['outcome']=='incomplete_prefix'and prev['segment']+1==a['segment']
            assert prev['logical_start']==a['logical_start']and prev['end']==a['end']and a['start']==prev['observed_end']+1
        else:assert a['start']==a['logical_start']
        if a['outcome']!='fatal':
            assert a['status']==206 and a['received_bytes']>0 and a['observed_end']==a['start']+a['received_bytes']-1
            assert a['received_bytes']<=a['end']-a['start']+1
            assert (a['received_bytes']==a['end']-a['start']+1)==(a['outcome']=='complete_response')
        if i+1==len(attempts)or attempts[i+1]['segment']==1:
            if s['source_error']is None:assert a['outcome']=='complete_response'
    assert len(accepted)==len(s['ranges'])
    for a,r in zip(accepted,s['ranges']):
        assert (a['start'],a['observed_end'])==(r['start'],r['end'])
        assert r['bytes']==Path(r['path']).stat().st_size==r['end']-r['start']+1
        assert file_hash(r['path'])==r['sha256']
    form=(ROOT/'confirmation.html').stat().st_size
    assert s['received_bytes']==form+sum(a['received_bytes']for a in attempts)<=p['byte_cap']
    replay=None
    if s['source_error']is None:
        directory=json.loads((ROOT/'directory.json').read_text())
        assert file_hash(ROOT/'directory.json')==s['directory_sha256']
        with zipfile.ZipFile(SparseArchive(p['fixed_archive_total'],s['ranges']))as archive:
            infos=archive.infolist();assert len(infos)==len(directory)==s['directory_entries']
            for info in infos:
                saved=directory[str(PurePosixPath(info.filename))]
                assert (info.CRC,info.compress_size,info.file_size,info.header_offset)==(saved['crc32'],saved['compressed_bytes'],saved['uncompressed_bytes'],saved['local_offset'])
            pilot=s['pilot_check'];raw=archive.read(pilot['archive_member'])
        assert raw==Path(pilot['path']).read_bytes()and file_hash(pilot['path'])==pilot['sha256']
        with Image.open(io.BytesIO(raw))as image:
            assert image.size==(p['pilot']['width'],p['pilot']['height']);image.convert('RGB').load()
        ann_path=next(Path(path)for path in p['input_sha256']if path.endswith('lvvis_validation_oracle.data'))
        images=json.loads(ann_path.read_bytes())['images'];suffixes={}
        for name in directory:
            suffix=PurePosixPath(name).parts[-2:];suffixes[suffix]=suffixes.get(suffix,0)+1
        covered=sum(suffixes.get(PurePosixPath(im['file_name']).parts[-2:],0)==1 for im in images)
        assert covered==pilot['GT_image_references_unique_in_archive']
        replay={'zipfile_independent_entries':len(infos),'pilot_CRC_and_decode':True,'unique_GT_image_paths':covered}
    else:
        assert s['pilot_check']is None and s['directory_entries']==0
        assert not(ROOT/'pilot.jpg').exists()
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),
        'stored_observed_prefixes':len(accepted),'incomplete_or_fatal_responses':len(rejected),'received_bytes':s['received_bytes'],'independent_replay':replay,
        'verification':{'all_input_fingerprints_unchanged':True,'finite_suffix_coverage_and_byte_accounting':True,'accepted_bytes_hashed':True,'no_model_score_or_deployment_change':True}})
    print('DONE suffix archive independent audit',replay,flush=True)


if __name__=='__main__':main()
