"""Verify the second strict rejection and its exact consumed-byte record."""
import json
from pathlib import Path
from semantic.external_archive_retry import ROOT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert s['source_error']and len(s['ranges'])==len(s['range_failures'])==1
    r=s['ranges'][0];assert r['start']==0 and r['end']==3 and Path(r['path']).read_bytes()==b'PK\x03\x04'and file_hash(r['path'])==r['sha256']
    f=s['range_failures'][0];assert f['end']-f['start']+1==131072 and f['received_bytes']==124964
    form=(ROOT/'confirmation.html').stat().st_size
    assert s['received_bytes']==form+4+f['received_bytes']and s['received_bytes']<=p['byte_cap']
    assert not(ROOT/'pilot.jpg').exists()and not(ROOT/'directory.json').exists()
    assert s['directory_entries']==0 and s['pilot_check']is None and s['no_model_evaluation']and s['no_complete_archive_download']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'failed_range_declared_bytes':131072,
        'failed_range_received_bytes':124964,'missing_bytes':6108,
        'verification':{'accepted_header_range_hashed':True,'failed_range_consumption_count_preserved':True,
        'genuine_EOF_rejected_without_synthesis':True,'no_pilot_or_model_score_claimed':True,'all_registered_inputs_unchanged':True}})
    print('DONE chunked archive failure independent audit')


if __name__=='__main__':main()
