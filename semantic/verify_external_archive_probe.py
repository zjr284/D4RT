"""Check saved accepted ranges and the rejected short-directory diagnosis."""
import json
from pathlib import Path
import struct
from semantic.external_archive_probe import ROOT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert s['source_access_error']['type']=='ValueError'and s['directory_entries']==0 and s['pilot_check']is None
    assert len(s['ranges'])==2 and s['no_model_evaluation']and s['no_complete_archive_download']
    for r in s['ranges']:
        data=Path(r['path']).read_bytes();assert len(data)==r['bytes']==r['end']-r['start']+1 and file_hash(r['path'])==r['sha256']
    first=Path(s['ranges'][0]['path']).read_bytes();assert first==b'PK\x03\x04'
    tail=Path(s['ranges'][1]['path']).read_bytes();at=tail.rfind(b'PK\x05\x06');x=struct.unpack_from('<4s4H2LH',tail,at)
    assert x[1]==x[2]==0 and x[3]==x[4]and at+22+x[7]==len(tail)
    form_size=(ROOT/'confirmation.html').stat().st_size;unaccepted=s['received_bytes']-form_size-sum(r['bytes']for r in s['ranges'])
    assert unaccepted==2228224 and x[5]==2229962 and x[5]-unaccepted==1738
    assert s['received_bytes']<=p['byte_cap']and s['protocol']['pilot_video_excluded_from_future_confirmation']=='00000'
    assert not(ROOT/'pilot.jpg').exists()and not(ROOT/'directory.json').exists()
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),
        'accepted_range_count':2,'archive_end_record_entries':x[4],'declared_directory_bytes':x[5],
        'unaccepted_directory_response_bytes':unaccepted,'missing_directory_bytes':x[5]-unaccepted,
        'verification':{'all_accepted_range_hashes_and_sizes_verified':True,'archive_end_record_independently_parsed':True,
        'short_directory_rejected_without_padding_or_trimming':True,'no_pilot_or_model_evaluation_claimed':True,'all_registered_inputs_unchanged':True}})
    print('DONE failed archive range independent audit')


if __name__=='__main__':main()
