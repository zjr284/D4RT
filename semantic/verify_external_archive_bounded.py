"""Independently replay saved ranges with stdlib zipfile, never fetch data."""
import io
import json
import zipfile
from pathlib import Path,PurePosixPath
from PIL import Image
from semantic.external_archive_bounded import ROOT,CHUNK,ATTEMPTS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


class SparseArchive(io.RawIOBase):
    def __init__(self,total,records):self.total=total;self.records=records;self.position=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.total+offset
        if self.position<0:raise ValueError('negative seek')
        return self.position
    def read(self,n=-1):
        end=self.total if n<0 else min(self.total,self.position+n);parts=[]
        while self.position<end:
            candidates=[r for r in self.records if r['start']<=self.position<=r['end']]
            if not candidates:raise ValueError('Replay asked for an unobserved byte')
            r=max(candidates,key=lambda r:r['end']);data=Path(r['path']).read_bytes()
            count=min(end-self.position,r['end']-self.position+1)
            parts.append(data[self.position-r['start']:self.position-r['start']+count]);self.position+=count
        return b''.join(parts)


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol']
    assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert s['no_model_evaluation']and s['no_complete_archive_download']and s['deployment_unchanged']=='native_gt'
    attempts=s['attempts'];accepted=[a for a in attempts if a['outcome']=='accepted']
    rejected=[a for a in attempts if a['outcome']=='rejected'];assert rejected==s['range_failures']
    for i,a in enumerate(attempts):
        assert 1<=a['attempt']<=ATTEMPTS and 0<a['end']-a['start']+1<=CHUNK
        if a['attempt']>1:
            previous=attempts[i-1];assert previous['start']==a['start']and previous['end']==a['end']
            assert previous['retryable']and previous['attempt']+1==a['attempt']
        if a['outcome']=='accepted':assert a['received_bytes']==a['end']-a['start']+1 and a['status']==206
    assert len(accepted)==len(s['ranges'])
    for a,r in zip(accepted,s['ranges']):
        assert (a['start'],a['end'])==(r['start'],r['end'])
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
        'accepted_ranges':len(accepted),'rejected_attempts':len(rejected),'received_bytes':s['received_bytes'],'independent_replay':replay,
        'verification':{'all_input_fingerprints_unchanged':True,'finite_retry_and_byte_accounting':True,'accepted_bytes_hashed':True,'no_model_score_or_deployment_change':True}})
    print('DONE bounded archive independent audit',replay,flush=True)


if __name__=='__main__':main()
