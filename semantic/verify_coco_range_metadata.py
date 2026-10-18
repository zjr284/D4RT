"""Independent stdlib ZIP member/CRC/schema audit over recorded ranges only."""
from collections import Counter,defaultdict
import io,json
from pathlib import Path,PurePosixPath
import zipfile
from semantic.coco_range_metadata import ROOT,TARGET,OVIS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
class SparseArchive(io.RawIOBase):
 def __init__(self,total,ranges):self.total=total;self.position=0;self.parts=[(r['requested_start'],Path(r['path']).read_bytes())for r in ranges]
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.position
 def seek(self,offset,whence=0):
  new=offset if whence==0 else self.position+offset if whence==1 else self.total+offset if whence==2 else -1
  if not 0<=new<=self.total:raise ValueError('invalid virtual seek')
  self.position=new;return new
 def read(self,n=-1):
  end=self.total if n<0 else min(self.total,self.position+n)
  if end-self.position>64*1024**2:raise ValueError('offline read would exceed member cap')
  output=[]
  while self.position<end:
   choices=[(a,b)for a,b in self.parts if a<=self.position<a+len(b)]
   if not choices:raise ValueError('ZIP requested an interval not acquired; refuse network fallback')
   start,data=max(choices,key=lambda item:item[0]+len(item[1]));stop=min(end,start+len(data));output.append(data[self.position-start:stop-start]);self.position=stop
  return b''.join(output)

def main():
 s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 raw_dns=(ROOT/'identity_response.data').read_bytes();dns=json.loads(raw_dns);assert len(raw_dns)==s['DoH_received_bytes']<=65536 and file_hash(ROOT/'identity_response.data')==s['DoH_response_sha256']and dns==json.loads((ROOT/'doh.json').read_text())
 aliases=[a['data'].rstrip('.')for a in dns.get('Answer',[])if a.get('type')==5 and a.get('name','').rstrip('.')=='images.cocodataset.org'];assert aliases==s['verified_CNAME_aliases']and any(a in ('images.cocodataset.org.s3.amazonaws.com','images.cocodataset.org.s3-website-us-east-1.amazonaws.com')for a in aliases)and s['DoH_GET_requests']==1
 rows=json.loads((ROOT/'ranges.json').read_text());head=json.loads((ROOT/'head.json').read_text());total=int(head['headers']['Content-Length']);assert total==s['archive_total_bytes']and head['status']==200
 assert len(rows)==s['GET_requests']<=11 and s['HEAD_requests']==0 and file_hash(ROOT/'ranges.json')==s['ranges_sha256']
 received=0
 for r in rows:
  a,b=r['requested_start'],r['requested_end'];h=r['headers'];assert r['status']==206 and h['Content-Range']==f'bytes {a}-{b}/{total}'and int(h['Content-Length'])==b-a+1
  assert h['Content-Encoding']in(None,'identity')and 0<=r['received_bytes']<=b-a+1
  for key in ('ETag','Last-Modified'):
   if head['headers'][key]is not None:assert h[key]==head['headers'][key]
  f=Path(r['path']);assert f.stat().st_size==r['received_bytes']and file_hash(f)==r['sha256'];received+=r['received_bytes']
 assert received==s['received_bytes']<=p['maximum_received_bytes']
 recorded=json.loads((ROOT/'directory.json').read_text());assert file_hash(ROOT/'directory.json')==s['directory_sha256']
 with zipfile.ZipFile(SparseArchive(total,rows))as archive:
  assert set(recorded)=={str(PurePosixPath(i.filename))for i in archive.infolist()}
  for info in archive.infolist():
   e=recorded[str(PurePosixPath(info.filename))];assert (info.flag_bits,info.compress_type,info.CRC,info.compress_size,info.file_size,info.header_offset)==(e['flags'],e['method'],e['crc32'],e['compressed_bytes'],e['uncompressed_bytes'],e['local_offset'])
  raw=archive.read(TARGET)
 assert raw==(ROOT/'instances_val2017.json').read_bytes()and file_hash(ROOT/'instances_val2017.json')==s['annotation_sha256'];payload=json.loads(raw)
 images=payload['images'];cats=payload['categories'];anns=payload['annotations'];licenses=payload['licenses'];ids={i['id']for i in images};cids={c['id']for c in cats};aids={a['id']for a in anns};assert len(images)==len(ids)==s['images']==5000 and len(cats)==len(cids)==s['categories']==80 and len(anns)==len(aids)==s['annotations']
 for im in images:assert im['width']>0 and im['height']>0 and im['license']in {l['id']for l in licenses}and im['file_name']==f'{im["id"]:012}.jpg'
 for ann in anns:assert ann['image_id']in ids and ann['category_id']in cids and ann['iscrowd']in(0,1)and ann['area']>=0 and isinstance(ann['segmentation'],(dict,list))
 norm=lambda x:' '.join(x.lower().replace('_',' ').split());names={norm(c['name']):c['id']for c in json.loads(OVIS.read_bytes())['categories']};mapping={c['id']:names[norm(c['name'])]for c in cats if norm(c['name'])in names};presence=defaultdict(set)
 for ann in anns:
  if ann['category_id']in mapping and not ann['iscrowd']and ann['area']>0:presence[mapping[ann['category_id']]].add(ann['image_id'])
 assert {str(k):v for k,v in mapping.items()}==s['shared_name_mapping']and {str(c):len(presence[c])for c in mapping.values()}==s['shared_classes_metadata_images']
 assert dict(Counter('RLE'if isinstance(a['segmentation'],dict)else'polygon'for a in anns))==s['segmentation_encodings']and licenses==s['licenses_declared']and dict(Counter(str(im['license'])for im in images))==s['license_image_counts']
 assert s['no_GT_masks_decoded']and s['no_RGB_download_or_model_evaluation']and s['deployment_unchanged']=='native_gt'
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'offline_stdlib_ZIP_directory_and_member_CRC_recomputed':True,'actual_received_bytes':received,'GET_requests_rechecked':len(rows),'all_metadata_links_licenses_and_name_mapping_recomputed':True,'image_ids':5000,'category_ids':80,'annotation_ids':len(aids),'frozen_inputs_models_and_selection_unchanged':True,'no_images_masks_or_model_outputs':True})
 print('DONE independent COCO metadata verification',received,len(aids),flush=True)
if __name__=='__main__':main()
