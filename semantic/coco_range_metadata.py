"""Bounded official GCP range retrieval of COCO val instance metadata only."""
from collections import Counter,defaultdict
import fcntl,json,struct,time,urllib.request,zlib
from pathlib import Path
from semantic.external_archive_probe import RangeZIP
from semantic.external_holdout_readiness import category_mapping
from semantic.external_coco_adapter import OVIS
from semantic import dino_class_video_sampling as prior
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/coco_range_metadata')
URL='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip';TARGET='annotations/instances_val2017.json';LIMIT=64*1024**2
class Ranges:
 def __init__(self,total,version):self.total=total;self.version=version;self.received=0;self.records=[];self.start=time.monotonic()
 def read(self,start,end):
  assert 0<=start<=end<self.total and end-start+1<=32*1024**2
  parts=[];current=start
  for attempt in range(4):
   if current>end:return b''.join(parts)
   assert len(self.records)<11 and time.monotonic()-self.start<300 and end-current+1<=LIMIT-self.received
   request=urllib.request.Request(URL,headers={'Range':f'bytes={current}-{end}','Accept-Encoding':'identity'})
   with urllib.request.urlopen(request,timeout=45)as response:
    row={'requested_start':current,'requested_end':end,'status':response.status,'headers':{k:response.headers.get(k)for k in ['Content-Length','Content-Range','Content-Encoding','ETag','Last-Modified']},'received_bytes':0};self.records.append(row);atomic_json(ROOT/'ranges.json',self.records)
    assert response.status==206 and row['headers']['Content-Range']==f'bytes {current}-{end}/{self.total}'and row['headers']['Content-Encoding']in(None,'identity')
    assert int(row['headers']['Content-Length'])==end-current+1
    for key,value in self.version.items():
     if value is not None:assert row['headers'][key]==value
    blocks=[];remaining=end-current+1
    while remaining:
     block=response.read(min(65536,remaining))
     if not block:break
     assert self.received+len(block)<=LIMIT and time.monotonic()-self.start<300;blocks.append(block);row['received_bytes']+=len(block);self.received+=len(block);remaining-=len(block)
    data=b''.join(blocks);path=ROOT/'ranges'/f'{len(self.records)-1:02}.bin';path.parent.mkdir(exist_ok=True);path.write_bytes(data);row.update(path=str(path),sha256=file_hash(path),short_read=bool(remaining));atomic_json(ROOT/'ranges.json',self.records);parts.append(data);current+=len(data)
    print('COCO range',start,end,'actual',len(data),'remaining',end-current+1,flush=True)
  assert current==end+1,'Logical range short-read budget exhausted';return b''.join(parts)

def decode_member(reader,entry):
 assert not(entry['flags']&1)and entry['method']in(0,8)and entry['compressed_bytes']<=32*1024**2 and entry['uncompressed_bytes']<=64*1024**2
 offset=entry['local_offset'];v=struct.unpack('<4s5H3L2H',reader.read(offset,offset+29));assert v[0]==b'PK\x03\x04'and v[2]==entry['flags']and v[3]==entry['method']
 names=reader.read(offset+30,offset+30+v[9]+v[10]-1);assert names[:v[9]].decode('utf8'if entry['flags']&2048 else'cp437')==TARGET
 first=offset+30+v[9]+v[10];data=reader.read(first,first+entry['compressed_bytes']-1)
 if entry['method']==8:
  decoder=zlib.decompressobj(-15);raw=decoder.decompress(data,entry['uncompressed_bytes']+1);assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
 else:raw=data
 assert len(raw)==entry['uncompressed_bytes']and zlib.crc32(raw)&0xffffffff==entry['crc32'];return raw

def metadata_stats(payload,ovis_categories):
 images={a['id']:a for a in payload['images']};cats={a['id']:a for a in payload['categories']};licenses={a['id']:a for a in payload['licenses']}
 assert len(images)==len(payload['images'])==5000 and len(cats)==len(payload['categories'])==80
 seen=set();present=defaultdict(set);types=Counter()
 for image in images.values():assert image['width']>0 and image['height']>0 and image['license']in licenses and image['file_name']==f'{image["id"]:012}.jpg'
 for a in payload['annotations']:
  assert a['id']not in seen and a['image_id']in images and a['category_id']in cats and a['iscrowd']in(0,1)and a['area']>=0;seen.add(a['id']);assert isinstance(a['segmentation'],(dict,list));types['RLE'if isinstance(a['segmentation'],dict)else'polygon']+=1
  if not a['iscrowd']and a['area']>0:present[a['category_id']].add(a['image_id'])
 mapping=category_mapping(payload['categories'],ovis_categories)
 return {'images':len(images),'annotations':len(payload['annotations']),'categories':len(cats),'segmentation_encodings':dict(types),'licenses_declared':payload['licenses'],'license_image_counts':dict(Counter(str(a['license'])for a in images.values())),'shared_name_mapping':{str(k):v for k,v in mapping.items()},'shared_classes_metadata_images':{str(mapping[k]):len(present[k])for k in sorted(mapping)},'no_GT_masks_decoded':True}

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'summary.json').exists()or(ROOT/'head.json').exists()or(ROOT/'identity_request.json').exists()or(ROOT/'failure.json').exists():raise RuntimeError('Prior acquisition exists; do not restart requests')
  old=json.loads((prior.ROOT/'summary.json').read_text());v=json.loads((prior.ROOT/'verification.json').read_text());assert file_hash(prior.ROOT/'summary.json')==v['summary_sha256'];selection=json.loads((prior.ROOT/'frozen_selection.json').read_text())
  paths=[prior.ROOT/'summary.json',prior.ROOT/'verification.json',prior.ROOT/'frozen_selection.json',OVIS,Path('semantic/coco_range_metadata.py'),Path('semantic/OVIS_COCO_RANGE_METADATA_PROTOCOL.md'),Path('semantic/coco_metadata_acquisition.py'),Path('semantic_artifacts/coco_endpoint_probe/summary.json'),Path('semantic_artifacts/coco_endpoint_probe/verification.json'),Path('semantic_artifacts/coco_endpoint_probe/responses.json'),Path('semantic/external_archive_probe.py')]
  paths.extend(prior.ckpt(a,s)for a in prior.ARMS for s in range(3));p={'round':64,'source_url':URL,'target_member':TARGET,'maximum_received_bytes':LIMIT,'maximum_HEAD':0,'maximum_DoH_GET':1,'maximum_Range_GET':11,'frozen_selection':selection,'input_sha256':{str(x):file_hash(x)for x in paths}};atomic_json(ROOT/'protocol.json',p)
  started=time.monotonic();dns_url='https://dns.google/resolve?name=images.cocodataset.org&type=CNAME'
  atomic_json(ROOT/'identity_request.json',{'url':dns_url,'method':'GET','byte_cap':65536})
  with urllib.request.urlopen(dns_url,timeout=45)as response:
   assert response.status==200;raw_dns=response.read(65537);assert len(raw_dns)<=65536
  (ROOT/'identity_response.data').write_bytes(raw_dns);dns=json.loads(raw_dns);atomic_json(ROOT/'doh.json',dns)
  aliases=[a['data'].rstrip('.')for a in dns.get('Answer',[])if a.get('type')==5 and a.get('name','').rstrip('.')=='images.cocodataset.org']
  assert dns.get('Status')==0 and any(a in ('images.cocodataset.org.s3.amazonaws.com','images.cocodataset.org.s3-website-us-east-1.amazonaws.com')for a in aliases),'Public DNS did not establish expected official bucket alias; do not fetch'
  saved=json.loads(Path('semantic_artifacts/coco_endpoint_probe/responses.json').read_text());candidate=next(r for r in saved if r['url']==URL);assert candidate['status']==200
  head={'status':200,'headers':{k:candidate['headers'].get(k)for k in ['Content-Length','ETag','Last-Modified','Content-Encoding']},'HEAD_reused_from_round63':True};atomic_json(ROOT/'head.json',head);total=int(head['headers']['Content-Length']);assert total==252907541 and head['headers']['ETag']=='"f4bbac642086de4f52a3fdda2de5fa2c"'
  ranges=Ranges(total,{k:head['headers'][k]for k in ['ETag','Last-Modified']});ranges.start=started;assert ranges.read(0,3)==b'PK\x03\x04';reader=RangeZIP(ranges.read,total);entries=reader.directory();atomic_json(ROOT/'directory.json',entries);assert TARGET in entries
  raw=decode_member(ranges,entries[TARGET]);path=ROOT/'instances_val2017.json';path.write_bytes(raw);payload=json.loads(raw);ovis=json.loads(OVIS.read_bytes())['categories'];stats=metadata_stats(payload,ovis)
  for path0,h in p['input_sha256'].items():assert file_hash(path0)==h
  atomic_json(ROOT/'summary.json',{'protocol':p,**stats,'archive_total_bytes':total,'HEAD_requests':0,'DoH_GET_requests':1,'DoH_received_bytes':len(raw_dns),'DoH_response_sha256':file_hash(ROOT/'identity_response.data'),'verified_CNAME_aliases':aliases,'GET_requests':len(ranges.records),'received_bytes':ranges.received,'annotation_sha256':file_hash(path),'target_ZIP_entry':entries[TARGET],'ranges_sha256':file_hash(ROOT/'ranges.json'),'directory_sha256':file_hash(ROOT/'directory.json'),'no_RGB_download_or_model_evaluation':True,'deployment_unchanged':'native_gt'});print('DONE COCO metadata acquisition',stats,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
