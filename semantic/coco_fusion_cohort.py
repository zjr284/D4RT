"""Activate the fixed metadata reserve for frozen complementary feature confirmation."""
from collections import Counter,defaultdict
import fcntl,hashlib,io,json,math,re,sqlite3,time,urllib.request
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
from PIL import Image
from semantic import coco_spatial_cohort as prior
from semantic import feature_fusion_control as models
from semantic.ovis_frame_index import fingerprint
from semantic.coco_endpoint_probe import NoRedirect
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/coco_fusion_cohort')
META=prior.META

def freeze():
 source=json.loads((models.ROOT/'summary.json').read_text());verification=json.loads((models.ROOT/'verification.json').read_text());assert verification['summary_sha256']==file_hash(models.ROOT/'summary.json')
 selection=json.loads((models.ROOT/'frozen_selection.json').read_text());assert source['validation_selected_arm']==selection['validation_selected_arm']=='b14_ade_decoder' and source['eligible_for_separately_preregistered_confirmation']==['b14_ade_decoder']
 prior_p=json.loads((prior.ROOT/'protocol.json').read_text());prior_m=json.loads((prior.ROOT/'manifest.json').read_text());old_m=json.loads((prior.old.ROOT/'manifest.json').read_text())
 excluded={im['id']for role in ('candidate','reserve')for im in old_m[role]}|{im['id']for im in prior_m['candidate']}
 candidate=prior_m['reserve'];assert len(excluded)==200 and len(candidate)==40 and not({im['id']for im in candidate}&excluded)
 m={'candidate':candidate,'activated_original_metadata_reserve':True,'original_selection_seed':20261079,'excluded_previously_evaluated_COCO_ids':sorted(excluded)}
 paths=[Path(x)for x in source['protocol']['input_sha256']]+[models.ROOT/n for n in ('summary.json','verification.json','frozen_selection.json')]+[prior.ROOT/n for n in ('protocol.json','manifest.json','image_records.json','verification.json')]+[Path('semantic/coco_fusion_cohort.py'),Path('semantic/coco_fusion_overview.py'),Path('semantic/OVIS_COCO_FUSION_COHORT_PROTOCOL.md')]
 frozen={}
 for arm in models.ARMS:
  frozen[arm]={'offsets':source['results'][arm]['offsets'],'checkpoint_sha256':{str(seed):file_hash(models.ckpt(arm,seed))for seed in range(3)}}
  paths.extend(models.ckpt(arm,seed)for seed in range(3))
 p={'round':85,'RGB_only_no_GT_mask_or_model_evaluation':True,'primary_model_fixed':'b14_ade_decoder','primary_reference':'b14_zero','models':frozen,'query_seed_base':20261077,'query_count':1024,'bootstrap_seed':20261086,'bootstrap_repetitions':2000,'primary':'strict image_equal_macro decoder_minus_zero spatial_profile','max_GET':160,'max_segments_per_image':4,'max_image_bytes':8*1024**2,'max_total_received_bytes':64*1024**2,'deadline_seconds':900,'socket_timeout_seconds':30,'minimum_GET_start_interval':1,'new_HEAD':0,'near_dhash_threshold':6,'previous_LVVIS_RGB_sources':prior_p['previous_LVVIS_RGB_sources'],'input_sha256':{str(path):file_hash(path)for path in sorted(set(paths))}}
 for name,value in [('protocol.json',p),('manifest.json',m)]:
  if(ROOT/name).exists():assert json.loads((ROOT/name).read_text())==value
  else:atomic_json(ROOT/name,value)
 return json.loads((META/'instances_val2017.json').read_bytes()),m,p

def acquire(payload,m,p):
 licenses={row['id']:row for row in payload['licenses']};ip=ROOT/'image_records.json';records=json.loads(ip.read_text())if ip.exists()else[];done={r['image_id']for r in records};rp=ROOT/'requests.json';requests=json.loads(rp.read_text())if rp.exists()else[];bp=ROOT/'budget.json'
 if not bp.exists():atomic_json(bp,{'started_unix':time.time(),'deadline_unix':time.time()+p['deadline_seconds']})
 budget=json.loads(bp.read_text());opener=urllib.request.build_opener(NoRedirect());received=sum(r['received_bytes']for r in requests)
 for im in m['candidate']:
  iid=im['id']
  if iid in done:continue
  uri=urlsplit(im['coco_url']);assert uri.hostname=='images.cocodataset.org'and uri.path==f'/val2017/{im["file_name"]}'and not uri.query and not uri.fragment;url='https://s3.amazonaws.com/images.cocodataset.org'+uri.path;previous=[r for r in requests if r['image_id']==iid];pieces=[];pos=0;total=None;version=None
  for r in previous:
   if r.get('path'):data=Path(r['path']).read_bytes();assert len(data)==r['received_bytes']and file_hash(r['path'])==r['sha256'];pieces.append(data);pos+=len(data)
   if r.get('total_bytes')is not None:total=r['total_bytes'];version=r['version']
  for attempt in range(len(previous),4):
   if total is not None and pos==total:break
   assert len(requests)<p['max_GET']and received<p['max_total_received_bytes']and time.time()<budget['deadline_unix'];headers={'Accept-Encoding':'identity'}
   if pos:headers['Range']=f'bytes={pos}-{total-1}'
   delay=1-(time.time()-(requests[-1]['started_unix']if requests else budget['started_unix']))
   if delay>0:time.sleep(delay)
   row={'image_id':iid,'attempt':attempt,'url':url,'range_start':pos,'started_unix':time.time(),'received_bytes':0};requests.append(row);atomic_json(rp,requests);path=ROOT/'downloads'/f'{iid:012}_{attempt}.data';path.parent.mkdir(exist_ok=True)
   try:
    with opener.open(urllib.request.Request(url,headers=headers),timeout=30)as response:
     h={key:response.headers.get(key)for key in ('Content-Length','Content-Range','Content-Encoding','ETag','Last-Modified','Content-Type')};row.update(status=response.status,headers=h);assert h['Content-Encoding']in(None,'identity')
     if not pos:assert response.status==200;total=int(h['Content-Length']);assert 0<total<=p['max_image_bytes'];version={key:h[key]for key in ('ETag','Last-Modified')}
     else:assert response.status==206 and h['Content-Range']==f'bytes {pos}-{total-1}/{total}'and int(h['Content-Length'])==total-pos and all(h[key]==value for key,value in version.items()if value is not None)
     row.update(total_bytes=total,version=version);atomic_json(rp,requests)
     with path.open('xb')as out:
      while pos<total:
       chunk=response.read(min(65536,total-pos))
       if not chunk:break
       assert received+len(chunk)<=p['max_total_received_bytes']and time.time()<budget['deadline_unix'];out.write(chunk);received+=len(chunk);pos+=len(chunk);row['received_bytes']+=len(chunk)
   except Exception as e:row.update(error_type=type(e).__name__,error=str(e))
   finally:
    if path.exists():row.update(path=str(path),sha256=file_hash(path));pieces.append(path.read_bytes())
    atomic_json(rp,requests)
   if len(requests)>=2 and all(r['received_bytes']==0 and r.get('error_type')for r in requests[-2:]):raise RuntimeError('Two consecutive zero-byte transport failures; bounded stop')
   if row.get('status')in(400,401,403,404,429):raise RuntimeError('Nonrecoverable HTTP stop')
  if total is None or pos!=total:raise RuntimeError(f'Bounded incomplete candidate {iid}: {pos}/{total}; preserve prefix, do not replace')
  raw=b''.join(pieces);assert len(raw)==total;etag=(version['ETag']or'').strip('"');assert re.fullmatch('[0-9a-fA-F]{32}',etag)and hashlib.md5(raw).hexdigest()==etag.lower()
  with Image.open(io.BytesIO(raw))as image:assert image.format=='JPEG';image.verify()
  result=fingerprint(raw);assert(result['width'],result['height'])==(im['width'],im['height']);path=ROOT/'images'/im['file_name'];path.parent.mkdir(exist_ok=True);assert not path.exists()or file_hash(path)==result['file_sha256']
  if not path.exists():path.write_bytes(raw)
  records.append({'image_id':iid,'role':'candidate','path':str(path),'file_name':im['file_name'],'source_url':url,'original_coco_url':im['coco_url'],'flickr_url':im['flickr_url'],'license_id':im['license'],'license':licenses[im['license']],'S3_ETag':version['ETag'],'S3_ETag_MD5_verified':True,**result});done.add(iid);atomic_json(ip,records);print('new spatial COCO RGB',len(records),'of40','bytes',received,flush=True)
 return records,requests

def references(p):
 refs=prior.references(p)
 for record in json.loads((prior.ROOT/'image_records.json').read_text()):
  refs.append({'dataset':'COCO_previously_evaluated_additional40','path':record['path'],'image_id':record['image_id'],**{key:record[key]for key in ('file_sha256','pixel_sha256','dhash64')}})
 return refs

def screen(records,p):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 refs=references(p);hashes=np.array([int(r['dhash64'],16)for r in refs],np.uint64);pairs=[]
 def add(left,right,distance):pairs.append({'pair_id':len(pairs),'image_id':left['image_id'],'external_path':left['path'],'external_sha256':left['file_sha256'],'reference_dataset':right.get('dataset','new_candidate_COCO'),'reference_path':right['path'],'reference_sha256':right['file_sha256'],'reference_image_id':right.get('image_id'),'distance':int(distance),'exact_file':left['file_sha256']==right['file_sha256'],'exact_pixels':left['pixel_sha256']==right['pixel_sha256']})
 for r in records:
  distances=np.bitwise_count(hashes^np.uint64(int(r['dhash64'],16)))
  for i in np.flatnonzero(distances<=6):ref=refs[int(i)];assert file_hash(ref['path'])==ref['file_sha256'];add(r,ref,distances[i])
 for i,left in enumerate(records):
  for right in records[i+1:]:
   distance=(int(left['dhash64'],16)^int(right['dhash64'],16)).bit_count()
   if distance<=6:add(left,right,distance)
 atomic_json(ROOT/'pairs.json',pairs);groups=defaultdict(list)
 for pair in pairs:groups[pair['image_id']].append(pair)
 sheets=[]
 for iid,group in groups.items():
  entries=list({**{r['external_path']:f'NEW candidate {iid}'for r in group},**{r['reference_path']:r['reference_dataset']+' '+Path(r['reference_path']).parent.name+'/'+Path(r['reference_path']).name for r in group}}.items())
  for page,start in enumerate(range(0,len(entries),24)):
   items=entries[start:start+24];rows=math.ceil(len(items)/4);fig,axes=plt.subplots(rows,4,figsize=(16,rows*2.7),squeeze=False)
   for axis in axes.ravel():axis.axis('off')
   for axis,(path,label)in zip(axes.ravel(),items):
    with Image.open(path)as image:axis.imshow(image.convert('RGB'))
    axis.set_title(label,fontsize=8)
   fig.suptitle(f'RGB-only spatial COCO {iid} page{page}');fig.tight_layout(rect=(0,0,1,.97));out=ROOT/f'image_{iid}_page{page}.png';fig.savefig(out,dpi=120);plt.close(fig);sheets.append({'image_id':iid,'page':page,'path':str(out),'sha256':file_hash(out),'panels':[{'path':path,'sha256':file_hash(path)}for path,_ in items]})
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'materials.json',{'image_records_sha256':file_hash(ROOT/'image_records.json'),'images':len(records),'reference_counts':dict(Counter(r['dataset']for r in refs)),'pair_count':len(pairs),'pairs_sha256':file_hash(ROOT/'pairs.json'),'exact_pairs':sum(r['exact_file']or r['exact_pixels']for r in pairs),'sheets':sheets,'no_GT_masks_or_model_outputs':True});print('DONE new COCO RGB materials',len(records),len(pairs),len(sheets),flush=True)

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'materials.json').exists():raise RuntimeError('Completed RGB materials exist; review without reacquisition')
  payload,m,p=freeze();records,requests=acquire(payload,m,p);assert len(records)==40;screen(records,p)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
