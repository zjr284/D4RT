"""Fixed COCO candidate/reserve and RGB-only acquisition/screening."""
from collections import defaultdict,Counter
import fcntl,hashlib,json,math,os,re,sqlite3,time,urllib.request,urllib.error
from pathlib import Path
from urllib.parse import urlsplit
os.environ.setdefault('MPLCONFIGDIR','/tmp/d4rt-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from semantic import coco_range_metadata as meta
from semantic import dino_backbone_training as capacity
from semantic import dino_class_video_sampling as balanced
from semantic.dino_low_batch_confirmation import checkpoint as oldckpt
from semantic.ovis_frame_index import fingerprint
from semantic.coco_endpoint_probe import NoRedirect
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/coco_cohort');META=meta.ROOT;INDEX=Path('semantic_artifacts/ovis_frame_index');SEED=20261065
PREVIOUS_RGB=[Path('semantic_artifacts/external_local_rgb/image_records.json'),Path('semantic_artifacts/external_class_panel/image_records.json'),Path('semantic_artifacts/external_capacity_panel/image_records.json')]
ARMS=('S14_gt256','S14_gt32','B14_gt256','B14_gt32','B14_class','B14_class_video','S14_native_sam')
def ckpt(arm,seed):
 if arm in capacity.ARMS:return capacity.ckpt(arm,seed)
 if arm.startswith('B14_class'):return balanced.ckpt(arm[4:],seed)
 return oldckpt('native_sam',seed)

def select_cohort(payload,mapping,count=80,quota=5):
 images=sorted(payload['images'],key=lambda im:im['id']);rank=[images[int(i)]['id']for i in np.random.default_rng(SEED).permutation(len(images))];byid={im['id']:im for im in images};classes=sorted(set(mapping.values())-{1});pools={c:set()for c in classes}
 for a in payload['annotations']:
  cid=mapping.get(a['category_id'])
  if cid in pools and not a['iscrowd']and a['area']>0:pools[cid].add(a['image_id'])
 seen=set();roles={};allfg=set().union(*pools.values())
 for role in ('candidate','reserve'):
  chosen=[]
  for _ in range(quota):
   for cid in classes:
    image=next((i for i in rank if i in pools[cid]and i not in seen),None)
    if image is not None and len(chosen)<count:chosen.append(image);seen.add(image)
  for i in rank:
   if len(chosen)<count and i in allfg and i not in seen:chosen.append(i);seen.add(i)
  assert len(chosen)==count;roles[role]=[byid[i]for i in chosen]
 return {'seed':SEED,'class_quota_rounds':quota,**roles,'metadata_pool_sizes':{str(c):len(pools[c])for c in classes},'role_class_image_counts':{role:{str(c):sum(im['id']in pools[c]for im in roles[role])for c in classes}for role in roles}}

def freeze():
 summary=json.loads((META/'summary.json').read_text());verify=json.loads((META/'verification.json').read_text());assert file_hash(META/'summary.json')==verify['summary_sha256']
 payload=json.loads((META/'instances_val2017.json').read_bytes());mapping={int(k):int(v)for k,v in summary['shared_name_mapping'].items()};manifest=select_cohort(payload,mapping)
 selection=json.loads((balanced.ROOT/'frozen_selection.json').read_text());bv=json.loads((balanced.ROOT/'verification.json').read_text());assert file_hash(balanced.ROOT/'summary.json')==bv['summary_sha256']and selection['validation_selected_arm']=='class_video'
 offsets={'S14_gt256':-1.5,'S14_gt32':-1.25,'B14_gt256':-2.0,'B14_gt32':-2.0,'B14_class':selection['offsets']['class'],'B14_class_video':selection['offsets']['class_video'],'S14_native_sam':-1.75}
 paths=[META/'summary.json',META/'verification.json',META/'instances_val2017.json',balanced.ROOT/'summary.json',balanced.ROOT/'verification.json',balanced.ROOT/'frozen_selection.json',capacity.ROOT/'frozen_selection.json',Path('semantic/coco_cohort.py'),Path('semantic/OVIS_COCO_COHORT_PROTOCOL.md'),INDEX/'index.sqlite',INDEX/'summary.json',INDEX/'verification.json',*PREVIOUS_RGB]
 paths.extend(ckpt(a,s)for a in ARMS for s in range(3));p={'round':65,'selection_seed':SEED,'primary':['B14_class_video','S14_gt32'],'secondary':[['B14_gt32','S14_gt32'],['B14_class_video','B14_gt32'],['B14_gt256','S14_gt256'],['S14_native_sam','S14_gt256']],'models':{a:{'offset':offsets[a],'checkpoint_sha256':{str(s):file_hash(ckpt(a,s))for s in range(3)}}for a in ARMS},'max_requests':320,'byte_cap':256*1024**2,'per_image_byte_cap':5*1024**2,'input_sha256':{str(x):file_hash(x)for x in paths}}
 for name,record in [('protocol.json',p),('manifest.json',manifest)]:
  path=ROOT/name
  if path.exists():assert json.loads(path.read_text())==record
  else:atomic_json(path,record)
 return payload,manifest,p

def acquire(payload,m,p):
 budgetpath=ROOT/'budget.json'
 if not budgetpath.exists():atomic_json(budgetpath,{'started_unix':time.time(),'deadline_unix':time.time()+900})
 budget=json.loads(budgetpath.read_text());rp=ROOT/'requests.json';requests=json.loads(rp.read_text())if rp.exists()else[];ip=ROOT/'image_records.json';records=json.loads(ip.read_text())if ip.exists()else[];completed={r['image_id']:r for r in records};received=sum(r.get('received_bytes',0)for r in requests);licenses={l['id']:l for l in payload['licenses']};opener=urllib.request.build_opener(NoRedirect())
 for role in ('candidate','reserve'):
  for im in m[role]:
   image_id=im['id'];path=ROOT/'images'/im['file_name']
   if image_id in completed:assert file_hash(path)==completed[image_id]['file_sha256'];continue
   uri=urlsplit(im['coco_url']);assert uri.hostname=='images.cocodataset.org'and uri.path==f'/val2017/{im["file_name"]}'and not uri.query and not uri.fragment
   url='https://s3.amazonaws.com/images.cocodataset.org'+uri.path;parts=[];pos=0;total=None;version=None;priorrows=[r for r in requests if r['image_id']==image_id]
   for row in priorrows:
    if row.get('path'):
     data=Path(row['path']).read_bytes();assert len(data)==row['received_bytes']and file_hash(row['path'])==row['sha256'];parts.append(data);pos+=len(data)
    if row.get('total_bytes')is not None:total=row['total_bytes'];version=row['version']
   for attempt in range(len(priorrows),2):
    if total is not None and pos==total:break
    assert len(requests)<p['max_requests']and received<p['byte_cap']and time.time()<budget['deadline_unix'];headers={'Accept-Encoding':'identity'}
    if pos:headers['Range']=f'bytes={pos}-{total-1}'
    row={'image_id':image_id,'role':role,'attempt':attempt,'url':url,'range_start':pos,'received_bytes':0};requests.append(row);atomic_json(rp,requests);datafile=ROOT/'downloads'/f'{image_id:012}_{attempt}.data';datafile.parent.mkdir(exist_ok=True)
    try:
     with opener.open(urllib.request.Request(url,headers=headers),timeout=30)as response:
      h={k:response.headers.get(k)for k in ['Content-Length','Content-Range','Content-Encoding','ETag','Last-Modified','Content-Type']};row.update(status=response.status,headers=h);atomic_json(rp,requests);assert h['Content-Encoding']in(None,'identity')
      if not pos:
       assert response.status==200;total=int(h['Content-Length']);assert 0<total<=p['per_image_byte_cap'];version={k:h[k]for k in ('ETag','Last-Modified')}
      else:
       assert response.status==206 and h['Content-Range']==f'bytes {pos}-{total-1}/{total}'and int(h['Content-Length'])==total-pos
       for k,v in version.items():
        if v is not None:assert h[k]==v
      row.update(total_bytes=total,version=version);atomic_json(rp,requests)
      with datafile.open('xb')as f:
       while pos<total:
        block=response.read(min(65536,total-pos))
        if not block:break
        assert received+len(block)<=p['byte_cap']and time.time()<budget['deadline_unix'];f.write(block);received+=len(block);pos+=len(block);row['received_bytes']+=len(block)
    except Exception as e:row.update(error_type=type(e).__name__,error=str(e))
    finally:
     if datafile.exists():row.update(path=str(datafile),sha256=file_hash(datafile));parts.append(datafile.read_bytes())
     atomic_json(rp,requests)
   if total is None or pos!=total:raise RuntimeError(f'Image {image_id} not complete after at most two requests; no replacement')
   raw=b''.join(parts);assert len(raw)==total;etag=(version['ETag']or'').strip('"');md5=bool(re.fullmatch('[0-9a-fA-F]{32}',etag))
   if md5:assert hashlib.md5(raw).hexdigest()==etag.lower()
   with Image.open(__import__('io').BytesIO(raw))as image:image.verify()
   check=fingerprint(raw);assert (check['width'],check['height'])==(im['width'],im['height']);path.parent.mkdir(exist_ok=True)
   if path.exists():assert file_hash(path)==check['file_sha256']
   else:path.write_bytes(raw)
   record={'role':role,'image_id':image_id,'file_name':im['file_name'],'path':str(path),'source_url':url,'original_coco_url':im['coco_url'],'flickr_url':im['flickr_url'],'license_id':im['license'],'license':licenses[im['license']],'S3_ETag_MD5_verified':md5,'S3_ETag':version['ETag'],**check};records.append(record);completed[image_id]=record;atomic_json(ip,records)
   if len(records)%10==0:print('COCO RGB acquired',len(records),'of160','bytes',received,flush=True)
 return records,requests

def screen(records,p):
 conn=sqlite3.connect(f'file:{INDEX/"index.sqlite"}?mode=ro',uri=True);native=[json.loads(r[0])for r in conn.execute('SELECT record FROM frames ORDER BY path')];conn.close();assert len(native)==42149
 refs=[{'dataset':'OVIS','path':str(Path('semantic_artifacts/ovis/train')/r['path']),**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in native]
 old=[]
 for path in PREVIOUS_RGB:old.extend(json.loads(path.read_text()))
 assert len(old)==1040
 refs.extend({'dataset':'LVVIS_seen130','path':r['path'],**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in old);hashes=np.array([int(r['dhash64'],16)for r in refs],np.uint64);pairs=[]
 def add(left,right,dataset,distance):
  pairs.append({'pair_id':len(pairs),'image_id':left['image_id'],'role':left['role'],'external_path':left['path'],'external_sha256':left['file_sha256'],'reference_dataset':dataset,'reference_path':right['path'],'reference_sha256':right['file_sha256'],'reference_image_id':right.get('image_id'),'reference_role':right.get('role'),'distance':int(distance),'exact_file':left['file_sha256']==right['file_sha256'],'exact_pixels':left['pixel_sha256']==right['pixel_sha256']})
 for r in records:
  distances=np.bitwise_count(hashes^np.uint64(int(r['dhash64'],16)))
  for index in np.flatnonzero(distances<=6):
   ref=refs[int(index)];assert file_hash(ref['path'])==ref['file_sha256'];add(r,ref,ref['dataset'],distances[index])
 for i,left in enumerate(records):
  for right in records[i+1:]:
   d=(int(left['dhash64'],16)^int(right['dhash64'],16)).bit_count()
   if d<=6:add(left,right,'new_coco',d)
 atomic_json(ROOT/'pairs.json',pairs);by=defaultdict(list)
 for pair in pairs:by[pair['image_id']].append(pair)
 sheets=[]
 for image_id,group in by.items():
  entries={r['external_path']:f'NEW {r["role"]} {image_id}'for r in group}
  entries.update({r['reference_path']:r['reference_dataset']+' '+Path(r['reference_path']).parent.name+'/'+Path(r['reference_path']).name for r in group});entries=list(entries.items())
  for page,start in enumerate(range(0,len(entries),24)):
   items=entries[start:start+24];rows=math.ceil(len(items)/4);fig,axes=plt.subplots(rows,4,figsize=(16,rows*2.7),squeeze=False)
   for axis in axes.ravel():axis.axis('off')
   for axis,(path,label)in zip(axes.ravel(),items):
    with Image.open(path)as im:axis.imshow(im.convert('RGB'))
    axis.set_title(label,fontsize=8)
   fig.suptitle(f'COCO RGB-only {image_id} page{page}',fontsize=12);fig.tight_layout(rect=(0,0,1,.97));out=ROOT/f'image_{image_id}_page{page}.png';fig.savefig(out,dpi=120);plt.close(fig);sheets.append({'image_id':image_id,'page':page,'path':str(out),'sha256':file_hash(out),'panels':[{'path':path,'sha256':file_hash(path)}for path,_ in items]})
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'materials.json',{'images':len(records),'roles':dict(Counter(r['role']for r in records)),'image_records_sha256':file_hash(ROOT/'image_records.json'),'pair_count':len(pairs),'pair_sources':dict(Counter(r['reference_dataset']for r in pairs)),'exact_pairs':sum(r['exact_file']or r['exact_pixels']for r in pairs),'pairs_sha256':file_hash(ROOT/'pairs.json'),'sheets':sheets,'no_GT_masks_or_model_outputs':True});print('DONE COCO cohort materials',len(records),len(pairs),len(sheets),flush=True)

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'materials.json').exists():raise RuntimeError('Completed RGB materials exist; do not duplicate requests')
  payload,m,p=freeze();records,requests=acquire(payload,m,p);assert len(records)==160;screen(records,p)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
