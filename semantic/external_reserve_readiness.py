"""Refresh RGB-only independence for the original, unscored external reserve."""
from collections import Counter,defaultdict
import fcntl,json,math,sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from semantic import external_semantic_confirmation as original
from semantic import coco_fusion_cohort as coco
from semantic import feature_standardization_control as b14
from semantic.dino_low_batch_confirmation import checkpoint
from semantic.dino_backbone_resource import WEIGHTS as B14_WEIGHTS,VENDOR
from semantic.ovis_frame_index import fingerprint
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/external_reserve_readiness');ARMS=('source_gt','source_gt32','lambda0','native_sam','B14_raw','B14_z')
LV_SOURCES=(Path('semantic_artifacts/external_local_rgb/image_records.json'),Path('semantic_artifacts/external_class_panel/image_records.json'),Path('semantic_artifacts/external_capacity_panel/image_records.json'))
COCO_SOURCES=(coco.prior.old.ROOT/'image_records.json',coco.prior.ROOT/'image_records.json',coco.ROOT/'image_records.json')

def ckpt(arm,seed):return checkpoint(arm,seed)if arm in ARMS[:4]else b14.ckpt('b14_gt32'if arm=='B14_raw'else'z_b14_gt32',seed)

def freeze():
 old=json.loads(original.COHORT.read_text());elig=json.loads((original.REVIEW/'frozen_eligibility.json').read_text());clips=[c for c in old['provisional_reserve']if c['group']in elig['reserve']];assert len(clips)==40 and[c['group']for c in clips]==elig['reserve'];groups={c['group']for c in clips};existing=[]
 for folder in Path('semantic_artifacts').glob('external*/features'):
  existing.extend(str(path)for path in folder.rglob('*.npz')if path.stem in groups)
 assert not existing,('Previously evaluated reserve feature caches',existing)
 oldsource=json.loads((original.RGB/'summary.json').read_text());bs=json.loads((b14.ROOT/'summary.json').read_text());assert file_hash(b14.ROOT/'summary.json')==json.loads((b14.ROOT/'verification.json').read_text())['summary_sha256'];stats=bs['statistics']['b14_gt32']
 source_offsets={'source_gt':-1.5,'source_gt32':-1.25,'lambda0':-1.5,'native_sam':-1.75};models={}
 for arm in ARMS:
  offset=source_offsets[arm]if arm in source_offsets else bs['results']['b14_gt32'if arm=='B14_raw'else'z_b14_gt32']['offsets']['balanced_profile']
  if arm in source_offsets:assert oldsource['protocol']['selection'][arm]['offset']==offset
  models[arm]={'offset':offset,'checkpoint_sha256':{str(seed):file_hash(ckpt(arm,seed))for seed in range(3)}}
 paths=[original.COHORT,original.REVIEW/'summary.json',original.REVIEW/'verification.json',original.REVIEW/'frozen_eligibility.json',original.RGB/'summary.json',original.RGB/'verification.json',original.ROOT/'summary.json',original.ROOT/'verification.json',b14.ROOT/'summary.json',b14.ROOT/'verification.json',Path(stats['path']),original.rgb.prior.resolution.WEIGHTS,B14_WEIGHTS,coco.prior.old.INDEX/'index.sqlite',coco.prior.old.INDEX/'summary.json',coco.prior.old.INDEX/'verification.json',Path('semantic/external_reserve_readiness.py'),Path('semantic/OVIS_EXTERNAL_RESERVE_READINESS_PROTOCOL.md'),Path('semantic/ovis_frame_index.py')]+list(LV_SOURCES)+list(COCO_SOURCES)+sorted(VENDOR.rglob('*.py'))
 paths.extend(ckpt(arm,seed)for arm in ARMS for seed in range(3))
 records=json.loads((original.RGB/'image_records.json').read_text());byname={r['file_name']:r for r in records};selected=[]
 for clip in clips:
  for im in clip['frames']:
   r=byname[im['file_name']];assert r['group']==clip['group']and r['image_id']==im['id'];actual=fingerprint(Path(r['path']).read_bytes())
   for key in ('file_sha256','pixel_sha256','dhash64','width','height'):assert r[key]==actual[key]
   assert(r['width'],r['height'])==(im['width'],im['height']);selected.append(r);paths.append(Path(r['path']))
 p={'round':92,'models':models,'statistics':stats,'primary':['source_gt32','source_gt'],'secondary':[['native_sam','source_gt'],['lambda0','source_gt'],['B14_raw','source_gt32'],['B14_z','B14_raw'],['B14_z','source_gt32']],'query_seed_base':20261041,'queries_per_nonempty_frame':256,'bootstrap_seed':20261093,'bootstrap_repetitions':2000,'no_new_network_GT_masks_features_or_scores':True,'no_existing_reserve_model_cache':True,'input_sha256':{str(path):file_hash(path)for path in sorted(set(paths))}}
 atomic_json(ROOT/'protocol.json',p);atomic_json(ROOT/'manifest.json',{'confirmation':clips,'original_reserve_activated':True,'original_manifest_sha256':file_hash(original.COHORT)});atomic_json(ROOT/'image_records.json',selected);return clips,selected,p

def references(selected):
 conn=sqlite3.connect(f'file:{coco.prior.old.INDEX/"index.sqlite"}?mode=ro',uri=True);native=[json.loads(row[0])for row in conn.execute('SELECT record FROM frames ORDER BY path')];conn.close();assert len(native)==42149;refs=[{'dataset':'OVIS_all42149','path':str(Path('semantic_artifacts/ovis/train')/r['path']),**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in native];exclude={r['file_name']for r in selected};seen=set()
 for source in LV_SOURCES:
  for r in json.loads(source.read_text()):
   if r['file_name']not in exclude and r['path']not in seen:refs.append({'dataset':'LVVIS_old_RGB_excluding_reserve','path':r['path'],'group':r['group'],**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}});seen.add(r['path'])
 for source in COCO_SOURCES:
  for r in json.loads(source.read_text()):refs.append({'dataset':'COCO_previously_evaluated240','path':r['path'],'image_id':r['image_id'],**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}})
 return refs

def plot_panels(items,path,title,columns=8):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=math.ceil(len(items)/columns);fig,axes=plt.subplots(rows,columns,figsize=(columns*2.8,rows*2.1),squeeze=False)
 for axis in axes.ravel():axis.axis('off')
 for axis,item in zip(axes.ravel(),items):
  with Image.open(item['path'])as im:axis.imshow(im.convert('RGB'))
  axis.set_title(item['label'],fontsize=7)
 fig.suptitle(title);fig.tight_layout(rect=(0,0,1,.98));fig.savefig(path,dpi=110);plt.close(fig)
 return {'path':str(path),'sha256':file_hash(path),'panels':[{'path':r['path'],'sha256':file_hash(r['path']),'label':r['label']}for r in items]}

def screen(clips,records,p):
 refs=references(records);hashes=np.array([int(r['dhash64'],16)for r in refs],np.uint64);pairs=[]
 def add(left,right,distance):
  assert file_hash(right['path'])==right['file_sha256'];pairs.append({'pair_id':len(pairs),'group':left['group'],'image_id':left['image_id'],'external_path':left['path'],'reference_dataset':right.get('dataset','new_reserve_cross_group'),'reference_group':right.get('group'),'reference_path':right['path'],'distance':int(distance),'exact_file':left['file_sha256']==right['file_sha256'],'exact_pixels':left['pixel_sha256']==right['pixel_sha256']})
 for r in records:
  distances=np.bitwise_count(hashes^np.uint64(int(r['dhash64'],16)))
  for ix in np.flatnonzero(distances<=6):add(r,refs[int(ix)],distances[ix])
 for i,left in enumerate(records):
  for right in records[i+1:]:
   if left['group']==right['group']:continue
   distance=(int(left['dhash64'],16)^int(right['dhash64'],16)).bit_count()
   if distance<=6:add(left,right,distance)
 atomic_json(ROOT/'pairs.json',pairs);bygroup=defaultdict(list)
 for pair in pairs:bygroup[pair['group']].append(pair)
 sheets=[]
 for group,rows in bygroup.items():
  unique={}
  for row in rows:unique[row['external_path']]='NEW '+row['group']+'/'+Path(row['external_path']).name;unique[row['reference_path']]=row['reference_dataset']+' '+str(row['reference_group']or'')+'/'+Path(row['reference_path']).name
  items=[{'path':path,'label':label}for path,label in unique.items()]
  for page,start in enumerate(range(0,len(items),32)):sheets.append({'group':group,'page':page,**plot_panels(items[start:start+32],ROOT/f'near_{group}_{page}.png','RGB-only reserve '+group,columns=4)})
 overviews=[]
 for page,start in enumerate(range(0,len(records),80)):overviews.append(plot_panels([{'path':r['path'],'label':r['group']+' t'+str(r['time'])}for r in records[start:start+80]],ROOT/f'overview_{page}.png','Original reserve RGB page'+str(page),columns=8))
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'materials.json',{'images':len(records),'groups':len(clips),'reference_counts':dict(Counter(r['dataset']for r in refs)),'pair_count':len(pairs),'exact_pairs':sum(r['exact_file']or r['exact_pixels']for r in pairs),'pairs_sha256':file_hash(ROOT/'pairs.json'),'image_records_sha256':file_hash(ROOT/'image_records.json'),'near_sheets':sheets,'all320_RGB_overviews':overviews,'no_GT_masks_features_or_scores':True});print('DONE original reserve RGB-only refresh',len(records),len(pairs),len(sheets),flush=True)

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'materials.json').exists():raise RuntimeError('Completed reserve materials exist; review instead of repeat')
  clips,records,p=freeze();screen(clips,records,p)
if __name__=='__main__':
 try:main()
 except Exception as e:
  ROOT.mkdir(parents=True,exist_ok=True);atomic_json(ROOT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
