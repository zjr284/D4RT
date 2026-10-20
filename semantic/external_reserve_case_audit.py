"""Prepare deterministic frozen-error case sheets; no model inference."""
from collections import Counter,defaultdict
import argparse,fcntl,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as masks
from semantic import external_reserve_confirmation as source
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/external_reserve_case_audit');TYPES=('corrected','regressed','persistent_error','both_correct','mixed_seed')

def case_key(policy,group,image_id,u,v):return hashlib.sha256(f'{policy}|{group}|{image_id}|{u}|{v}'.encode()).hexdigest()
def types(gt,sam,labels):
 a=gt==labels[None];b=sam==labels[None];out=np.full(len(labels),'mixed_seed',dtype='<U20');out[a.all(0)&b.all(0)]='both_correct';out[(~a).all(0)&(~b).all(0)]='persistent_error';out[(~a).all(0)&b.all(0)]='corrected';out[a.all(0)&(~b).all(0)]='regressed';return out

def prepare():
 s=json.loads((source.ROOT/'summary.json').read_text());verification=json.loads((source.ROOT/'verification.json').read_text());assert file_hash(source.ROOT/'summary.json')==verification['summary_sha256'];clips=json.loads((source.ROOT/'manifest.json').read_text())['confirmation'];names={0:'Background',**{r['id']:r['name']for r in json.loads(source.old.OVIS.read_text())['categories']}}
 paths=[source.ROOT/'summary.json',source.ROOT/'verification.json',source.ROOT/'manifest.json',source.ANN,source.old.OVIS,source.old.MAPPING_SUMMARY,Path('semantic/external_reserve_case_audit.py'),Path('semantic/OVIS_EXTERNAL_RESERVE_CASE_PROTOCOL.md')]+[Path(p)for p in s['output_hashes']if '/queries/'in p or'/logits/'in p]+[source.old.RGB/'images'/im['file_name']for clip in clips for im in clip['frames']]
 p={'round':97,'scope':'post-score visual error case audit','selection_rule':'minimum SHA256 of policy/group/imageID/u/v after fixed eligibility','source_summary_sha256':verification['summary_sha256'],'input_sha256':{str(x):file_hash(x)for x in sorted(set(paths))}};atomic_json(ROOT/'protocol.json',p)
 outputs={}
 for policy in source.POLICIES:
  out={}
  for arm in ('source_gt','native_sam','B14_raw'):
   arrays=[]
   for seed in range(3):
    with np.load(s['results'][policy]['arms'][arm]['runs'][seed]['output_path'])as a:z=a['logits'].copy();ids=a['group_ids'].copy();target=a['labels'].copy()
    z[:,0]+=s['protocol']['models'][arm]['offset'];arrays.append(z.argmax(1))
   out[arm]=np.stack(arrays)
  outputs[policy]=(out,ids,target)
 candidates=[];group_counts={};global_by={k:[]for k in TYPES};chosen={};bird_by=defaultdict(list)
 def choose(pool,reason):
  if not pool:return
  r=min(pool,key=lambda r:r['selection_sha256']);key=(r['policy'],r['group'],r['slot'],r['u'],r['v']);chosen.setdefault(key,{**r,'selection_reasons':[]})['selection_reasons'].append(reason)
 for clip in clips:
  group=clip['group'];group_counts[group]={k:0 for k in TYPES};group_candidates=[]
  for policy in source.POLICIES:
   with np.load(source.qp(policy,group))as a:xy=a['xy'].copy();labels=a['labels'].copy()
   out,ids,target=outputs[policy];indices=np.flatnonzero(ids==int(group));assert np.array_equal(labels,target[indices]);gt=out['source_gt'][:,indices];sam=out['native_sam'][:,indices];kinds=types(gt,sam,labels)
   if policy=='strict':group_counts[group]=dict(Counter(kinds));group_counts[group].update({k:group_counts[group].get(k,0)for k in TYPES})
   seen=set()
   for j,(u,v,t)in enumerate(xy):
    pixel=(int(u),int(v),int(t))
    if pixel in seen:continue
    seen.add(pixel);im=clip['frames'][int(t)];record={'policy':policy,'group':group,'slot':int(t),'image_id':im['id'],'path':str(source.old.RGB/'images'/im['file_name']),'u':int(u),'v':int(v),'GT':int(labels[j]),'type':str(kinds[j]),'predictions':{arm:out[arm][:,indices[j]].astype(int).tolist()for arm in out},'selection_sha256':case_key(policy,group,im['id'],int(u),int(v))}
    if policy=='strict':group_candidates.append(record);global_by[record['type']].append(record)
    if policy=='consensus'and group=='00821'and record['GT']==2:bird_by[int(t)].append(record)
  errors=[r for r in group_candidates if r['type']!='both_correct'];choose(errors or group_candidates,'strict_group_error_or_all_correct')
 for kind in TYPES:choose(global_by[kind],'strict_global_'+kind)
 for slot in range(8):choose(bird_by[slot],'consensus_Bird_frame_'+str(slot))
 selected=list(chosen.values());data=json.loads(source.ANN.read_bytes());mapping={int(k):int(v)for k,v in json.loads(source.old.MAPPING_SUMMARY.read_text())['shared_category_mapping'].items()};cn={a['id']:a['name']for a in data['categories']};by=defaultdict(list)
 for ann in data['annotations']:by[ann['image_id']].append(ann)
 for index,r in enumerate(selected):
  with Image.open(r['path'])as im:w,h=im.size
  fx=(r['u']+.5)*w/128-.5;fy=(r['v']+.5)*h/128-.5;x=min(w-1,int((r['u']+.5)*w/128));y=min(h-1,int((r['v']+.5)*h/128));hits=[]
  for a in by[r['image_id']]:
   if masks.decode(a['segmentation'])[y,x]:hits.append({'ann_id':a['id'],'category_id':a['category_id'],'name':cn[a['category_id']],'mapped':mapping.get(a['category_id']),'crowd':bool(a.get('iscrowd',0))})
  known={a['mapped']for a in hits if a['mapped']is not None and not a['crowd']};strict=0 if not hits else(hits[0]['mapped']if len(hits)==1 and not hits[0]['crowd']and hits[0]['mapped']is not None else-1);consensus=0 if not hits else next(iter(known))if len(known)==1 and not any(a['crowd']for a in hits)else-1
  r.update(case_id=index,width=w,height=h,original_center=[fx,fy],original_pixel=[x,y],original_point_hits=hits,native_point_strict_label=strict,native_point_consensus_label=consensus,image_sha256=file_hash(r['path']))
  assert r['GT']==(strict if r['policy']=='strict'else consensus),r
 atomic_json(ROOT/'selection.json',{'source_summary_sha256':p['source_summary_sha256'],'class_names':names,'all40_group_strict_counts':group_counts,'selected':selected,'card_count':len(selected)});print('selected frozen cases',len(selected),flush=True)

def render():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 selection=json.loads((ROOT/'selection.json').read_text());names=selection['class_names'];rows=selection['selected'];sheets=[]
 for start in range(0,len(rows),6):
  cases=rows[start:start+6];fig,axes=plt.subplots(3,4,figsize=(15,10.8));panels=[]
  for ax in axes.ravel():ax.axis('off')
  for j,r in enumerate(cases):
   row,col=divmod(j,2);main=axes[row,col*2];cropax=axes[row,col*2+1]
   with Image.open(r['path'])as im:rgb=im.convert('RGB');rgb.load()
   x,y=r['original_center'];radius=max(16,int(min(rgb.size)*.1));box=(max(0,int(x)-radius),max(0,int(y)-radius),min(rgb.width,int(x)+radius),min(rgb.height,int(y)+radius));crop=rgb.crop(box);main.imshow(rgb);main.plot(x,y,'+',color='red',ms=12,mew=1.5);cropax.imshow(crop);cropax.plot(x-box[0],y-box[1],'+',color='red',ms=12,mew=1.5)
   pred='; '.join(arm+':'+','.join(names[str(c)]for c in r['predictions'][arm])for arm in ('source_gt','native_sam'))
   main.set_title(f"#{r['case_id']} {r['group']} t{r['slot']} {r['policy']} GT={names[str(r['GT'])]}\n{r['type']}",fontsize=8);cropax.set_title(pred.replace('; ','\n'),fontsize=6.5);panels.append({'case_id':r['case_id'],'image_sha256':r['image_sha256'],'crop_box':box})
  fig.suptitle('Frozen point classification cases; red cross = query; crops for review only',fontsize=12);fig.tight_layout(rect=(0,0,1,.96));path=ROOT/f'cases_{start//6:02}.png';fig.savefig(path,dpi=110);plt.close(fig);sheets.append({'path':str(path),'sha256':file_hash(path),'panels':panels})
 atomic_json(ROOT/'materials.json',{'selection_sha256':file_hash(ROOT/'selection.json'),'sheets':sheets,'case_count':len(rows)});print('rendered case sheets',len(sheets),flush=True)

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with(ROOT/'run.lock').open('w')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(ROOT/'materials.json').exists():raise RuntimeError('Materials already exist; inspect rather than regenerate')
  prepare();render()
if __name__=='__main__':main()
