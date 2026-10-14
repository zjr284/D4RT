"""Public metadata preflight; no target-dataset model evaluation or video archive."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import ssl
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from semantic import dino_low_batch_confirmation as previous
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_holdout_readiness')
ARMS=('source_gt','source_gt32','lambda0','native_sam')
PAGES={
    'ytvis_home':'https://youtube-vos.org/',
    'ytvis_dataset':'https://youtube-vos.org/dataset/vis/',
    'ytvis_terms':'https://youtube-vos.org/dataset/term/',
    'ytvis_2019_download':'https://codalab.lisn.upsaclay.fr/competitions/6064',
    'lvvis_author_readme':'https://raw.githubusercontent.com/haochenheheda/LVVIS/main/README.md'}
ANNOTATIONS={
    'lvvis_validation':'https://drive.google.com/uc?export=download&id=1stPD818M3gv7zUV3UIZG1Suru7Tk54jo',
    'lvvis_validation_oracle':'https://drive.google.com/uc?export=download&id=1LuOptZe2GjXa-9QuYveidS-e8xkqsapV'}
TOTAL_LIMIT=256*2**20


def normalize_name(name):return ' '.join(name.lower().replace('_',' ').split())


def category_mapping(external,ovis):
    targets={normalize_name(c['name']):int(c['id'])for c in ovis}
    if len(targets)!=len(ovis):raise ValueError('Ambiguous normalized OVIS names')
    return {int(c['id']):targets[normalize_name(c['name'])]for c in external if normalize_name(c['name'])in targets}


def mapped_label_map(annotations,frame_index,mapping,height,width,size=128):
    out=np.zeros((size,size),np.int16);occupied=np.zeros_like(out,dtype=bool)
    for ann in annotations:
        segmentations=ann['segmentations']
        if frame_index>=len(segmentations)or segmentations[frame_index]is None:continue
        mask=coco_mask.decode(segmentations[frame_index])
        if mask.ndim==3:mask=np.any(mask,axis=2)
        if mask.shape!=(height,width):raise ValueError('External mask dimensions mismatch')
        raster=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST),dtype=bool)
        overlap=raster&occupied;out[raster&~occupied]=mapping.get(int(ann['category_id']),-1);out[overlap]=-1;occupied|=raster
    return out


class HTTPSOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if urllib.parse.urlsplit(newurl).scheme!='https':raise ValueError('Refuse non-HTTPS redirect')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def safe_url(url):
    p=urllib.parse.urlsplit(url);return urllib.parse.urlunsplit((p.scheme,p.netloc,p.path,'',''))


def fetch(name,url,limit,remaining):
    destination=ROOT/'downloads'/f'{name}.bin';report=destination.with_suffix('.json')
    if report.exists():
        r=json.loads(report.read_text())
        if r.get('path'):assert file_hash(r['path'])==r['sha256']
        return r
    destination.parent.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(HTTPSOnlyRedirects(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    partial=destination.with_suffix('.partial');received=0
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'D4RT academic metadata preflight/1.0'})
        with opener.open(req,timeout=20)as response:
            cap=min(limit,remaining);length=response.headers.get('Content-Length')
            if length and int(length)>cap:raise ValueError('Declared response exceeds metadata cap')
            with partial.open('wb')as stream:
                while True:
                    chunk=response.read(min(65536,cap-received+1))
                    if not chunk:break
                    received+=len(chunk)
                    if received>cap:raise ValueError('Response exceeds metadata cap')
                    stream.write(chunk)
            partial.replace(destination)
            r={'source_url':url,'status':response.status,'final_url_without_transient_query':safe_url(response.url),
               'content_type':response.headers.get('Content-Type'),'bytes':received,'path':str(destination),'sha256':file_hash(destination)}
    except Exception as error:
        partial.unlink(missing_ok=True)
        r={'source_url':url,'error_type':type(error).__name__,'error':str(error)[:400],'bytes_received_before_failure':received,
           'status':getattr(error,'code',None),'certificate_verification_kept_enabled':True}
    atomic_json(report,r);return r


def metadata_audit(path,ovis):
    try:payload=json.loads(Path(path).read_bytes())
    except Exception:return {'parseable_JSON':False,'model_evaluation_ready':False}
    if not isinstance(payload,dict):return {'parseable_JSON':True,'expected_annotation_schema':False,'model_evaluation_ready':False}
    cats=payload.get('categories',[])
    if not cats or not isinstance(cats,list)or any(not isinstance(c,dict)or 'id'not in c or'name'not in c for c in cats):
        return {'parseable_JSON':True,'expected_annotation_schema':False,'model_evaluation_ready':False}
    mapping=category_mapping(cats,ovis);index={int(c['id']):c['name']for c in cats}
    return {'parseable_JSON':True,'expected_annotation_schema':'annotations'in payload and ('videos'in payload or'images'in payload),
      'format':'VIS'if'videos'in payload else'COCO_image'if'images'in payload else'unknown',
      'category_count':len(cats),'video_count':len(payload.get('videos',[])),'image_count':len(payload.get('images',[])),
      'annotation_count':len(payload.get('annotations',[])),'shared_category_count':len(mapping),
      'shared_mapping':{str(k):{'source_name':index[k],'ovis_id':v}for k,v in mapping.items()},
      'nonshared_category_count':len(cats)-len(mapping),'source_metadata_inspected':True,
      'model_evaluation_ready':False,'reason':'Images, duplicate screening and final query/selection protocol not yet validated'}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed external metadata preflight exists; inspect before retry')
    old=json.loads((previous.ROOT/'summary.json').read_text());v=json.loads((previous.ROOT/'verification.json').read_text())
    assert file_hash(previous.ROOT/'summary.json')==v['summary_sha256']
    paths=[previous.ROOT/'summary.json',previous.ROOT/'verification.json',Path('semantic/external_holdout_readiness.py'),
           Path('semantic/OVIS_EXTERNAL_HOLDOUT_PROTOCOL.md'),Path('semantic_artifacts/ovis/annotations_train.json')]
    paths+=[previous.checkpoint(a,s)for a in ARMS for s in range(3)]
    p={'round':36,'analysis':'external metadata/resource preflight; no new performance result','pages':PAGES,'annotation_endpoints':ANNOTATIONS,
       'page_cap_bytes':4*2**20,'annotation_cap_bytes':192*2**20,'total_cap_bytes':TOTAL_LIMIT,
       'arms':list(ARMS),'frozen_selection':{a:old['protocol']['selection'][a]for a in ARMS},
       'mapping_rule':'lowercase, underscore to space, collapse whitespace; no semantic merges',
       'unmapped_region_value':-1,'TLS_verification_enabled':True,'model_evaluation_updates':0,
       'input_sha256':{str(p):file_hash(p)for p in paths}}
    pp=ROOT/'protocol.json'
    if pp.exists()and json.loads(pp.read_text())!=p:raise ValueError('External readiness protocol changed')
    atomic_json(pp,p);sources={};received=0
    for name,url in {**PAGES,**ANNOTATIONS}.items():
        cap=4*2**20 if name in PAGES else 192*2**20
        result=fetch(name,url,cap,TOTAL_LIMIT-received);received+=result.get('bytes',result.get('bytes_received_before_failure',0));sources[name]=result
        print('metadata source',name,result.get('status'),result.get('bytes'),result.get('error_type'),flush=True)
        atomic_json(ROOT/'progress.json',{'protocol_sha256':file_hash(pp),'sources':sources})
    ovis=json.loads(Path('semantic_artifacts/ovis/annotations_train.json').read_text())['categories'];audits={}
    for name in ANNOTATIONS:
        if sources[name].get('path'):audits[name]=metadata_audit(sources[name]['path'],ovis)
        else:audits[name]={'model_evaluation_ready':False,'reason':'Annotation retrieval failed'}
    usage=shutil.disk_usage(ROOT);resources={'disk_free_bytes':usage.free,'disk_total_bytes':usage.total,
        'modules':{n:importlib.util.find_spec(n)is not None for n in ('torch','PIL','pycocotools','requests','gdown')}}
    import torch
    resources['cuda_available']=torch.cuda.is_available()
    if resources['cuda_available']:resources['gpu_name']=torch.cuda.get_device_name(0);resources['gpu_total_bytes']=torch.cuda.get_device_properties(0).total_memory
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'sources':sources,'annotation_audits':audits,'resources':resources,'network_bytes_received':received,
        'no_target_model_evaluation':True,'no_video_archive_download':True,'no_credentials_accounts_or_paid_compute_used':True,
        'deployment_unchanged':'native_gt','remaining_OVIS_independent_videos':0,
        'pending':['Find readable full annotation schema','Obtain official selected images','Screen dataset duplicates','Preregister fixed confirmation and complete preprocessing verification']})
    print('DONE external metadata preflight',audits,flush=True)


if __name__=='__main__':main()
