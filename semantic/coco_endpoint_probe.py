"""Two bounded HTTPS HEAD diagnostics; no redirect, GET, or certificate bypass."""
import json,time,urllib.request,urllib.error
from pathlib import Path
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
ROOT=Path('semantic_artifacts/coco_endpoint_probe')
URLS=['https://images.cocodataset.org/annotations/annotations_trainval2017.zip','https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip']
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):return None

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 if(ROOT/'summary.json').exists()or(ROOT/'responses.json').exists():raise RuntimeError('Probe already attempted; no repeated HEAD')
 paths=[Path('semantic/coco_endpoint_probe.py'),Path('semantic/OVIS_COCO_ENDPOINT_PROTOCOL.md'),Path('semantic_artifacts/coco_metadata_acquisition/summary.json'),Path('semantic_artifacts/coco_metadata_acquisition/verification.json'),Path('semantic_artifacts/dino_class_video_sampling/frozen_selection.json')]
 p={'round':63,'maximum_HEAD':2,'GET_allowed':False,'automatic_redirects':False,'urls':URLS,'input_sha256':{str(x):file_hash(x)for x in paths}};atomic_json(ROOT/'protocol.json',p);opener=urllib.request.build_opener(NoRedirect());rows=[];started=time.monotonic()
 for index,url in enumerate(URLS):
  assert time.monotonic()-started<90
  row={'url':url,'method':'HEAD','received_body_bytes':0};rows.append(row);atomic_json(ROOT/'responses.json',rows)
  try:
   with opener.open(urllib.request.Request(url,method='HEAD',headers={'Accept-Encoding':'identity'}),timeout=30)as response:
    row.update(status=response.status,headers=dict(response.headers.items()),final_url=response.geturl());assert response.geturl()==url
  except urllib.error.HTTPError as e:row.update(status=e.code,error_type=type(e).__name__,error=str(e),headers=dict(e.headers.items()))
  except Exception as e:row.update(status=None,error_type=type(e).__name__,error=str(e))
  atomic_json(ROOT/'responses.json',rows);print('COCO HEAD diagnostic',index,row,flush=True)
  if index==0 and row.get('status')==200:break
 for path,h in p['input_sha256'].items():assert file_hash(path)==h
 atomic_json(ROOT/'summary.json',{'protocol':p,'responses':rows,'responses_sha256':file_hash(ROOT/'responses.json'),'canonical_HTTPS_200':rows[0].get('status')==200,'path_style_candidate_200':any(r['url']==URLS[1]and r.get('status')==200 for r in rows),'candidate_bucket_identity_not_independently_proven':not(rows[0].get('status')==200),'GET_requests':0,'body_bytes_received':0,'model_evaluation':False,'deployment_unchanged':'native_gt'})
 print('DONE COCO HTTPS endpoint probe',flush=True)
if __name__=='__main__':main()
