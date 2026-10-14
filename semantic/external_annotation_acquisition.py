"""Anonymous public large-file confirmation, capped JSON metadata only."""
import http.cookiejar
from html.parser import HTMLParser
import json
from pathlib import Path
import ssl
import urllib.parse
import urllib.request
import urllib.error
from semantic import external_holdout_readiness as previous
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_annotation_acquisition')


class DownloadForm(HTMLParser):
    def __init__(self):super().__init__();self.forms=[];self.current=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form':self.current={'action':a.get('action'),'method':a.get('method','get').lower(),'fields':{}}
        if tag=='input'and self.current is not None and a.get('name'):
            self.current['fields'][a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form'and self.current is not None:self.forms.append(self.current);self.current=None


def validated_form(html,expected_id):
    parser=DownloadForm();parser.feed(html)
    accepted=[]
    for form in parser.forms:
        action=urllib.parse.urlsplit(form['action']or'');fields=form['fields']
        if action.scheme=='https'and action.hostname=='drive.usercontent.google.com'and action.path=='/download':
            if form['method']!='get'or action.query or action.username or action.password:raise ValueError('Unexpected public download action')
            if set(fields)-{'id','export','confirm','uuid'}:raise ValueError('Unexpected public form fields')
            if fields.get('id')!=expected_id or fields.get('export')!='download'or fields.get('confirm')!='t':raise ValueError('Wrong public file identity/operation')
            accepted.append(form)
    if len(accepted)!=1:raise ValueError('No unique permitted public data download form; no login/CAPTCHA handling')
    return accepted[0]


def read_capped(response,cap,destination):
    if cap<=0:raise ValueError('Metadata budget exhausted')
    declared=response.headers.get('Content-Length')
    if declared and int(declared)>cap:raise ValueError('Declared response exceeds metadata cap')
    received=0
    with destination.open('wb')as stream:
        while True:
            chunk=response.read(min(65536,cap-received+1))
            if not chunk:break
            received+=len(chunk)
            if received>cap:raise ValueError('Streaming response exceeds metadata cap')
            stream.write(chunk)
    return received


def acquire(name,url,remaining):
    folder=ROOT/'downloads';folder.mkdir(exist_ok=True);report_path=folder/(name+'.json');path=folder/(name+'.data')
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report.get('path'):assert file_hash(report['path'])==report['sha256']
        return report
    opener=urllib.request.build_opener(previous.HTTPSOnlyRedirects(),urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    canonical=urllib.parse.urlsplit(url);expected_id=urllib.parse.parse_qs(canonical.query)['id'][0]
    received=0;form_path=folder/(name+'_confirmation.html');partial=path.with_suffix('.partial');used_form=False
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'D4RT academic metadata acquisition/1.0'})
        with opener.open(request,timeout=20)as response:
            received+=read_capped(response,min(4*2**20,remaining),form_path)
        raw=form_path.read_bytes()
        try:payload=json.loads(raw);is_json=isinstance(payload,dict)
        except (ValueError,UnicodeDecodeError):is_json=False
        if is_json:
            form_path.replace(path);content_type='application/json';status=200;final_url=previous.safe_url(url)
        else:
            form=validated_form(raw.decode('utf-8'),expected_id);used_form=True
            # Public confirmation fields stay in memory, including transient UUID.
            action=form['action']+'?'+urllib.parse.urlencode(form['fields'])
            request=urllib.request.Request(action,headers={'User-Agent':'D4RT academic metadata acquisition/1.0'})
            with opener.open(request,timeout=20)as response:
                received+=read_capped(response,min(192*2**20,remaining-received),partial)
                content_type=response.headers.get('Content-Type');status=response.status;final_url=previous.safe_url(response.url)
            payload=json.loads(partial.read_bytes())
            if not isinstance(payload,dict):raise ValueError('Expected annotation JSON object')
            partial.replace(path)
        report={'canonical_source_url':url,'used_public_confirmation':used_form,'bytes_received':received,'path':str(path),
            'sha256':file_hash(path),'status':status,'content_type':content_type,'final_url_without_transient_query':final_url,
            'TLS_verification_preserved':True,'no_account_credentials_or_existing_cookies':True}
        if form_path.exists():report['confirmation_snapshot_sha256']=file_hash(form_path)
    except Exception as error:
        # Do not serialize a request URL or transient confirmation values.
        partial.unlink(missing_ok=True)
        report={'canonical_source_url':url,'used_public_confirmation':used_form,'bytes_received':received,
            'error_type':type(error).__name__,'status':getattr(error,'code',None),'TLS_verification_preserved':True,
            'no_account_credentials_or_existing_cookies':True,'no_annotation_ready':True}
        if form_path.exists():report['confirmation_snapshot_sha256']=file_hash(form_path)
    atomic_json(report_path,report);return report


def schema_checks(path):
    payload=json.loads(Path(path).read_bytes());cats=payload.get('categories',[]);annotations=payload.get('annotations',[])
    required={'categories','annotations'};valid=required<=set(payload)and ('videos'in payload or'images'in payload)
    if not valid:return {'supported_keys_present':False}
    ids=[c['id']for c in cats];annids=[a['id']for a in annotations]
    if len(set(ids))!=len(ids)or len(set(annids))!=len(annids):raise ValueError('Duplicate category/annotation IDs')
    nodes=payload['videos']if'videos'in payload else payload['images'];node_ids={v['id']for v in nodes};field='video_id'if'videos'in payload else'image_id'
    if any(a['category_id']not in set(ids)or a[field]not in node_ids for a in annotations):raise ValueError('Annotation references missing category or video/image')
    kinds={}
    for a in annotations:
        seq=a.get('segmentations',[a.get('segmentation')])
        for segment in seq:
            if segment is None:kind='null'
            elif isinstance(segment,dict)and'size'in segment and'counts'in segment:kind='RLE'
            elif isinstance(segment,list):kind='polygon_or_uncompressed_list'
            else:kind='other'
            kinds[kind]=kinds.get(kind,0)+1
    return {'supported_keys_present':True,'unique_category_and_annotation_ids':True,'all_annotation_references_valid':True,
        'nodes':len(nodes),'segment_encoding_counts':kinds}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed annotation acquisition exists; inspect rather than repeat')
    old=json.loads((previous.ROOT/'summary.json').read_text());v=json.loads((previous.ROOT/'verification.json').read_text())
    assert file_hash(previous.ROOT/'summary.json')==v['summary_sha256']
    paths=[previous.ROOT/'summary.json',previous.ROOT/'verification.json',Path('semantic/external_annotation_acquisition.py'),
           Path('semantic/OVIS_EXTERNAL_ANNOTATION_PROTOCOL.md'),Path('semantic_artifacts/ovis/annotations_train.json')]
    paths+=[previous.previous.checkpoint(a,s)for a in previous.ARMS for s in range(3)]
    for arm in previous.ARMS:
        for seed in range(3):assert file_hash(previous.previous.checkpoint(arm,seed))==old['protocol']['frozen_selection'][arm]['checkpoint_sha256'][str(seed)]
    p={'round':37,'annotation_endpoints':previous.ANNOTATIONS,'allowed_public_form_host':'drive.usercontent.google.com',
       'single_annotation_cap_bytes':192*2**20,'total_metadata_cap_including_round36':previous.TOTAL_LIMIT,
       'round36_network_bytes':old['network_bytes_received'],'frozen_selection':old['protocol']['frozen_selection'],
       'model_evaluation_updates':0,'TLS_verification_enabled':True,'input_sha256':{str(p):file_hash(p)for p in paths}}
    pp=ROOT/'protocol.json'
    if pp.exists()and json.loads(pp.read_text())!=p:raise ValueError('Frozen annotation acquisition protocol changed')
    atomic_json(pp,p);received=old['network_bytes_received'];downloads={};audits={};schema={}
    ovis=json.loads(Path('semantic_artifacts/ovis/annotations_train.json').read_text())['categories']
    for name,url in previous.ANNOTATIONS.items():
        r=acquire(name,url,previous.TOTAL_LIMIT-received);received+=r['bytes_received'];downloads[name]=r
        if r.get('path'):
            audits[name]=previous.metadata_audit(r['path'],ovis);schema[name]=schema_checks(r['path'])
        else:audits[name]={'model_evaluation_ready':False,'reason':'Public metadata download did not yield usable JSON'}
        print('annotation acquisition',name,'status',r.get('status'),'bytes',r['bytes_received'],'JSON',bool(r.get('path')),'error',r.get('error_type'),flush=True)
        atomic_json(ROOT/'progress.json',{'protocol_sha256':file_hash(pp),'downloads':downloads,'annotation_audits':audits})
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'downloads':downloads,'annotation_audits':audits,'schema_checks':schema,
       'total_network_bytes_including_round36':received,'no_target_model_evaluation':True,'no_video_archive_download':True,
       'deployment_unchanged':'native_gt','remaining_OVIS_independent_videos':0,
       'pending':['Official selected image retrieval','OVIS duplicate screening','Fixed new confirmation/reserve protocol','Feature/query preprocessing validation']})
    print('DONE public annotation acquisition',flush=True)


if __name__=='__main__':main()
