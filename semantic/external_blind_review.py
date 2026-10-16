"""Enumerate every near match and render RGB-only review sheets before prediction."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_blind_review')
PRIOR=Path('semantic_artifacts/external_local_rgb')
INDEX=Path('semantic_artifacts/ovis_frame_index')
COHORT=Path('semantic_artifacts/external_rgb_screen/manifest.json')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'review_materials.json').exists():raise RuntimeError('Review materials exist; inspect, do not regenerate')
    previous=json.loads((PRIOR/'summary.json').read_text());verify=json.loads((PRIOR/'verification.json').read_text());assert file_hash(PRIOR/'summary.json')==verify['summary_sha256']
    paths=[PRIOR/'summary.json',PRIOR/'verification.json',PRIOR/'image_records.json',INDEX/'index.sqlite',COHORT,Path('semantic/external_blind_review.py'),Path('semantic/OVIS_EXTERNAL_BLIND_REVIEW_PROTOCOL.md')]
    records=json.loads((PRIOR/'image_records.json').read_text());paths.extend(Path(r['path'])for r in records)
    p={'round':50,'RGB_only_review':True,'input_sha256':{str(path):file_hash(path)for path in paths}}
    atomic_json(ROOT/'protocol.json',p)
    connection=sqlite3.connect(f'file:{INDEX / "index.sqlite"}?mode=ro',uri=True);ovis=[json.loads(row[0])for row in connection.execute('SELECT record FROM frames ORDER BY path')];connection.close()
    hashes=np.array([int(r['dhash64'],16)for r in ovis],np.uint64);pairs=[];by_group=defaultdict(list)
    for external in records:
        if not external['near_match_count']:continue
        distances=np.bitwise_count(hashes^np.uint64(int(external['dhash64'],16)));matches=np.flatnonzero(distances<=6);assert len(matches)==external['near_match_count']
        for index in matches:
            native=ovis[int(index)];path=Path('semantic_artifacts/ovis/train')/native['path'];assert file_hash(path)==native['file_sha256']
            pair={'pair_id':len(pairs),'group':external['group'],'role':external['role'],'external_image_id':external['image_id'],
                'external_path':external['path'],'external_sha256':external['file_sha256'],'ovis_path':str(path),'ovis_sha256':native['file_sha256'],'distance':int(distances[index])}
            pairs.append(pair);by_group[external['group']].append(pair)
    atomic_json(ROOT/'pairs.json',pairs);sheets=[]
    for group,group_pairs in sorted(by_group.items()):
        panels={}
        for pair in group_pairs:
            panels[pair['external_path']]=f'EXT {group} image {pair["external_image_id"]}'
        for pair in group_pairs:panels[pair['ovis_path']]='OVIS '+str(Path(pair['ovis_path']).relative_to('semantic_artifacts/ovis/train'))
        count=len(panels);columns=4;rows=math.ceil(count/columns)
        fig,axes=plt.subplots(rows,columns,figsize=(16,rows*2.7),squeeze=False)
        for axis in axes.ravel():axis.axis('off')
        for axis,(path,label)in zip(axes.ravel(),panels.items()):
            with Image.open(path)as image:axis.imshow(image.convert('RGB'))
            axis.set_title(label,fontsize=8)
        fig.suptitle(f'RGB-only group {group}: all flagged external frames + all matched OVIS frames',fontsize=12)
        fig.tight_layout(rect=(0,0,1,0.97));path=ROOT/f'group_{group}.png';fig.savefig(path,dpi=120);plt.close(fig)
        sheets.append({'group':group,'path':str(path),'sha256':file_hash(path),'panels':[{'path':path,'label':label,'sha256':file_hash(path)}for path,label in panels.items()]})
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'review_materials.json',{'pair_count':len(pairs),'pairs_sha256':file_hash(ROOT/'pairs.json'),'sheets':sheets,'no_model_evaluation':True})
    print('DONE RGB review preparation',len(pairs),'pairs',len(sheets),'sheets',flush=True)


if __name__=='__main__':main()
