"""Independent metadata selection, all RGB and complete two-source matching replay."""
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import struct
import zipfile
from PIL import Image
import numpy as np
from semantic.external_capacity_panel import ROOT,PREVIOUS,GROUPS,COHORT,LOCAL,INDEX,ARCHIVE,ANN,OVIS,EXTRA
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    manifest=json.loads((ROOT/'manifest.json').read_text());assert file_hash(ROOT/'manifest.json')==s['manifest_sha256']
    clips=json.loads(GROUPS.read_text())['groups'];cohort=json.loads(COHORT.read_text());excluded={c['group']for role in ('candidate_confirmation','provisional_reserve')for c in cohort[role]}
    excluded.update(c['group']for c in json.loads((EXTRA/'manifest.json').read_text())['panel']);assert sorted(excluded)==manifest['excluded_previous90']and len(excluded)==90
    eligible=[c for c in clips if c['group']not in excluded];assert len(eligible)==manifest['eligible_model_unevaluated_pool']==743
    payload=json.loads(ANN.read_bytes());cats=json.loads(OVIS.read_bytes())['categories'];normalize=lambda x:' '.join(x.lower().replace('_',' ').split())
    names={normalize(c['name']):c['id']for c in cats};mapping={c['id']:names[normalize(c['name'])]for c in payload['categories']if normalize(c['name'])in names}
    images={im['id']:c['group']for c in eligible for im in c['frames']};presence=defaultdict(set)
    for ann in payload['annotations']:
        target=mapping.get(ann['category_id'])
        if target and target!=1 and ann['image_id']in images and not ann.get('iscrowd',0)and ann.get('area',0)>0:presence[target].add(images[ann['image_id']])
    rank=[eligible[int(i)]['group']for i in np.random.default_rng(p['new_panel_seed']).permutation(len(eligible))]
    selected=[]
    class_ids=sorted(set(mapping.values())-{1})
    for iteration in (0,1):
        for cid in class_ids:
            remaining=[g for g in rank if g in presence[cid]and g not in selected]
            if remaining and len(selected)<40:selected.append(remaining[0])
    for nonperson_only in (True,False):
        for group in rank:
            valid=any(group in presence[cid]for cid in class_ids)if nonperson_only else True
            if valid and group not in selected and len(selected)<40:selected.append(group)
    choices={str(cid):{'eligible_group_count':len(presence[cid]),'selected_groups':[g for g in selected if g in presence[cid]]}for cid in class_ids}
    assert choices==manifest['selection_by_class']and selected==[c['group']for c in manifest['panel']]and len(selected)==40
    assert not set(selected)&excluded
    assert [int(k)for k,v in choices.items()if v['selected_groups']]==s['selected_metadata_classes']
    assert [int(k)for k,v in choices.items()if not v['selected_groups']]==s['missing_metadata_classes']
    records=json.loads((ROOT/'image_records.json').read_text());material=json.loads((ROOT/'materials.json').read_text())
    assert file_hash(ROOT/'materials.json')==s['materials_sha256']and file_hash(ROOT/'image_records.json')==material['image_records_sha256']
    expected=[(clip['group'],t,im)for clip in manifest['panel']for t,im in enumerate(clip['frames'])];assert len(expected)==len(records)==s['images_verified']
    conn=sqlite3.connect(f'file:{INDEX / "index.sqlite"}?mode=ro',uri=True);native=[json.loads(r[0])for r in conn.execute('SELECT record FROM frames ORDER BY path')];conn.close()
    idxs=json.loads((INDEX/'summary.json').read_text());assert file_hash(INDEX/'index.sqlite')==idxs['index_sha256']
    refs=[{'dataset':'OVIS','path':str(Path('semantic_artifacts/ovis/train')/r['path']),**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in native]
    refs.extend({'dataset':'previous_external90','path':r['path'],'group':r['group'],**{k:r[k]for k in ('file_sha256','pixel_sha256','dhash64')}}for r in json.loads((LOCAL/'image_records.json').read_text())+json.loads((EXTRA/'image_records.json').read_text()))
    source=Path(json.loads((ARCHIVE/'summary.json').read_text())['archive_path']);pairs=[]
    with zipfile.ZipFile(source)as z:
        for record,(group,t,im)in zip(records,expected):
            assert (record['group'],record['time'],record['image_id'],record['file_name'])==(group,t,im['id'],im['file_name'])
            raw=Path(record['path']).read_bytes();assert raw==z.read(record['archive_member'])and hashlib.sha256(raw).hexdigest()==record['file_sha256']
            with Image.open(io.BytesIO(raw))as original:
                rgb=original.convert('RGB');rgb.load();assert rgb.size==(im['width'],im['height'])==(record['width'],record['height'])
                assert hashlib.sha256(struct.pack('<II',*rgb.size)+rgb.tobytes()).hexdigest()==record['pixel_sha256']
                gray=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.BILINEAR).resize((9,8),Image.Resampling.BILINEAR))
                bits=''.join('1'if gray[y,x+1]>gray[y,x]else'0'for y in range(8)for x in range(8));assert f'{int(bits,2):016x}'==record['dhash64']
            for ref in refs:
                distance=(int(record['dhash64'],16)^int(ref['dhash64'],16)).bit_count()
                if distance>6:continue
                pairs.append({'pair_id':len(pairs),'group':group,'image_id':record['image_id'],'external_path':record['path'],'external_sha256':record['file_sha256'],
                    'reference_dataset':ref['dataset'],'reference_path':ref['path'],'reference_sha256':ref['file_sha256'],'reference_group':ref.get('group'),
                    'distance':distance,'exact_file':record['file_sha256']==ref['file_sha256'],'exact_pixels':record['pixel_sha256']==ref['pixel_sha256']})
    for i,left in enumerate(records):
        for right in records[i+1:]:
            if left['group']==right['group']:continue
            distance=(int(left['dhash64'],16)^int(right['dhash64'],16)).bit_count()
            if distance<=6:pairs.append({'pair_id':len(pairs),'group':left['group'],'image_id':left['image_id'],'external_path':left['path'],'external_sha256':left['file_sha256'],
                'reference_dataset':'new_panel','reference_path':right['path'],'reference_sha256':right['file_sha256'],'reference_group':right['group'],'distance':distance,
                'exact_file':left['file_sha256']==right['file_sha256'],'exact_pixels':left['pixel_sha256']==right['pixel_sha256']})
    assert pairs==json.loads((ROOT/'pairs.json').read_text())and file_hash(ROOT/'pairs.json')==material['pairs_sha256']
    assert len(pairs)==material['pair_count']==s['near_or_exact_pairs']and dict(Counter(r['reference_dataset']for r in pairs))==s['pair_sources']
    assert sum(r['exact_file']or r['exact_pixels']for r in pairs)==material['exact_pairs']
    for sheet in material['sheets']:
        assert file_hash(sheet['path'])==sheet['sha256']
        for panel in sheet['panels']:assert file_hash(panel['path'])==panel['sha256']
    decisions=json.loads((ROOT/'review_decisions.json').read_text());assert file_hash(ROOT/'review_decisions.json')==s['decisions_sha256']
    assert [d['pair_id']for d in decisions['decisions']]==list(range(len(pairs)))
    bad=set()
    for pair,d in zip(pairs,decisions['decisions']):
        assert d['judgment']in ('distinct','same','uncertain')and d['reason']
        if pair['exact_file']or pair['exact_pixels']:assert d['judgment']!='distinct'
        if d['judgment']!='distinct':
            bad.add(pair['group'])
            if pair['reference_dataset']=='new_panel':bad.add(pair['reference_group'])
    assert sorted(bad)==s['excluded_groups']
    frozen=json.loads((ROOT/'frozen_eligibility.json').read_text());assert file_hash(ROOT/'frozen_eligibility.json')==s['eligibility_sha256']
    assert frozen['accepted_groups']==[g for g in selected if g not in bad]==s['accepted_groups']and frozen['no_replacement']
    assert len(selected)==s['selected_groups']and s['no_model_evaluation']and s['no_GT_masks_decoded']and s['original_random40_and_reserve40_unchanged']
    assert file_hash(Path('semantic/finalize_external_capacity_panel.py'))==s['finalizer_sha256']
    assert not Path('semantic_artifacts/external_capacity_confirmation').exists()
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'independent_metadata_pool':len(eligible),
        'classes_selected':len(s['selected_metadata_classes']),'groups_selected':len(selected),'all_RGB_recomputed':len(records),'all_complete_pairs_recomputed':len(pairs),
        'verification':{'all_frozen_inputs_and_twelve_heads_unchanged':True,'seeded_metadata_selection_and_previous90_exclusion_recomputed':True,
            'all_RGB_CRC_dimensions_pixels_and_three_matching_scopes_recomputed':True,'all_review_decisions_and_frozen_eligibility_covered':True,'no_new_model_output_or_reselection':True}})
    print('DONE independent class-panel verification',len(selected),len(pairs),flush=True)


if __name__=='__main__':main()
