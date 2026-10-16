"""Verify complete match enumeration and judgment-derived eligibility, no scoring."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
from semantic.external_blind_review import ROOT,PRIOR,INDEX,COHORT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    prior=json.loads((PRIOR/'summary.json').read_text())
    for path,h in prior['protocol']['input_sha256'].items():assert file_hash(path)==h
    records=json.loads((PRIOR/'image_records.json').read_text())
    conn=sqlite3.connect(f'file:{INDEX / "index.sqlite"}?mode=ro',uri=True);native=[json.loads(row[0])for row in conn.execute('SELECT record FROM frames ORDER BY path')];conn.close()
    pairs=json.loads((ROOT/'pairs.json').read_text());expected=[]
    for external in records:
        count=0
        for row in native:
            distance=(int(external['dhash64'],16)^int(row['dhash64'],16)).bit_count()
            if distance>6:continue
            count+=1;path=Path('semantic_artifacts/ovis/train')/row['path']
            expected.append({'pair_id':len(expected),'group':external['group'],'role':external['role'],'external_image_id':external['image_id'],
                'external_path':external['path'],'external_sha256':external['file_sha256'],'ovis_path':str(path),'ovis_sha256':row['file_sha256'],'distance':distance})
        assert count==external['near_match_count']
    assert pairs==expected and len(pairs)==s['pair_count']==79
    materials=json.loads((ROOT/'review_materials.json').read_text());assert file_hash(ROOT/'review_materials.json')==s['review_materials_sha256']
    assert file_hash(ROOT/'pairs.json')==materials['pairs_sha256']
    for sheet in materials['sheets']:
        assert file_hash(sheet['path'])==sheet['sha256']
        expected_paths={p[key]for p in pairs if p['group']==sheet['group']for key in ('external_path','ovis_path')}
        assert {p['path']for p in sheet['panels']}==expected_paths
        for panel in sheet['panels']:assert file_hash(panel['path'])==panel['sha256']
    decisions=json.loads((ROOT/'review_decisions.json').read_text());assert file_hash(ROOT/'review_decisions.json')==s['decisions_sha256']
    assert file_hash(Path('semantic/finalize_external_blind_review.py'))==decisions['finalizer_sha256']
    ds=decisions['decisions'];assert [d['pair_id']for d in ds]==list(range(len(pairs)))
    for d,pair in zip(ds,pairs):assert d['group']==pair['group']and d['judgment']in ('distinct','same','uncertain')and d['reason']
    assert dict(Counter(d['judgment']for d in ds))=={k:v for k,v in s['judgments'].items()if v}
    exclusions=sorted({d['group']for d in ds if d['judgment']!='distinct'});assert exclusions==s['excluded_groups']
    frozen=json.loads((ROOT/'frozen_eligibility.json').read_text());assert file_hash(ROOT/'frozen_eligibility.json')==s['frozen_eligibility_sha256']
    manifest=json.loads(COHORT.read_text());assert file_hash(COHORT)==frozen['original_manifest_sha256']
    assert frozen['candidate_confirmation']==[c['group']for c in manifest['candidate_confirmation']if c['group']not in exclusions]
    assert frozen['reserve']==[c['group']for c in manifest['provisional_reserve']if c['group']not in exclusions]
    assert len(frozen['candidate_confirmation'])==s['accepted_confirmation_groups']and len(frozen['reserve'])==s['accepted_reserve_groups']
    assert not Path('semantic_artifacts/external_semantic_confirmation').exists()
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'complete_pairs_recomputed':len(pairs),
        'verification':{'all_old_inputs_and_models_unchanged':True,'all_near_matches_including_beyond_top20_recomputed':True,
        'all_review_panels_and_judgments_covered':True,'eligibility_derived_without_replacement_or_predictions':True},
        'visual_judgments_automatically_proven':False})
    print('DONE independent blind review audit',len(pairs),flush=True)


if __name__=='__main__':main()
