"""Record completed RGB-only visual judgments; freeze eligibility before prediction."""
import json
from pathlib import Path
from semantic.external_blind_review import ROOT,PRIOR,COHORT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

# Judgments entered after viewing all seven complete RGB review sheets.
VISUAL_REASONS={
    '00006':'External jellyfish against blue water; matches show a road cyclist, sky aerobatics or a turtle/divers in green water. Subjects and scene structures differ.',
    '00045':'External shopping bag on a metal cart and pavement; all 22 matches show goldfish inside an aquarium. Different objects, background and camera view.',
    '00160':'External indoor clarinet performance with a white wall and lamp; both matches show outdoor race banners and vegetation. Different scenes.',
    '00362':'External small drones inside a white-walled room with a table; all three matches show an outdoor expandable gate and vegetation. Different scenes.',
    '00391':'External hands holding a black device over a wooden surface; all 19 matches show a fish in a translucent bowl on another surface. Different objects and scene geometry.',
    '00406':'External aerial forest clearing with a yellow construction vehicle; matches show aquarium fish, underwater turtles/divers or a dog behind a fence. Different scenes.',
    '00420':'External popcorn opening above a tube against a black background; all matches show outdoor boats, aircraft, horse/rider, fenced elephant, scooter or lizards. Different subjects and backgrounds.'}


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed blind review exists')
    assert not Path('semantic_artifacts/external_semantic_confirmation').exists(),'Prediction stage must not exist before eligibility freeze'
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for path,h in protocol['input_sha256'].items():assert file_hash(path)==h
    previous=json.loads((PRIOR/'summary.json').read_text())
    for path,h in previous['protocol']['input_sha256'].items():assert file_hash(path)==h
    materials=json.loads((ROOT/'review_materials.json').read_text());pairs=json.loads((ROOT/'pairs.json').read_text())
    assert file_hash(ROOT/'pairs.json')==materials['pairs_sha256']and len(pairs)==materials['pair_count']==79
    assert {p['group']for p in pairs}==set(VISUAL_REASONS)
    for sheet in materials['sheets']:
        assert file_hash(sheet['path'])==sheet['sha256']
        for panel in sheet['panels']:assert file_hash(panel['path'])==panel['sha256']
    decisions=[{'pair_id':p['pair_id'],'group':p['group'],'judgment':'distinct','reason':VISUAL_REASONS[p['group']]}for p in pairs]
    atomic_json(ROOT/'review_decisions.json',{'reviewer':'Codex RGB-only visual assessment','all_seven_sheets_viewed':True,
        'no_GT_masks_or_predictions_read':True,'decisions':decisions,'finalizer_sha256':file_hash(Path('semantic/finalize_external_blind_review.py'))})
    excluded=sorted({d['group']for d in decisions if d['judgment']in ('same','uncertain')})
    manifest=json.loads(COHORT.read_text());eligibility={'original_manifest_sha256':file_hash(COHORT),'excluded_groups':excluded,
        'candidate_confirmation':[c['group']for c in manifest['candidate_confirmation']if c['group']not in excluded],
        'reserve':[c['group']for c in manifest['provisional_reserve']if c['group']not in excluded],
        'no_replacement':True,'frozen_before_feature_or_prediction_stage':True,'decisions_sha256':file_hash(ROOT/'review_decisions.json')}
    atomic_json(ROOT/'frozen_eligibility.json',eligibility)
    atomic_json(ROOT/'summary.json',{'protocol':protocol,'pair_count':len(pairs),'flagged_groups':len(VISUAL_REASONS),'judgments':{'distinct':len(pairs),'same':0,'uncertain':0},
        'group_reasons':VISUAL_REASONS,'excluded_groups':excluded,'accepted_confirmation_groups':len(eligibility['candidate_confirmation']),
        'accepted_reserve_groups':len(eligibility['reserve']),'review_materials_sha256':file_hash(ROOT/'review_materials.json'),
        'decisions_sha256':file_hash(ROOT/'review_decisions.json'),'frozen_eligibility_sha256':file_hash(ROOT/'frozen_eligibility.json'),
        'no_model_evaluation':True,'no_GT_masks_read':True,'limits':['one RGB visual reviewer','coarse near-hash screening cannot prove no duplicate or pretraining overlap']})
    print('DONE blind review and eligibility freeze',len(pairs),excluded,flush=True)


if __name__=='__main__':main()
