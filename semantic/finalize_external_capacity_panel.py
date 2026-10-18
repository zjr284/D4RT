"""Freeze class-panel eligibility; never auto-judge a nonempty RGB review."""
import json
from pathlib import Path
from semantic.external_capacity_panel import ROOT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed class-panel preparation exists')
    assert not Path('semantic_artifacts/external_capacity_confirmation').exists()
    p=json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    materials=json.loads((ROOT/'materials.json').read_text());pairs=json.loads((ROOT/'pairs.json').read_text())
    assert len(pairs)==materials['pair_count']and file_hash(ROOT/'pairs.json')==materials['pairs_sha256']
    dp=ROOT/'review_decisions.json'
    if pairs:
        if not dp.exists():raise RuntimeError('Actual RGB review required; do not guess decisions')
        decisions=json.loads(dp.read_text())['decisions']
    else:
        decisions=[];atomic_json(dp,{'review_required':False,'reason':'No exact or near matches in the complete frozen screen','decisions':[]})
    assert [d['pair_id']for d in decisions]==list(range(len(pairs)))
    excluded=set()
    for pair,d in zip(pairs,decisions):
        assert d['judgment']in ('distinct','same','uncertain')and d['reason']
        if pair['exact_file']or pair['exact_pixels']:assert d['judgment']!='distinct'
        if d['judgment']!='distinct':
            excluded.add(pair['group'])
            if pair['reference_dataset']=='new_panel':excluded.add(pair['reference_group'])
    manifest=json.loads((ROOT/'manifest.json').read_text())
    accepted=[c['group']for c in manifest['panel']if c['group']not in excluded]
    eligible={'panel_manifest_sha256':file_hash(ROOT/'manifest.json'),'accepted_groups':accepted,'excluded_groups':sorted(excluded),
        'no_replacement':True,'frozen_before_feature_and_prediction':True,'decisions_sha256':file_hash(dp)}
    atomic_json(ROOT/'frozen_eligibility.json',eligible)
    selected=[int(cid)for cid,r in manifest['selection_by_class'].items()if bool(r['selected_groups'])]
    atomic_json(ROOT/'summary.json',{'protocol':p,'manifest_sha256':file_hash(ROOT/'manifest.json'),'materials_sha256':file_hash(ROOT/'materials.json'),
        'eligible_unevaluated_pool':manifest['eligible_model_unevaluated_pool'],'previous90_excluded':len(manifest['excluded_previous90']),
        'selected_groups':len(manifest['panel']),'selected_metadata_classes':selected,'missing_metadata_classes':[int(cid)for cid,r in manifest['selection_by_class'].items()if not r['selected_groups']],
        'images_verified':materials['images'],'near_or_exact_pairs':len(pairs),'pair_sources':materials['pair_sources'],
        'excluded_groups':sorted(excluded),'accepted_groups':accepted,'manual_review_required':bool(pairs),
        'decisions_sha256':file_hash(dp),'eligibility_sha256':file_hash(ROOT/'frozen_eligibility.json'),
        'finalizer_sha256':file_hash(Path('semantic/finalize_external_capacity_panel.py')),'no_model_evaluation':True,'no_GT_masks_decoded':True,
        'original_random40_and_reserve40_unchanged':True,'deployment_unchanged':'native_gt'})
    print('DONE class panel freeze',len(accepted),'groups',len(selected),'metadata classes',flush=True)


if __name__=='__main__':main()
