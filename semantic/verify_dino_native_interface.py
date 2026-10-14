"""Independent native-interface output, metric and foundation checks."""
from __future__ import annotations
import json
import subprocess
import numpy as np
from semantic.dino_native_interface_probe import ROOT,old_predictions,PREVIOUS
from semantic.verify_dino_full_interface import metrics
from semantic.semantic_feature_control import VENDOR,WEIGHTS
from semantic.model import GEOMETRY_KEYS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json
from tools.load_opend4rt import DEFAULT_VARIANT


def main():
    summary=json.loads((ROOT/'summary.json').read_text());protocol=summary['protocol']
    assert protocol==json.loads((ROOT/'protocol.json').read_text())
    for path,sha in protocol['input_sha256'].items():assert file_hash(path)==sha
    ids=protocol['video_ids'];cases=summary['cases'];assert len(ids)==len(set(ids))==40 and len(cases)==123
    assert {(c['video_ids'][0],c['seed'])for c in cases if c['kind']=='B1'}=={(v,s)for v in ids for s in range(3)}
    assert [(c['video_ids'],c['seed'])for c in cases if c['kind']=='B2']==[([345,574],s)for s in range(3)]
    for c in cases:
        assert c['queries_per_video']==2048 and c['background_offset']==-1.5
        for name in ('geometry','changed_time_geometry'):
            assert set(c[name])==set(GEOMETRY_KEYS)
            assert all(v['bitwise_equal']and v['max_absolute_difference']==v['mean_absolute_difference']==0 for v in c[name].values())
        assert c['semantic_time']['bitwise_equal']and c['semantic_time']['max_absolute_difference']==0
    statistics={};allmetrics={}
    for seed in range(3):
        name=f'online_seed{seed}.npz';assert file_hash(ROOT/name)==summary['output_sha256'][name]
        with np.load(ROOT/name)as p:online={k:p[k].copy()for k in p.files}
        with np.load(old_predictions(seed))as p:old={k:p[k].copy()for k in p.files}
        assert online['logits'].shape==(81920,26)and np.isfinite(online['logits']).all()
        logits=online['logits'].copy();logits[:,0]-=1.5;assert np.array_equal(logits.argmax(-1),online['predictions'])
        changes=0;largest=0
        for vid in ids:
            a=online['video_ids']==vid;b=old['video_ids']==vid;assert a.sum()==b.sum()==2048
            assert np.array_equal(online['labels'][a],old['labels'][b])
            ref=old['logits'][b].copy();ref[:,0]-=1.5
            count=int((online['predictions'][a]!=ref.argmax(-1)).sum());changes+=count
            error=float(np.abs(online['logits'][a]-old['logits'][b]).max());largest=max(largest,error)
            c=next(c for c in cases if c['kind']=='B1'and c['seed']==seed and c['video_ids']==[vid])
            assert count==c['online_vs_saved_label_changes']and error==c['online_vs_saved_raw_logits']['max_absolute_difference']
            assert metrics(online['predictions'][a],online['labels'][a])==c['per_video'][str(vid)]['online']
        allmetrics[str(seed)]=metrics(online['predictions'],online['labels']);assert allmetrics[str(seed)]==summary['metrics'][str(seed)]['online']
        statistics[str(seed)]={'label_changes':changes,'max_raw_logit_difference':largest}
    fp=protocol['inherited_protocol']
    while 'dinov2_weight_sha256'not in fp:
        if 'inherited_protocol'in fp:fp=fp['inherited_protocol']
        elif 'inherited_feature_coverage_protocol'in fp:fp=fp['inherited_feature_coverage_protocol']
        elif 'inherited_feature_protocol'in fp:fp=fp['inherited_feature_protocol']
        else:raise ValueError('Foundation fingerprint missing')
    foundations={str(WEIGHTS):file_hash(WEIGHTS),str(DEFAULT_VARIANT/'opend4rt.ckpt'):file_hash(DEFAULT_VARIANT/'opend4rt.ckpt'),str(DEFAULT_VARIANT/'model.yaml'):file_hash(DEFAULT_VARIANT/'model.yaml')}
    assert foundations[str(WEIGHTS)]==fp['dinov2_weight_sha256']and foundations[str(DEFAULT_VARIANT/'opend4rt.ckpt')]==fp['d4rt_backbone_sha256']and foundations[str(DEFAULT_VARIANT/'model.yaml')]==fp['d4rt_model_config_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()==fp['dinov2_source_commit']
    native=json.loads((PREVIOUS/'manifest.json').read_text())
    for c in native['final_reserve']:
        assert c['video_id']not in ids and not (ROOT/f"{c['video_id']}.npz").exists()
        for family in ('old','native'):assert not (PREVIOUS/'features'/family/'confirmation'/f"{c['video_id']}.npz").exists()
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'foundation_sha256_after_probe':foundations,
        'counts':{'B1_cases':120,'B2_cases':3,'geometry_output_array_comparisons':1476,'B1_seed_query_comparisons':245760,'B2_seed_query_comparisons':12288},
        'independent_metrics':allmetrics,'online_vs_previous':statistics,
        'verification':{'all_case_counts_and_geometry_time_checks_valid':True,'all_full_predictions_and_confusion_metrics_recomputed':True,
                        'all_inputs_outputs_foundations_unchanged':True,'final40_reserve_not_evaluated':True}})
    print('DONE native interface audit',statistics,flush=True)


if __name__=='__main__':main()
