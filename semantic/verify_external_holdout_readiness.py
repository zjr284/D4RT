"""Verify public snapshots, frozen source models and honest readiness state."""
import json
from pathlib import Path
import ssl
from semantic.external_holdout_readiness import ROOT,ARMS,previous
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol']
    assert p==json.loads((ROOT/'protocol.json').read_text())and s['no_target_model_evaluation']and s['no_video_archive_download']
    assert p['TLS_verification_enabled']and ssl.create_default_context().verify_mode==ssl.CERT_REQUIRED
    previous_summary=json.loads((previous.ROOT/'summary.json').read_text());checkpoints={}
    for arm in ARMS:
        assert p['frozen_selection'][arm]==previous_summary['protocol']['selection'][arm]
        for seed in range(3):
            sha=file_hash(previous.checkpoint(arm,seed));assert sha==p['frozen_selection'][arm]['checkpoint_sha256'][str(seed)]
            checkpoints[f'{arm}_{seed}']=sha
    count=0;received=0
    for name,r in s['sources'].items():
        assert r['source_url'].startswith('https://')
        if r.get('path'):
            path=Path(r['path']);assert path.stat().st_size==r['bytes']and file_hash(path)==r['sha256']
            assert r['final_url_without_transient_query'].startswith('https://')and '?'not in r['final_url_without_transient_query']
            assert r['bytes']<=p['page_cap_bytes']if name in p['pages']else r['bytes']<=p['annotation_cap_bytes']
            received+=r['bytes'];count+=1
            if name in p['annotation_endpoints']:
                try:json.loads(path.read_bytes());valid=True
                except (ValueError,UnicodeDecodeError):valid=False
                assert valid==s['annotation_audits'][name]['parseable_JSON']
        else:
            assert r['certificate_verification_kept_enabled'];received+=r['bytes_received_before_failure']
    assert received==s['network_bytes_received']and received<=p['total_cap_bytes']
    assert all(not a['model_evaluation_ready']for a in s['annotation_audits'].values())
    assert s['deployment_unchanged']=='native_gt'and s['remaining_OVIS_independent_videos']==0
    assert not list(ROOT.glob('**/*logits*.npz'))and not list(ROOT.glob('**/*.pt'))and not list(ROOT.glob('**/*.zip'))
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    failure_path=Path('semantic_artifacts/external_holdout_readiness_startup_failure.json')
    failure=json.loads(failure_path.read_text());assert failure['no_new_data_or_model_outputs_before_correction']
    current=Path('semantic/external_holdout_readiness.py').read_text()
    import hashlib
    assert hashlib.sha256(current.replace('else 192*2**20','else192*2**20').encode()).hexdigest()==failure['initial_source_sha256_reconstructed_from_single_syntax_fix']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'checkpoint_sha256':checkpoints,
        'counts':{'public_response_snapshots_verified':count,'frozen_source_heads_verified':12,'new_performance_results':0},
        'startup_failure_sha256':file_hash(failure_path),
        'verification':{'all_public_snapshot_hashes_and_caps_verified':True,'HTTP200_not_mistaken_for_JSON_annotation':True,
        'all_four_source_choices_offsets_and_weights_unchanged':True,'no_model_scoring_training_or_archive_download':True,
        'TLS_verification_preserved':True,'startup_failure_and_syntax_correction_preserved':True}})
    print('DONE independent external resource audit',count)


if __name__=='__main__':main()
