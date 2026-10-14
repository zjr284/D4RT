"""Independent successful JSON schema/mapping/RLE and frozen-head audit."""
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as coco_mask
from semantic.external_annotation_acquisition import ROOT,previous
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol'];assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    assert s['no_target_model_evaluation']and s['no_video_archive_download']and s['deployment_unchanged']=='native_gt'
    received=p['round36_network_bytes'];success=[];checks=[]
    ovis=json.loads(Path('semantic_artifacts/ovis/annotations_train.json').read_text())['categories']
    target={' '.join(c['name'].lower().replace('_',' ').split()):int(c['id'])for c in ovis}
    for name,d in s['downloads'].items():
        received+=d['bytes_received'];assert d['TLS_verification_preserved']and d['no_account_credentials_or_existing_cookies']
        if not d.get('path'):assert d['no_annotation_ready'];continue
        path=Path(d['path']);assert file_hash(path)==d['sha256']and path.stat().st_size<=p['single_annotation_cap_bytes']
        payload=json.loads(path.read_bytes());audit=s['annotation_audits'][name];schema=s['schema_checks'][name]
        cats=payload['categories'];anns=payload['annotations'];images=payload.get('images',[]);videos=payload.get('videos',[])
        assert audit['category_count']==len(cats)and audit['annotation_count']==len(anns)
        assert audit['image_count']==len(images)and audit['video_count']==len(videos)
        ids={c['id']for c in cats};assert len(ids)==len(cats)and len({a['id']for a in anns})==len(anns)
        mapping={str(c['id']):{'source_name':c['name'],'ovis_id':target[' '.join(c['name'].lower().replace('_',' ').split())]}for c in cats if' '.join(c['name'].lower().replace('_',' ').split())in target}
        assert mapping==audit['shared_mapping']and len(mapping)==audit['shared_category_count']
        assert len(cats)-len(mapping)==audit['nonshared_category_count']and not audit['model_evaluation_ready']
        node_ids={n['id']for n in (videos if videos else images)};key='video_id'if videos else'image_id'
        assert all(a['category_id']in ids and a[key]in node_ids for a in anns)
        assert schema['unique_category_and_annotation_ids']and schema['all_annotation_references_valid']
        if images:
            lookup={n['id']:n for n in images}
            for ix in np.linspace(0,len(anns)-1,10,dtype=int):
                ann=anns[int(ix)];image=lookup[ann['image_id']];mask=coco_mask.decode(ann['segmentation'])
                assert mask.shape==(image['height'],image['width'])
                checks.append({'annotation_index':int(ix),'image_id':ann['image_id'],'decoded_shape':list(mask.shape),'foreground_pixels':int(mask.sum())})
        success.append(name)
    assert received==s['total_network_bytes_including_round36']and received<=p['total_metadata_cap_including_round36']
    for arm,choice in p['frozen_selection'].items():
        for seed in range(3):assert file_hash(previous.previous.checkpoint(arm,seed))==choice['checkpoint_sha256'][str(seed)]
    assert not list(ROOT.glob('**/*logits*.npz'))and not list(ROOT.glob('**/*.pt'))and not list(ROOT.glob('**/*.zip'))
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'successful_annotation_sources':success,
        'RLE_sample_checks':checks,'counts':{'real_annotation_sources_verified':len(success),'frozen_heads_verified':12,'new_model_scores':0},
        'verification':{'all_successful_JSONs_schema_and_refs_independently_verified':True,'all_strict_shared_class_mappings_recomputed':True,
        'deterministic_RLE_samples_match_image_dimensions':True,'network_budget_accounted_for_this_run':True,
        'all_source_models_offsets_and_inputs_unchanged':True,'failed_source_retained_as_failure':True,'image_and_confirmation_readiness_not_claimed':True}})
    print('DONE external annotation independent audit',success)


if __name__=='__main__':main()
