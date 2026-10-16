"""Independent per-image annotation indexing for class-coverage summaries."""
from collections import defaultdict
import json
from pathlib import PurePosixPath
from semantic.external_category_feasibility import ROOT,ADAPTER,ANN,OVIS
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json


def main():
    s=json.loads((ROOT/'summary.json').read_text());p=s['protocol']
    assert p==json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    payload=json.loads(ANN.read_bytes());ovis=json.loads(OVIS.read_bytes())
    normalize=lambda name:' '.join(name.lower().replace('_',' ').split())
    target={normalize(c['name']):c['id']for c in ovis['categories']};names={c['id']:c['name']for c in ovis['categories']}
    mapping={c['id']:target[normalize(c['name'])]for c in payload['categories']if normalize(c['name'])in target}
    by_image=defaultdict(list)
    for ann in payload['annotations']:by_image[ann['image_id']].append(ann)
    all_images={im['id']:im for im in payload['images']};clips=json.loads((ADAPTER/'groups.json').read_text())['groups']
    selected={im['id']for clip in clips for im in clip['frames']};assert len(selected)==len(clips)*8
    for key,image_ids in [('all_images',set(all_images)),('fixed_eight_frame_clips',selected)]:
        expected=s[key];rows=defaultdict(list);crowd=0;nonpositive=0
        for iid in image_ids:
            for ann in by_image[iid]:
                if ann['category_id']not in mapping:continue
                if ann.get('iscrowd',0):crowd+=1;continue
                if ann.get('area',0)<=0:nonpositive+=1;continue
                rows[mapping[ann['category_id']]].append(iid)
        per_class={str(cid):{'name':names[cid],'annotation_rows':len(rows[cid]),'images':len(set(rows[cid])),
            'groups':len({str(PurePosixPath(all_images[iid]['file_name']).parent)for iid in rows[cid]})}for cid in sorted(mapping.values())}
        assert per_class==expected['per_class']
        positive_images={iid for ids in rows.values()for iid in ids}
        fields={'observed_common_classes':sum(bool(ids)for ids in rows.values()),'common_annotation_rows':sum(len(ids)for ids in rows.values()),
            'images_with_common_annotations':len(positive_images),'groups_with_common_annotations':len({str(PurePosixPath(all_images[iid]['file_name']).parent)for iid in positive_images}),
            'total_images':len(image_ids),'total_groups':len({str(PurePosixPath(all_images[iid]['file_name']).parent)for iid in image_ids}),
            'ignored_common_crowd_rows':crowd,'nonpositive_or_missing_area_common_rows':nonpositive}
        assert all(expected[k]==v for k,v in fields.items())
    assert s['strict_shared_vocabulary_classes']==len(mapping)and s['vocabulary_categories']==len(payload['categories'])
    assert s['actually_annotated_external_categories']==len({a['category_id']for a in payload['annotations']})
    assert s['no_model_evaluation']and s['selection_unchanged']and not s['formal_confirmation_frozen']
    atomic_json(ROOT/'verification.json',{'summary_sha256':file_hash(ROOT/'summary.json'),'replayed_classes':len(mapping),'replayed_fixed_frames':len(selected),
        'verification':{'all_inputs_unchanged':True,'mapping_and_zero_counts_recomputed':True,'all_and_fixed_frame_scopes_recomputed':True,
        'crowd_area_and_unique_group_counts_recomputed':True,'no_class_or_score_based_reselection':True}})
    print('DONE independent category coverage audit',flush=True)


if __name__=='__main__':main()
