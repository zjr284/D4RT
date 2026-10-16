"""Vocabulary intersection is distinct from actual validation instance coverage."""
from collections import Counter,defaultdict
import json
from pathlib import Path,PurePosixPath
from semantic.external_coco_adapter import ANN,OVIS
from semantic.external_holdout_readiness import category_mapping
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

ROOT=Path('semantic_artifacts/external_category_feasibility')
ADAPTER=Path('semantic_artifacts/external_coco_adapter')


def counts(annotations,images,mapping,ovis_names):
    selected=set(images);group_of={k:str(PurePosixPath(v['file_name']).parent)for k,v in images.items()}
    row_counts=Counter();image_ids=defaultdict(set);groups=defaultdict(set);crowd=0;nonpositive=0
    for ann in annotations:
        if ann['image_id']not in selected or ann['category_id']not in mapping:continue
        if ann.get('iscrowd',0):crowd+=1;continue
        if ann.get('area',0)<=0:nonpositive+=1;continue
        cid=mapping[ann['category_id']];row_counts[cid]+=1;image_ids[cid].add(ann['image_id']);groups[cid].add(group_of[ann['image_id']])
    return {'per_class':{str(cid):{'name':ovis_names[cid],'annotation_rows':row_counts[cid],'images':len(image_ids[cid]),'groups':len(groups[cid])}for cid in sorted(mapping.values())},
        'observed_common_classes':sum(bool(row_counts[c])for c in mapping.values()),'common_annotation_rows':sum(row_counts.values()),
        'images_with_common_annotations':len(set().union(*image_ids.values()))if image_ids else 0,
        'groups_with_common_annotations':len(set().union(*groups.values()))if groups else 0,
        'total_images':len(images),'total_groups':len(set(group_of.values())),
        'ignored_common_crowd_rows':crowd,'nonpositive_or_missing_area_common_rows':nonpositive}


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'summary.json').exists():raise RuntimeError('Completed category feasibility exists')
    previous=Path('semantic_artifacts/ovis_frame_index');v=json.loads((previous/'verification.json').read_text());assert file_hash(previous/'summary.json')==v['summary_sha256']
    paths=[ANN,OVIS,ADAPTER/'groups.json',ADAPTER/'summary.json',ADAPTER/'verification.json',previous/'summary.json',previous/'verification.json',Path('semantic/external_category_feasibility.py'),Path('semantic/OVIS_EXTERNAL_CATEGORY_FEASIBILITY_PROTOCOL.md')]
    p={'round':43,'no_image_or_model_access':True,'input_sha256':{str(path):file_hash(path)for path in paths}}
    atomic_json(ROOT/'protocol.json',p)
    payload=json.loads(ANN.read_bytes());ovis=json.loads(OVIS.read_bytes());mapping=category_mapping(payload['categories'],ovis['categories']);names={c['id']:c['name']for c in ovis['categories']}
    all_images={i['id']:i for i in payload['images']};clips=json.loads((ADAPTER/'groups.json').read_text())['groups']
    clip_images={i['id']:i for clip in clips for i in clip['frames']};assert len(clip_images)==8*len(clips)
    all_counts=counts(payload['annotations'],all_images,mapping,names);clip_counts=counts(payload['annotations'],clip_images,mapping,names)
    for path,h in p['input_sha256'].items():assert file_hash(path)==h
    atomic_json(ROOT/'summary.json',{'protocol':p,'vocabulary_categories':len(payload['categories']),
        'actually_annotated_external_categories':len({a['category_id']for a in payload['annotations']}),
        'strict_shared_vocabulary_classes':len(mapping),'all_images':all_counts,'fixed_eight_frame_clips':clip_counts,
        'selection_unchanged':True,'formal_confirmation_frozen':False,'no_model_evaluation':True,'deployment_unchanged':'native_gt'})
    print('DONE external category feasibility',all_counts['observed_common_classes'],clip_counts['observed_common_classes'],flush=True)


if __name__=='__main__':main()
