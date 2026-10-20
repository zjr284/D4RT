"""Independently replay frozen predictions, deterministic selection and RLE point hits.

Does not import the case preparation implementation or run a model.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from semantic import external_reserve_confirmation as source
from semantic.run import atomic_json

ROOT = Path('semantic_artifacts/external_reserve_case_audit')
KINDS = ('corrected', 'regressed', 'persistent_error', 'both_correct', 'mixed_seed')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def check_inputs(protocol):
    for path, expected in protocol['input_sha256'].items():
        assert digest(path) == expected, path


def main():
    protocol, selection, materials = [read(n) for n in ('protocol.json', 'selection.json', 'materials.json')]
    summary, review = read('summary.json'), read('visual_review.json')
    assert protocol['round'] == 97
    check_inputs(protocol)
    old = json.loads((source.ROOT / 'summary.json').read_text())
    assert digest(source.ROOT / 'summary.json') == protocol['source_summary_sha256'] == selection['source_summary_sha256']
    for path, expected in old['output_hashes'].items():
        if '/queries/' in path or '/logits/' in path:
            assert digest(path) == expected, path
    clips = json.loads((source.ROOT / 'manifest.json').read_text())['confirmation']
    annotation = json.loads(source.ANN.read_text())
    images = {im['id']: im for im in annotation['images']}
    annotations = defaultdict(list)
    for ann in annotation['annotations']:
        annotations[ann['image_id']].append(ann)
    category_names = {c['id']: c['name'] for c in annotation['categories']}
    mapping = {int(k): int(v) for k, v in json.loads(source.old.MAPPING_SUMMARY.read_text())['shared_category_mapping'].items()}
    ovis_names = {'0': 'Background', **{str(c['id']): c['name'] for c in json.loads(source.old.OVIS.read_text())['categories']}}
    assert selection['class_names'] == ovis_names
    pools = {kind: [] for kind in KINDS}
    per_group = {}
    bird_frames = defaultdict(list)
    expected = {}
    replayed_outputs = 0

    def add_minimum(pool, reason):
        if pool:
            chosen = sorted(pool, key=lambda item: item['selection_sha256'])[0]
            key = (chosen['policy'], chosen['group'], chosen['slot'], chosen['u'], chosen['v'])
            if key not in expected:
                expected[key] = dict(chosen, selection_reasons=[])
            expected[key]['selection_reasons'].append(reason)

    replay = {}
    for policy in ('strict', 'consensus'):
        preds = {}
        ids, target = None, None
        for arm in ('source_gt', 'native_sam', 'B14_raw'):
            seed_rows = []
            for seed in range(3):
                run = old['results'][policy]['arms'][arm]['runs'][seed]
                with np.load(run['output_path']) as data:
                    scores = data['logits'].copy()
                    yi, gi = data['labels'].copy(), data['group_ids'].copy()
                if ids is None:
                    ids, target = gi, yi
                assert np.array_equal(ids, gi) and np.array_equal(target, yi)
                scores[:, 0] = scores[:, 0] + old['protocol']['models'][arm]['offset']
                seed_rows.append(np.argmax(scores, axis=1))
                replayed_outputs += 1
            preds[arm] = np.asarray(seed_rows)
        replay[policy] = (preds, ids, target)

    for clip in clips:
        group = clip['group']
        group_pool = []
        for policy in ('strict', 'consensus'):
            with np.load(source.qp(policy, group)) as data:
                xyz, y = data['xy'].copy(), data['labels'].copy()
            preds, ids, target = replay[policy]
            positions = np.where(ids == int(group))[0]
            assert np.array_equal(y, target[positions]) and len(xyz) == len(y)
            # Correct-seed counts define the five classes, without using work.types.
            ng = np.sum(preds['source_gt'][:, positions] == y, axis=0)
            ns = np.sum(preds['native_sam'][:, positions] == y, axis=0)
            kinds = []
            for a, b in zip(ng, ns):
                kinds.append({(0, 3): 'corrected', (3, 0): 'regressed',
                              (0, 0): 'persistent_error', (3, 3): 'both_correct'}.get((a, b), 'mixed_seed'))
            if policy == 'strict':
                per_group[group] = {kind: kinds.count(kind) for kind in KINDS}
            first = {}
            for j, coordinates in enumerate(xyz):
                first.setdefault(tuple(map(int, coordinates)), j)
            for (u, v, slot), j in first.items():
                im = clip['frames'][slot]
                assert images[im['id']] == im
                pixel_string = '|'.join(map(str, (policy, group, im['id'], u, v)))
                r = {'policy': policy, 'group': group, 'slot': slot, 'u': u, 'v': v,
                     'image_id': im['id'], 'path': str(source.old.RGB / 'images' / im['file_name']),
                     'GT': int(y[j]), 'type': kinds[j],
                     'predictions': {arm: preds[arm][:, positions[j]].astype(int).tolist() for arm in preds},
                     'selection_sha256': hashlib.sha256(pixel_string.encode()).hexdigest()}
                if policy == 'strict':
                    group_pool.append(r)
                    pools[kinds[j]].append(r)
                elif group == '00821' and int(y[j]) == 2:
                    bird_frames[slot].append(r)
        errors = list(filter(lambda item: item['type'] != 'both_correct', group_pool))
        add_minimum(errors if errors else group_pool, 'strict_group_error_or_all_correct')
    for kind in KINDS:
        add_minimum(pools[kind], 'strict_global_' + kind)
    for slot in range(8):
        add_minimum(bird_frames[slot], 'consensus_Bird_frame_' + str(slot))
    assert per_group == selection['all40_group_strict_counts'] and len(per_group) == 40
    assert len(expected) == selection['card_count'] == len(selection['selected']) == 35
    assert [tuple(r[k] for k in ('policy', 'group', 'slot', 'u', 'v')) for r in selection['selected']] == list(expected)
    native_hits = 0
    for index, (key, r) in enumerate(expected.items()):
        saved = selection['selected'][index]
        for field, value in r.items():
            assert value == saved[field], (index, field)
        im = images[r['image_id']]
        w, h = im['width'], im['height']
        with Image.open(r['path']) as image:
            assert image.size == (w, h)
        # Integer arithmetic checks the actual nearest-neighbour source pixel.
        px = min(w - 1, ((2 * r['u'] + 1) * w) // 256)
        py = min(h - 1, ((2 * r['v'] + 1) * h) // 256)
        center = [(2 * r['u'] + 1) * w / 256 - .5, (2 * r['v'] + 1) * h / 256 - .5]
        assert saved['case_id'] == index and saved['original_pixel'] == [px, py]
        assert saved['original_center'] == center and (saved['width'], saved['height']) == (w, h)
        hits = []
        for ann in annotations[r['image_id']]:
            mask = mask_api.decode(ann['segmentation'])
            assert mask.shape == (h, w)
            if int(mask[py, px]) == 1:
                hits.append({'ann_id': ann['id'], 'category_id': ann['category_id'],
                             'name': category_names[ann['category_id']], 'mapped': mapping.get(ann['category_id']),
                             'crowd': bool(ann.get('iscrowd', 0))})
        assert hits == saved['original_point_hits']
        native_hits += len(hits)
        strict = -1
        if not hits:
            strict = 0
        elif len(hits) == 1 and not hits[0]['crowd'] and hits[0]['mapped'] is not None:
            strict = hits[0]['mapped']
        eligible = [hit['mapped'] for hit in hits if hit['mapped'] is not None and not hit['crowd']]
        consensus = -1
        if not hits:
            consensus = 0
        elif not any(hit['crowd'] for hit in hits) and eligible and len(set(eligible)) == 1:
            consensus = eligible[0]
        assert (strict, consensus) == (saved['native_point_strict_label'], saved['native_point_consensus_label'])
        assert r['GT'] == {'strict': strict, 'consensus': consensus}[r['policy']]
        assert digest(r['path']) == saved['image_sha256']
    assert materials['selection_sha256'] == digest(ROOT / 'selection.json')
    all_panels = []
    for sheet in materials['sheets']:
        assert digest(sheet['path']) == sheet['sha256']
        with Image.open(sheet['path']) as image:
            image.verify()
        for panel in sheet['panels']:
            r = selection['selected'][panel['case_id']]
            x, y = r['original_center']
            radius = max(16, min(r['width'], r['height']) // 10)
            box = [max(0, int(x) - radius), max(0, int(y) - radius),
                   min(r['width'], int(x) + radius), min(r['height'], int(y) + radius)]
            assert box == panel['crop_box'] and panel['image_sha256'] == r['image_sha256']
            all_panels.append(panel['case_id'])
    assert all_panels == list(range(35)) and len(materials['sheets']) == 6
    assert review['materials_sha256'] == digest(ROOT / 'materials.json')
    assert review['sheets_viewed_sha256'] == [s['sha256'] for s in materials['sheets']]
    assert [r['case_id'] for r in review['observations']] == list(range(35))
    # This is a completeness/hash check, not independent certification of visual truth.
    assert review['reviewer'] == 'one Codex visual reviewer; not a human expert or independent GT'
    for path, expected_hash in summary['evidence_sha256'].items():
        assert digest(path) == expected_hash
    counts = {k: sum(row[k] for row in per_group.values()) for k in KINDS}
    assert counts == summary['strict_all_query_type_counts'] and sum(counts.values()) == 46848
    assert summary['selected_type_counts'] == dict(Counter(r['type'] for r in selection['selected']))
    assert summary['selected_policy_counts'] == dict(Counter(r['policy'] for r in selection['selected']))
    assert summary['visually_suspect_annotation_case_ids'] == [21]
    assert summary['new_model_inferences'] == 0 and summary['labels_or_scores_modified'] is False
    check_inputs(protocol)
    atomic_json(ROOT / 'verification.json', {
        'round': 97, 'summary_sha256': digest(ROOT / 'summary.json'),
        'verifier_sha256': digest('semantic/verify_external_reserve_case_audit.py'),
        'frozen_outputs_replayed': replayed_outputs,
        'all40_group_query_counts_and_deterministic_case_selection_recomputed': True,
        'cases_predictions_coordinates_and_native_RLE_hits_verified': 35,
        'native_instance_hits_checked': native_hits,
        'all320_RGB_and_old_query_logit_input_hashes_unchanged': True,
        'six_sheet_hashes_crop_metadata_and_case_coverage_verified': True,
        'visual_review_completeness_and_hash_checked_not_visual_truth_certification': True,
        'suspect_case21_image_id_filename_and_annotation_link_checked': True,
        'labels_and_old_scores_unchanged': True})
    print('DONE independent R97 verification: 18 outputs, 40 groups, 35 cases, 6 sheets', flush=True)


if __name__ == '__main__':
    main()
