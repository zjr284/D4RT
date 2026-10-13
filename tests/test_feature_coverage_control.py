import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from semantic.feature_coverage_control import group_manifest, contrast

m = {'training_sets': {'a': [{'video_id': 1}], 'b': [{'video_id': 2}]}, 'val': [{'video_id': 3}], 'confirmation': [{'video_id': 4}, {'video_id': 5}]}
assert group_manifest(m, 'a') == {'train': [{'video_id': 1}], 'val': m['val'], 'confirmation': m['confirmation']}
def run(value):
 return {'calibrated_confirmation': {k: value for k in ('macro_f1_present', 'foreground_macro_f1_present', 'foreground_accuracy')}, 'per_video': {'4': {'macro_f1_present': value}, '5': {'macro_f1_present': value}}}
r = contrast('new', 'old', {'new': {'runs': [run(.4), run(.5), run(.6)]}, 'old': {'runs': [run(.3), run(.4), run(.5)]}}, m)
assert np.allclose(r['macro_f1_present']['paired_differences'], .1)
assert abs(r['video_bootstrap']['mean_per_video_difference']-.1) < 1e-12
assert r['video_bootstrap']['positive_videos'] == 2 and len(r['per_video_differences']) == 2
print('Group isolation and paired seed/video contrasts passed')
