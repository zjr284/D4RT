import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import numpy as np
from semantic.dino_sam_control import read, contrast

m = {'training_sets': {'random80': []}, 'val': [], 'confirmation': [{'video_id': 4}]}
sample = {'decoded': np.arange(12).reshape(3, 4), 'labels': np.array([1, 0, 1]),
          'xy': np.array([[2, 3, 0], [4, 5, 0], [6, 7, 1]]), 't_tgt': np.array([1, 2, 3]),
          't_cam': np.array([4, 5, 6]), 'video_id': np.array(1)}
with patch('semantic.dino_sam_control.read_features', return_value=[{k: v.copy() for k, v in sample.items()}]):
 gt = read('random80_source_gt', 'train', m)[0]
 assert len(gt['labels']) == 2 and np.all(gt['xy'][:, 2] == 0)
 for k in ('decoded', 'labels', 'xy', 't_tgt', 't_cam'): assert np.array_equal(gt[k], sample[k][:2])
with patch('semantic.dino_sam_control.read_features', return_value=[{k: v.copy() for k, v in sample.items()}]):
 sam = read('random80_sam_background', 'train', m)[0]
 assert np.array_equal(sam['decoded'], sample['decoded'])
def run(x): return {'calibrated_confirmation': {k: x for k in ('macro_f1_present', 'foreground_macro_f1_present', 'foreground_accuracy')}, 'per_video': {'4': {'macro_f1_present': x}}}
r = contrast('sam', 'gt', {'sam': {'runs': [run(.4)]*3}, 'gt': {'runs': [run(.5)]*3}}, m)
assert np.allclose(r['macro_f1_present']['paired_differences'], -.1)
assert r['video_bootstrap']['negative_videos'] == 1
print('Source-GT isolation, SAM full-time retention and negative contrasts passed')
