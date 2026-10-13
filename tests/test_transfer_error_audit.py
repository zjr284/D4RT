import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from semantic.transfer_error_audit import describe, mean_description, group_classes, subset

y = np.array([0, 1, 1, 2, 2, 3]); pred = np.array([1, 1, 0, 1, 2, 0]); vids = np.array([10, 10, 11, 11, 12, 12])
r = describe(y, pred, vids)
assert r['per_class']['1']['queries'] == 2 and r['per_class']['1']['independent_videos'] == 2
assert r['per_class']['1']['f1'] == 0.4  # False positives from background and class 2 retained.
e = r['error_partition']; assert e['foreground_correct_count'] + e['foreground_to_background_count'] + e['foreground_wrong_class_count'] == 5
assert mean_description([r, r])['per_class']['1']['recall'] == 0.5
train = {str(c): {'queries': 1, 'recall': .5} for c in range(1, 5)}
confirm = {str(c): {'queries': 1, 'recall': .05} for c in range(1, 5)}
train['1']['queries'] = 0; train['2']['recall'] = .49; confirm['4']['recall'] = .1
assert group_classes(train, confirm) == {'unsupported_in_training': [1], 'low_training_fit': [2], 'fitted_training_low_confirmation': [3], 'other': [4]}
try: subset({'labels': y}, np.zeros(len(y), dtype=bool), 0)
except ValueError: pass
else: raise AssertionError('Empty subset must fail')
print('Full-subset false positives, error closure, averaging and fixed class-group boundaries passed')
