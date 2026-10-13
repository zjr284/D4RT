import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from semantic.semantic_feature_control import patch_at_queries

tokens = torch.tensor([[[[1., 2.], [3., 4.]]], [[[11., 12.], [13., 14.]]]])
xy = np.array([[0, 0, 0], [127, 127, 0], [63, 63, 0], [0, 0, 1]])
v = patch_at_queries(tokens, xy).numpy()[:, 0]
assert v[0] == 1 and v[1] == 4 and v[3] == 11
assert abs(v[2] - 2.4765625) < 1e-7
assert np.array_equal(patch_at_queries(tokens, xy[[3, 0]]).numpy()[:, 0], [11, 1])
for bad in [np.array([[128, 0, 0]]), np.array([[0, 0, 2]]), np.array([[-1, 0, 0]])]:
 try: patch_at_queries(tokens, bad)
 except ValueError: pass
 else: raise AssertionError('Out-of-range query must fail')
print('Pixel-center interpolation, edge replication, source-frame selection and coordinate bounds passed')
