import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from semantic.feature_fusion_control import join_features
rng=np.random.default_rng(8484)
b={'decoded':rng.normal(size=(3,768)).astype(np.float32),'labels':np.array([1,0,2]),'xy':np.array([[1,2,0],[3,4,0],[5,6,0]]),'video_id':np.array(12)}
e={**b,'decoded':rng.normal(size=(3,256)).astype(np.float32)}
zero=join_features(b,None);weight=rng.normal(size=(26,1024)).astype(np.float64)
assert np.array_equal(zero['decoded'].astype(np.float64)@weight.T,b['decoded'].astype(np.float64)@weight[:,:768].T)
for key in ('labels','xy','video_id'):
 bad={**e,key:e[key]+1}
 try:join_features(b,bad)
 except ValueError:pass
 else:raise AssertionError('Misaligned '+key+' accepted')
bad={**e,'decoded':e['decoded'].astype(np.float16)}
try:join_features(b,bad)
except ValueError:pass
else:raise AssertionError('Silent re-quantization accepted')
assert join_features(b,e)['decoded'].shape==(3,1024)
print('PASS matched-width zero-feature identity and provenance mismatch rejection')
