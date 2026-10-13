import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from semantic.background_calibration import predictions,pick_offset
rng=np.random.default_rng(42);logits=rng.normal(size=(100,26)).astype(np.float32);original=logits.copy()
assert np.array_equal(predictions(logits,0),logits.argmax(axis=1))
tie=np.zeros((1,26));assert predictions(tie,0).tolist()==[0]
p=predictions(logits,-2);q=predictions(logits,2)
assert ((p==0)<=(q==0)).all()
assert np.array_equal(logits,original)
a=np.full((2,26),-10.0);a[:,0]=2;a[:,1]=1
selected,curve=pick_offset([{'logits':a,'labels':np.ones(2,dtype=int)}],[-2.,0.,2.])
assert selected==-2.
selected,_=pick_offset([{'logits':np.zeros((2,26)),'labels':np.zeros(2,dtype=int)}],[-2.,0.,2.])
assert selected==0.
try:predictions(np.full((1,26),np.nan),0)
except ValueError:pass
else:raise AssertionError('Nonfinite logits must fail')
print('Zero-offset, tie, monotonicity, validation selection and immutability tests passed')
