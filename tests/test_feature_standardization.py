import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from semantic.feature_standardization_control import fit_statistics,transform

def main():
    train=np.array([[1.,2.,7.],[3.,4.,7.],[5.,6.,7.]],np.float16)
    mean,scale,std=fit_statistics(train)
    assert mean.dtype==scale.dtype==np.float32 and np.array_equal(mean,[3.,4.,7.])
    assert std[2]==0 and scale[2]==1
    z=transform(train,mean,scale)
    assert z.dtype==np.float32 and np.allclose(z.mean(0),0,atol=1e-7)
    assert np.allclose(z[:,:2].std(0),1,atol=1e-7)
    val=np.array([[100.,-50.,9.]],np.float16)
    fixed=transform(val,mean,scale)
    assert np.array_equal(mean,[3.,4.,7.]) and fixed[0,2]==2
    weight=np.array([[.3,.4,.5],[-.2,.1,.7]],np.float32)
    bias=np.array([.2,-.3],np.float32)
    raw_weight=weight/scale
    raw_bias=bias-raw_weight@mean
    assert np.allclose(fixed@weight.T+bias,val.astype(np.float32)@raw_weight.T+raw_bias,atol=2e-6)
    print('PASS train-only population moments, constant channel, fixed validation transform and affine logit identity')
if __name__=='__main__':
    main()
