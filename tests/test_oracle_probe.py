import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from semantic.oracle_probe import relabel_queries
maps=np.array([[[0,1],[2,-1]],[[3,0],[-1,4]]],dtype=np.int16)
xy=np.array([[1,0,0],[0,0,1],[1,1,0],[0,1,1],[1,1,1]])
assert relabel_queries(xy,maps).tolist()==[1,3,-1,-1,4]
assert (xy==np.array([[1,0,0],[0,0,1],[1,1,0],[0,1,1],[1,1,1]])).all()
for bad in ([[2,0,0]],[[0,0,2]],[[-1,0,0]]):
 try:relabel_queries(np.array(bad),maps)
 except ValueError:pass
 else:raise AssertionError('Out of bounds queries must fail')
print('Oracle query relabel tests passed')
