import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from semantic.source_gt_control import source_clips
c={'decoded':np.arange(12).reshape(3,4),'labels':np.array([1,2,0]),'xy':np.array([[0,1,0],[2,3,1],[4,5,0]]),'video_id':np.array(7)}
result=source_clips([c])[0]
assert result['labels'].tolist()==[1,0] and (result['xy'][:,2]==0).all()
assert result['decoded'].tolist()==[[0,1,2,3],[8,9,10,11]]
result['decoded'][0,0]=999
assert c['decoded'][0,0]==0
try:source_clips([dict(c,xy=np.array([[0,1,1],[2,3,1],[4,5,2]]))])
except ValueError:pass
else:raise AssertionError('Missing source GT must fail')
print('Source-only selection, feature preservation and isolation tests passed')

# Calibration metadata must survive safe weights-only reload.
import json,tempfile,torch
from semantic.source_gt_control import PROTOCOL
assert all(type(v) is float for v in PROTOCOL['calibration_grid'])
with tempfile.TemporaryDirectory() as folder:
    checkpoint=Path(folder)/'model.pt'
    torch.save({'model':torch.nn.Linear(4,3).state_dict(),'protocol':json.loads(json.dumps(PROTOCOL))},checkpoint)
    restored=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert restored['protocol']==PROTOCOL
print('Checkpoint metadata safely reloads')
