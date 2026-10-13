import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from semantic.frozen_error_audit import confusion_counts,decompose
labels=np.array([0,0,1,1,2,2]);pred=np.array([0,2,1,0,1,2])
m=confusion_counts(labels,pred,classes=3);r=decompose(m)
assert r['foreground_support']==4 and r['background_support']==2
assert r['foreground_correct_count']==2 and r['foreground_to_background_count']==1 and r['foreground_wrong_class_count']==1
assert r['background_correct_count']==1 and r['background_to_foreground_count']==1
assert r['foreground_correct_rate']+r['foreground_to_background_rate']+r['foreground_wrong_class_rate']==1
for y,p in ((np.array([]),np.array([])),(np.array([-1]),np.array([0]))):
 try:confusion_counts(y,p)
 except ValueError:pass
 else:raise AssertionError('Invalid labels must fail')
print('Confusion partition and denominator tests passed')
