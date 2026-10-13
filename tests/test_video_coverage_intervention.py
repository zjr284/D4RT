import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic.video_coverage_intervention import choose_coverage
clips=[{'video_id':i} for i in range(12)]
classes={i:{1} if i<6 else {2} for i in range(12)}
a=choose_coverage(clips,classes,count=8,minimum=3,num_classes=2)
assert a==choose_coverage(clips,classes,count=8,minimum=3,num_classes=2)
assert len({c['video_id'] for c in a})==8
for cls in (1,2):assert sum(cls in classes[c['video_id']] for c in a)>=3
for kwargs in ({'count':4,'minimum':3},{'count':8,'minimum':7}):
 try:choose_coverage(clips,classes,num_classes=2,**kwargs)
 except ValueError:pass
 else:raise AssertionError('Impossible coverage must fail')
print('Coverage quotas, determinism, unique videos and infeasibility tests passed')

from semantic.head_budget_probe import matched_classes
train={'support':{'0':2,'1':3},'per_class_recall':{'0':1.0,'1':0.5}}
test={'support':{'0':2,'1':1,'9':1},'per_class_recall':{'0':1.0,'1':0.2,'9':0.0}}
r=matched_classes(train,test)
assert r['categories']==[1] and r['test_foreground_classes_missing_from_training']==[9]
assert r['train_macro_recall']==0.5 and r['test_macro_recall']==0.2
print('Unseen training-class diagnostics preserve missing categories')
