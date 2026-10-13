import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic.fresh_video_confirmation import select_clips,bootstrap_video_differences
class Fake:
    videos={i:{'file_names':[f'{i}/{t}.jpg' for t in range(8)]} for i in range(100)}
    annotations={i:[{'segmentations':[{}]*8}] for i in range(100)}
    def frame_path(self,name):return Path(name)
d=Fake();a,n=select_clips(d,{0,1,2},count=40);b,_=select_clips(d,{0,1,2},count=40)
assert a==b and n==97 and len({c['video_id'] for c in a})==40
assert not {c['video_id'] for c in a}&{0,1,2}
try:select_clips(d,set(range(100)),count=40)
except ValueError:pass
else:raise AssertionError('Insufficient eligible videos must fail')
x=bootstrap_video_differences([1.,1.,1.]);assert x['percentile_95_interval']==[1.,1.]
y=bootstrap_video_differences([-1.,0.,1.]);assert y==bootstrap_video_differences([-1.,0.,1.])
assert y['positive_videos']==1 and y['negative_videos']==1 and y['tied_videos']==1
print('Fresh selection isolation, determinism and video bootstrap tests passed')
