"""Static scientific plot of already verified paired video differences."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semantic.oracle_probe import file_hash
from semantic import external_reserve_influence_audit as work

def main():
 s=json.loads((work.ROOT/'summary.json').read_text());v=json.loads((work.ROOT/'verification.json').read_text());assert file_hash(work.ROOT/'summary.json')==v['summary_sha256']
 out=Path('reports/analysis');out.mkdir(parents=True,exist_ok=True);groups=list(s['group_annotation_and_query_coverage']);fig,axes=plt.subplots(1,2,figsize=(12,12),sharex=True);lo=[];hi=[]
 for ax,policy in zip(axes,('strict','consensus')):
  result=s['results'][policy]['comparisons']['native_sam_minus_source_gt'];per=result['per_group_three_seed_differences'];labels=[]
  for i,g in enumerate(groups):
   values=per[g];labels.append(g+(' *'if values is None else''))
   if values is None:continue
   a=np.array(values)*100;mean=a.mean();lo.append(a.min());hi.append(a.max());color='#16836a'if mean>0 else'#c04c46';ax.barh(i,mean,color=color,alpha=.75);ax.plot([a.min(),a.max()],[i,i],color='#333333',lw=.8);ax.scatter([mean],[i],color='#222222',s=8,zorder=3)
  ax.set_yticks(range(len(groups)),labels,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.18);ax.set_xlabel('SAM - source GT, F1 percentage points');ax.set_title(('Strict: Person only'if policy=='strict'else'Sensitivity: Person + Bird in one group')+'\n'+f"Mean across valid groups = {result['influence']['full_video_equal_mean']*100:+.2f} pp",fontsize=11)
 for ax in axes:ax.set_xlim(min(lo)-5,max(hi)+5)
 fig.suptitle('Original reserve40: all groups retained',fontsize=14);fig.text(.5,.017,'Bars: mean of 3 training seeds. Lines: seed range, NOT confidence intervals.\n* No valid queries; no score assigned. These directory groups are not verified object tracks.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.055,1,.95));paths=[]
 for ext in ('png','pdf'):
  path=out/('reserve40_sam_video_differences_20261020.'+ext);fig.savefig(path,dpi=160);paths.append(path)
 plt.close(fig);p=work.ROOT/'plot_provenance.json';p.write_text(json.dumps({'source_summary_sha256':v['summary_sha256'],'script_sha256':file_hash(__file__),'outputs':{str(x):file_hash(x)for x in paths}},indent=2)+'\n');print('saved',*(str(x)for x in paths),flush=True)
if __name__=='__main__':main()
