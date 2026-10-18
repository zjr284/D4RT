"""RGB-only overview for a single Codex visual review; no GT or model reads."""
import json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from semantic.coco_fusion_cohort import ROOT
from semantic.oracle_probe import file_hash
from semantic.run import atomic_json

def main():
 path=ROOT/'overview.json'
 if path.exists():raise RuntimeError('Overview exists; inspect rather than regenerate')
 records=json.loads((ROOT/'image_records.json').read_text());assert len(records)==40;fig,axes=plt.subplots(8,5,figsize=(18,22),squeeze=False)
 for axis,r in zip(axes.ravel(),records):
  with Image.open(r['path'])as im:axis.imshow(im.convert('RGB'))
  axis.set_title(str(r['image_id']),fontsize=10);axis.axis('off')
 fig.tight_layout();out=ROOT/'candidate40_overview.png';fig.savefig(out,dpi=110);plt.close(fig);atomic_json(path,{'path':str(out),'sha256':file_hash(out),'image_ids':[r['image_id']for r in records],'script_sha256':file_hash('semantic/coco_fusion_overview.py'),'no_GT_or_model_reads':True});print('DONE RGB-only40 overview',flush=True)
if __name__=='__main__':main()
