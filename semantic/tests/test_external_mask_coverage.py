import unittest
import numpy as np
from pycocotools import mask as coco
from semantic.external_mask_coverage import audit_frame

def ann(aid,cid,mask,crowd=0):
 r=coco.encode(np.asfortranarray(mask.astype(np.uint8)));return dict(id=aid,category_id=cid,iscrowd=crowd,segmentation=r,area=int(mask.sum()))
class CoverageCases(unittest.TestCase):
 def test_disjoint_loss_causes_and_zero_id(self):
  masks=[]
  for places in ([0,1,2,3,4,5,6,7],[1],[2],[3],[4],[5]):
   m=np.zeros((4,4),bool);m.ravel()[places]=True;masks.append(m)
  rows=[ann(i,c,m,cr)for i,c,m,cr in zip(range(6),(3,3,4,99,3,1),masks,(0,0,0,0,1,0))]
  out=audit_frame(rows,{1:1,3:3,4:4},{1:'Person',3:'Cat',4:'Dog',99:'unknown'},4,4,4)
  c=next(x for x in out['classes']if x['class_id']==3)
  self.assertEqual([c[k]for k in ('candidate_128_pixels','kept_unique_pixels','unknown_or_crowd_overlap_pixels','different_known_overlap_pixels','same_class_only_overlap_pixels')],[8,3,2,2,1])
  self.assertEqual(out['annotations'][0]['annotation_id'],0)
 def test_native_duplicate_versus_sampling_vanish(self):
  m=np.zeros((256,256),bool);m[0,0]=1
  out=audit_frame([ann(0,3,m),ann(9,99,m)],{3:3},{3:'Cat',99:'unknown'},256,256)
  self.assertEqual(out['classes'][0]['native_nonempty_raster_empty_instances'],1)
  self.assertEqual(out['classes'][0]['candidate_128_pixels'],0)
  self.assertEqual(len(out['exact_native_geometry_duplicates']),1)
  self.assertEqual([r['annotation_id']for r in out['exact_native_geometry_duplicates'][0]['annotations']],[0,9])
if __name__=='__main__':unittest.main()
