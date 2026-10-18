import unittest
import numpy as np
from pycocotools import mask as coco
from semantic.external_label_sensitivity import consensus_map
class ConsensusCases(unittest.TestCase):
 def test_known_consensus_crowd_unknown_and_zero_id(self):
  rows=[]
  for aid,cid,places,crowd in [(0,3,[0,1,2,3,4],0),(1,99,[0,5],0),(2,3,[1],0),(3,4,[2],0),(4,3,[3],1),(5,1,[4],0)]:
   m=np.zeros((4,4),np.uint8);m.ravel()[places]=1;rows.append(dict(id=aid,category_id=cid,iscrowd=crowd,segmentation=coco.encode(np.asfortranarray(m))))
  out=consensus_map(rows,{1:1,3:3,4:4},4,4,4)
  self.assertEqual(out.ravel().tolist(),[3,3,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0])
  self.assertTrue(np.array_equal(out,consensus_map(list(reversed(rows)),{1:1,3:3,4:4},4,4,4)))
if __name__=='__main__':unittest.main()
