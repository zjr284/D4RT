import copy,itertools,unittest
import numpy as np
from pycocotools.coco import COCO
from semantic.coco_semantic_adapter import semantic_maps
class PolygonRLECases(unittest.TestCase):
 def test_polygon_unknown_crowd_uncompressed_RLE_and_IDzero(self):
  crowd=np.zeros((8,8),np.uint8);crowd[4:6,:2]=1;flat=crowd.ravel(order='F');counts=[len(list(g))for _,g in itertools.groupby(flat)];assert flat[0]==0
  anns=[dict(id=0,image_id=1,category_id=3,iscrowd=0,segmentation=[[0,0,4,0,4,4,0,4]]),dict(id=1,image_id=1,category_id=4,iscrowd=0,segmentation=[[2,0,6,0,6,2,2,2]]),dict(id=2,image_id=1,category_id=99,iscrowd=0,segmentation=[[0,2,2,2,2,4,0,4]]),dict(id=3,image_id=1,category_id=3,iscrowd=1,segmentation=dict(size=[8,8],counts=counts))]
  original=copy.deepcopy(anns);api=COCO();image=dict(id=1,width=8,height=8);api.dataset=dict(images=[image],annotations=anns,categories=[dict(id=3),dict(id=4),dict(id=99)]);api.createIndex();maps=semantic_maps(api,image,{3:3,4:4},8)
  self.assertEqual(anns,original)
  self.assertEqual([(maps['strict']==c).sum()for c in (3,4,-1,0)],[8,4,12,40]);self.assertEqual([(maps['consensus']==c).sum()for c in (3,4,-1,0)],[12,4,8,40])
  self.assertTrue(np.array_equal(maps['strict'][maps['strict']>0],maps['consensus'][maps['strict']>0]))
if __name__=='__main__':unittest.main()
