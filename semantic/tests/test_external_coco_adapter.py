import unittest
import numpy as np
from pycocotools import mask as coco_mask
from semantic.external_coco_adapter import groups,label_map,queries


class AdapterTests(unittest.TestCase):
    def test_numeric_order_and_fixed_pilot_exclusion(self):
        images=[{'id':i,'file_name':f'00001/{n}.jpg','height':4,'width':4}for i,n in enumerate((10,7,6,5,4,3,2,1))]
        pilot=[dict(im,id=im['id']+20,file_name=im['file_name'].replace('00001/','00000/'))for im in images]
        clips,excluded=groups(images+pilot)
        self.assertEqual([im['file_name']for im in clips[0]['frames']],['00001/1.jpg','00001/2.jpg','00001/3.jpg','00001/4.jpg','00001/5.jpg','00001/6.jpg','00001/7.jpg','00001/10.jpg'])
        self.assertEqual(excluded[0]['reason'],'fixed_pilot')
    def test_unknown_overlap_and_crowd_are_ignored(self):
        def ann(mask,category,crowd=0):return {'segmentation':coco_mask.encode(np.asfortranarray(mask,dtype=np.uint8)),'category_id':category,'iscrowd':crowd}
        a=np.zeros((4,4),np.uint8);a[:2,:2]=1
        b=np.zeros_like(a);b[1:3,1:3]=1
        c=np.zeros_like(a);c[3,3]=1
        entries=[ann(a,1),ann(b,99),ann(c,1,1)]
        result=label_map(entries,{1:3},4,4,size=4)
        self.assertEqual(result[0,0],3);self.assertEqual(result[1,1],-1)
        self.assertEqual(result[2,2],-1);self.assertEqual(result[3,3],-1);self.assertEqual(result[0,3],0)
        np.testing.assert_array_equal(result,label_map(entries[::-1],{1:3},4,4,size=4))
    def test_empty_and_small_foreground_queries(self):
        mask=np.zeros((128,128),np.int16)
        xy,y=queries(mask,12,7);self.assertEqual(xy.shape,(0,3));self.assertEqual(y.shape,(0,))
        mask[3,9]=4;xy,y=queries(mask,12,7)
        self.assertEqual(xy.shape,(256,3));self.assertTrue(np.all(xy==[9,3,7]));self.assertTrue(np.all(y==4))
    def test_invalid_path_is_rejected(self):
        with self.assertRaises(ValueError):groups([{'id':0,'file_name':'../00000.jpg','height':4,'width':4}])


if __name__=='__main__':unittest.main()
