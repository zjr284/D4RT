"""Exercise annotation ID0, background and ambiguous point association."""
import unittest
from types import SimpleNamespace
import numpy as np
from pycocotools import mask as coco_mask
from semantic.dino_instance_audit import associate


class InstanceAssociationTests(unittest.TestCase):
    def setUp(self):
        a=np.zeros((128,128),np.uint8);a[10:75,10:75]=1
        b=np.zeros_like(a);b[60:110,60:110]=1
        self.ds=SimpleNamespace(annotations={3:[
            {'id':0,'category_id':1,'segmentations':[coco_mask.encode(np.asfortranarray(a))]*8},
            {'id':7,'category_id':2,'segmentations':[coco_mask.encode(np.asfortranarray(b))]*8}]})
        self.clip={'video_id':3,'start':0}

    def test_zero_id_is_foreground_and_background_is_minus_one(self):
        xy=np.array([[20,20,0],[100,100,1],[0,0,2]],np.int16)
        np.testing.assert_array_equal(associate(self.ds,self.clip,xy,np.array([1,2,0])),[0,7,-1])

    def test_overlap_cannot_be_assigned_foreground_or_background(self):
        for label in (0,1,2):
            with self.subTest(label=label),self.assertRaises(AssertionError):
                associate(self.ds,self.clip,np.array([[65,65,0]]),np.array([label]))

    def test_wrong_category_is_rejected(self):
        with self.assertRaises(AssertionError):
            associate(self.ds,self.clip,np.array([[20,20,0]]),np.array([2]))


if __name__=='__main__':unittest.main()
