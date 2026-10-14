"""Ensure external unknown classes remain ignored instead of background."""
import unittest
import numpy as np
from pycocotools import mask as coco_mask
from semantic.external_holdout_readiness import category_mapping,mapped_label_map


class ExternalMappingTests(unittest.TestCase):
    def test_case_and_format_match_without_semantic_merges(self):
        target=[{'id':1,'name':'Person'},{'id':2,'name':'Bird'},{'id':13,'name':'Giant_panda'}]
        source=[{'id':45,'name':'person'},{'id':7,'name':'giant panda'},{'id':90,'name':'eagle'},{'id':8,'name':'owl'}]
        self.assertEqual(category_mapping(source,target),{45:1,7:13})

    def test_unknown_and_overlapping_regions_are_not_background(self):
        a=np.zeros((128,128),np.uint8);a[10:70,10:70]=1
        b=np.zeros_like(a);b[60:110,60:110]=1
        anns=[{'category_id':45,'segmentations':[coco_mask.encode(np.asfortranarray(a))]},
              {'category_id':90,'segmentations':[coco_mask.encode(np.asfortranarray(b))]}]
        raster=mapped_label_map(anns,0,{45:1},128,128)
        self.assertEqual(int(raster[20,20]),1)
        self.assertEqual(int(raster[100,100]),-1)
        self.assertEqual(int(raster[65,65]),-1)
        self.assertEqual(int(raster[0,0]),0)

    def test_ambiguous_target_names_and_bad_dimensions_fail(self):
        with self.assertRaises(ValueError):category_mapping([],[{'id':1,'name':'Bird'},{'id':2,'name':'bird'}])
        a=np.zeros((64,64),np.uint8);a[1:4,1:4]=1
        with self.assertRaises(ValueError):mapped_label_map([{'category_id':1,'segmentations':[coco_mask.encode(np.asfortranarray(a))]}],0,{1:1},128,128)


if __name__=='__main__':unittest.main()
