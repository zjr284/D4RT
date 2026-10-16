import unittest
from semantic.external_category_feasibility import counts


class CoverageTests(unittest.TestCase):
    def test_vocabulary_is_not_instance_coverage(self):
        images={1:{'file_name':'00001/00000.jpg'},2:{'file_name':'00002/00000.jpg'}}
        annotations=[{'image_id':1,'category_id':100,'area':3},{'image_id':1,'category_id':100,'area':4},
            {'image_id':2,'category_id':101,'area':5,'iscrowd':1},{'image_id':2,'category_id':101,'area':0},
            {'image_id':2,'category_id':999,'area':5}]
        result=counts(annotations,images,{100:1,101:2},{1:'Person',2:'Bird'})
        self.assertEqual(result['observed_common_classes'],1)
        self.assertEqual(result['per_class']['1'],{'name':'Person','annotation_rows':2,'images':1,'groups':1})
        self.assertEqual(result['per_class']['2']['annotation_rows'],0)
        self.assertEqual(result['ignored_common_crowd_rows'],1)
        self.assertEqual(result['nonpositive_or_missing_area_common_rows'],1)
        self.assertEqual(result['total_groups'],2)


if __name__=='__main__':unittest.main()
