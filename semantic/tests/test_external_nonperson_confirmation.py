import unittest
import numpy as np
from semantic.external_nonperson_confirmation import nonperson_queries

class ClassQueryTests(unittest.TestCase):
    def test_person_unknown_background_excluded_and_classes_balanced(self):
        labels=np.zeros((128,128),np.int16);labels[0,0]=-1;labels[0,1]=1;labels[0,2]=2;labels[1:3,0:4]=19
        xy,y=nonperson_queries(labels,12,7)
        self.assertEqual(xy.shape,(128,3));self.assertEqual(dict(zip(*np.unique(y,return_counts=True))),{2:64,19:64})
        self.assertTrue(np.all(xy[:,2]==7));np.testing.assert_array_equal(labels[xy[:,1],xy[:,0]],y)
        xx,yy=nonperson_queries(labels,12,7);np.testing.assert_array_equal(xy,xx);np.testing.assert_array_equal(y,yy)
    def test_empty_nonperson_frame_is_not_replaced(self):
        xy,y=nonperson_queries(np.ones((128,128),np.int16),5,0)
        self.assertEqual(xy.shape,(0,3));self.assertEqual(y.shape,(0,))

if __name__=='__main__':unittest.main()
