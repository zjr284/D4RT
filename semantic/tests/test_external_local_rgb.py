import unittest
from semantic.external_local_rgb import cross_group_flags


class CrossGroupTests(unittest.TestCase):
    def test_same_group_ignored_and_threshold_is_inclusive(self):
        records=[{'image_id':0,'group':'a','dhash64':'0000000000000000','file_sha256':'x','pixel_sha256':'p'},
            {'image_id':1,'group':'a','dhash64':'0000000000000000','file_sha256':'x','pixel_sha256':'p'},
            {'image_id':2,'group':'b','dhash64':'000000000000003f','file_sha256':'y','pixel_sha256':'q'},
            {'image_id':3,'group':'c','dhash64':'ffffffffffffffff','file_sha256':'z','pixel_sha256':'r'}]
        pairs=cross_group_flags(records)
        self.assertEqual([(p['left_image_id'],p['right_image_id'],p['distance'])for p in pairs],[(0,2,6),(1,2,6)])
        self.assertTrue(all(not p['exact_file']and not p['exact_pixels']for p in pairs))


if __name__=='__main__':unittest.main()
