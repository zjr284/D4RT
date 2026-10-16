import io
import unittest
from PIL import Image
import numpy as np
from semantic.ovis_frame_index import fingerprint,safe_path


def encode(image,level=6):
    out=io.BytesIO();image.save(out,format='PNG',compress_level=level);return out.getvalue()


class FrameIndexTests(unittest.TestCase):
    def test_equal_pixels_different_encoding(self):
        image=Image.fromarray(np.arange(600,dtype=np.uint8).reshape(10,20,3))
        a=fingerprint(encode(image,0));b=fingerprint(encode(image,9))
        self.assertNotEqual(a['file_sha256'],b['file_sha256']);self.assertEqual(a['pixel_sha256'],b['pixel_sha256'])
        self.assertEqual(a['dhash64'],b['dhash64'])
    def test_shape_prefix_changes_pixel_digest(self):
        values=np.arange(18,dtype=np.uint8)
        a=fingerprint(encode(Image.fromarray(values.reshape(2,3,3))))
        b=fingerprint(encode(Image.fromarray(values.reshape(3,2,3))))
        self.assertNotEqual(a['pixel_sha256'],b['pixel_sha256'])
    def test_dhash_bit_direction(self):
        gradient=np.tile(np.arange(64,dtype=np.uint8)*4,(64,1))
        self.assertEqual(fingerprint(encode(Image.fromarray(gradient)))['dhash64'],'ffffffffffffffff')
        self.assertEqual(fingerprint(encode(Image.new('RGB',(64,64))))['dhash64'],'0000000000000000')
    def test_unsafe_paths_fail(self):
        for name in ('../x.jpg','/tmp/x.jpg','a/b/c.jpg','a\\b.jpg'):
            with self.assertRaises(ValueError):safe_path(name)


if __name__=='__main__':unittest.main()
