"""Local fixtures validate ZIP lookup, traversal rejection and CRC failure."""
import io
import unittest
import zipfile
from semantic.external_archive_probe import RangeZIP


def fixture(names,compression):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=compression)as z:
        for name,data in names:z.writestr(name,data)
    return stream.getvalue()


class RangeZIPTests(unittest.TestCase):
    def test_stored_and_deflate_utf8_members(self):
        for compression in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):
            with self.subTest(compression=compression):
                raw=fixture([('00000/00000.jpg',b'payload'*40),('00001/鸟.jpg',b'second')],compression)
                z=RangeZIP(lambda a,b:raw[a:b+1],len(raw));z.directory()
                self.assertEqual(z.member('00000/00000.jpg'),b'payload'*40)
                self.assertEqual(z.member('00001/鸟.jpg'),b'second')

    def test_path_traversal_is_rejected(self):
        raw=fixture([('../outside.jpg',b'payload')],zipfile.ZIP_STORED)
        with self.assertRaises(ValueError):RangeZIP(lambda a,b:raw[a:b+1],len(raw)).directory()

    def test_bad_crc_is_rejected(self):
        raw=fixture([('00000/00000.jpg',b'payload')],zipfile.ZIP_STORED)
        z=RangeZIP(lambda a,b:raw[a:b+1],len(raw));z.directory()
        z.entries['00000/00000.jpg']['crc32']^=1
        with self.assertRaises(ValueError):z.member('00000/00000.jpg')

    def test_truncated_directory_is_rejected(self):
        raw=fixture([('00000/00000.jpg',b'payload')],zipfile.ZIP_STORED)
        with self.assertRaises(ValueError):RangeZIP(lambda a,b:raw[a:b+1],len(raw)-5).directory()


if __name__=='__main__':unittest.main()
