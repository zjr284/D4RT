import tempfile
import unittest
from pathlib import Path
import zipfile
from semantic.external_archive_download import audit_zip


class FullArchiveTests(unittest.TestCase):
    def create(self,path):
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED)as z:z.writestr('val/a.jpg',b'ABC')
        with zipfile.ZipFile(path)as z:
            return {i.filename:{'crc32':i.CRC,'compressed_bytes':i.compress_size,'uncompressed_bytes':i.file_size,'local_offset':i.header_offset,'method':i.compress_type}for i in z.infolist()}
    def test_full_member_crc_and_metadata(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'sample.zip';expected=self.create(path)
            self.assertEqual(audit_zip(path,expected),1)
            expected['val/a.jpg']['crc32']+=1
            with self.assertRaisesRegex(ValueError,'directory_mismatch'):audit_zip(path,expected)
    def test_corrupted_member_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'sample.zip';expected=self.create(path);path.write_bytes(path.read_bytes().replace(b'ABC',b'ABD',1))
            with self.assertRaises(zipfile.BadZipFile):audit_zip(path,expected)


if __name__=='__main__':unittest.main()
