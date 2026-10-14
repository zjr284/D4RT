"""Repeated short reads must complete exactly or preserve consumed byte count."""
import io
import unittest
from semantic.external_archive_retry import read_exact


class ShortStream:
    def __init__(self,data):self.stream=io.BytesIO(data)
    def read(self,n):return self.stream.read(min(n,3))


class ExactReadTests(unittest.TestCase):
    def test_repeated_short_reads_complete(self):
        consumed=[];self.assertEqual(read_exact(ShortStream(b'abcdefghij'),10,consumed.append),b'abcdefghij')
        self.assertEqual(sum(consumed),10)

    def test_early_eof_rejects_and_preserves_count(self):
        consumed=[]
        with self.assertRaises(ValueError):read_exact(ShortStream(b'abc'),5,consumed.append)
        self.assertEqual(sum(consumed),3)

    def test_extra_bytes_reject_and_are_counted(self):
        consumed=[]
        with self.assertRaises(ValueError):read_exact(ShortStream(b'abcd'),3,consumed.append)
        self.assertEqual(sum(consumed),4)


if __name__=='__main__':unittest.main()
