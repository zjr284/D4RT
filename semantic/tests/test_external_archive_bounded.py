import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from semantic import external_archive_bounded as m


class Response(io.BytesIO):
    status=206
    def __init__(self,data,version='fixed'):
        super().__init__(data)
        self.headers={'Content-Range':'bytes 0-3/10','Content-Length':'4','Last-Modified':version}


class Opener:
    def __init__(self,responses):self.responses=iter(responses);self.calls=0
    def open(self,request,timeout):
        self.calls+=1
        assert request.get_header('Accept-encoding')=='identity'
        return next(self.responses)


class BoundedTests(unittest.TestCase):
    def run_ranges(self,opener):return m.ChunkedRanges(opener,'https://example.test/data',10,[None,'fixed'])
    def test_retry_success_counts_failed_body(self):
        with tempfile.TemporaryDirectory()as d,patch.object(m,'ROOT',Path(d)):
            opener=Opener([Response(b'ab'),Response(b'abcd')]);ranges=self.run_ranges(opener)
            self.assertEqual(ranges.read(0,3),b'abcd')
            self.assertEqual(ranges.received,6);self.assertEqual(opener.calls,2)
            self.assertEqual(ranges.failures[0]['reason'],'early_eof')
    def test_retry_limit_preserves_all_failures(self):
        with tempfile.TemporaryDirectory()as d,patch.object(m,'ROOT',Path(d)):
            opener=Opener([Response(b'a')for _ in range(3)]);ranges=self.run_ranges(opener)
            with self.assertRaises(m.ShortResponse):ranges.read(0,3)
            self.assertEqual(opener.calls,3);self.assertEqual(ranges.received,3)
            self.assertEqual(len(ranges.failures),3)
    def test_version_failure_never_retries(self):
        with tempfile.TemporaryDirectory()as d,patch.object(m,'ROOT',Path(d)):
            opener=Opener([Response(b'abcd','changed')]);ranges=self.run_ranges(opener)
            with self.assertRaises(ValueError):ranges.read(0,3)
            self.assertEqual(opener.calls,1);self.assertEqual(ranges.received,0)
            self.assertFalse(ranges.failures[0]['retryable'])


if __name__=='__main__':unittest.main()
