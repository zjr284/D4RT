import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from semantic import external_archive_resume as m


class Response(io.BytesIO):
    status=206
    def __init__(self,data,start,end):
        super().__init__(data);self.headers={'Content-Range':f'bytes {start}-{end}/10','Content-Length':str(end-start+1),'Last-Modified':'fixed'}


class Opener:
    def __init__(self,parts):self.parts=iter(parts);self.requests=[]
    def open(self,request,timeout):
        start,end=map(int,request.get_header('Range').split('=')[1].split('-'));self.requests.append((start,end))
        return Response(next(self.parts),start,end)


class ResumeTests(unittest.TestCase):
    def test_suffix_starts_after_actual_prefix(self):
        with tempfile.TemporaryDirectory()as d,patch.object(m,'ROOT',Path(d)):
            opener=Opener([b'ab',b'cd']);r=m.ChunkedRanges(opener,'https://example.test',10,[None,'fixed'])
            self.assertEqual(r.read(0,3),b'abcd');self.assertEqual(opener.requests,[(0,3),(2,3)])
            self.assertEqual(r.received,4);self.assertEqual([a['outcome']for a in r.attempts],['incomplete_prefix','complete_response'])
    def test_zero_progress_stops(self):
        with tempfile.TemporaryDirectory()as d,patch.object(m,'ROOT',Path(d)):
            opener=Opener([b'']);r=m.ChunkedRanges(opener,'https://example.test',10,[None,'fixed'])
            with self.assertRaisesRegex(ValueError,'zero_progress'):r.read(0,3)
            self.assertEqual(len(opener.requests),1);self.assertEqual(r.records,[])
    def test_segment_limit_preserves_prefix_without_padding(self):
        with tempfile.TemporaryDirectory()as d,patch.object(m,'ROOT',Path(d)):
            opener=Opener([b'a']*4);r=m.ChunkedRanges(opener,'https://example.test',10,[None,'fixed'])
            with self.assertRaisesRegex(ValueError,'segment_limit'):r.read(0,7)
            self.assertEqual(opener.requests,[(0,7),(1,7),(2,7),(3,7)])
            self.assertEqual(r.received,4);self.assertEqual(sum(e['bytes']for e in r.records),4)


if __name__=='__main__':unittest.main()
