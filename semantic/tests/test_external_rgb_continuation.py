import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import urllib.request
import urllib.error
from semantic import external_rgb_continuation as m


class FakeOpener:
    def __init__(self,failures):self.failures=list(failures);self.calls=0
    def open(self,request,timeout):
        self.calls+=1;assert timeout==60
        if self.failures:raise self.failures.pop(0)
        return type('Response',(),{'status':206})()


class HeaderRetryTests(unittest.TestCase):
    def test_two_timeouts_then_success(self):
        with tempfile.TemporaryDirectory()as d,patch.object(m,'ROOT',Path(d)),patch.object(m.time,'sleep'):
            base=FakeOpener([TimeoutError(),TimeoutError()]);opener=m.HeaderRetryOpener(base)
            self.assertEqual(opener.open(urllib.request.Request('https://example.test',headers={'Range':'bytes=0-3'}),20).status,206)
            self.assertEqual(base.calls,3);self.assertEqual([r['outcome']for r in opener.records],['header_timeout','header_timeout','headers_received_after_retry'])
    def test_retry_limit_and_permission_error(self):
        with tempfile.TemporaryDirectory()as d,patch.object(m,'ROOT',Path(d)),patch.object(m.time,'sleep'):
            base=FakeOpener([TimeoutError()]*3);opener=m.HeaderRetryOpener(base)
            with self.assertRaises(TimeoutError):opener.open(urllib.request.Request('https://example.test'),20)
            self.assertEqual(base.calls,3)
            denied=FakeOpener([urllib.error.HTTPError('https://example.test',403,'denied',{},None)])
            with self.assertRaises(urllib.error.HTTPError):m.HeaderRetryOpener(denied).open(urllib.request.Request('https://example.test'),20)
            self.assertEqual(denied.calls,1)


if __name__=='__main__':unittest.main()
