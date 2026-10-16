import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from semantic import external_rgb_screen as m


class BlindRGBTests(unittest.TestCase):
    def test_fixed_roles_are_disjoint_and_do_not_depend_on_category(self):
        clips=[{'group':str(i),'category':i%3}for i in range(833)]
        a=m.fixed_manifest(clips);b=m.fixed_manifest([dict(c,category=99)for c in clips])
        for role in ('candidate_confirmation','provisional_reserve'):
            self.assertEqual([c['group']for c in a[role]],[c['group']for c in b[role]])
            self.assertEqual(len(a[role]),40)
        self.assertEqual(len({c['group']for role in ('candidate_confirmation','provisional_reserve')for c in a[role]}),80)
    def test_restart_keeps_consumption_and_prefix_history(self):
        with tempfile.TemporaryDirectory()as d,patch.object(m,'ROOT',Path(d)):
            state={'received_bytes':9,'records':[{'start':0,'end':3}],
                'attempts':[{'outcome':'incomplete_prefix'}],'forms':[{'bytes':5}]}
            (Path(d)/'transfer_progress.json').write_text(json.dumps(state))
            ranges=m.ChunkedRanges(None,'',10,[None,'fixed'])
            self.assertEqual(ranges.received,9);self.assertEqual(ranges.records,state['records'])
            self.assertEqual(ranges.forms,state['forms']);self.assertEqual(len(ranges.failures),1)


if __name__=='__main__':unittest.main()
