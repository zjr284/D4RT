import unittest
from semantic.external_capacity_panel import select_panel
class SelectionCases(unittest.TestCase):
 def test_metadata_only_quota_exclusion_and_crowd(self):
  clips=[{'group':f'{i:05}','frames':[{'id':i}]}for i in range(60)];anns=[dict(image_id=i,category_id=10,area=1,iscrowd=0)for i in range(30)]
  anns+=[dict(image_id=i,category_id=20,area=1,iscrowd=0)for i in (40,41)]+[dict(image_id=42,category_id=20,area=1,iscrowd=1)]
  cats=[dict(id=10,name='Bird'),dict(id=20,name='Dog')];mapping=[dict(id=2,name='Bird'),dict(id=4,name='Dog')]
  m,_=select_panel(clips,{'00000'},anns,cats,mapping,count=8)
  chosen=[c['group']for c in m['panel']];self.assertEqual(len(chosen),8);self.assertNotIn('00000',chosen);self.assertTrue({'00040','00041'}<=set(chosen));self.assertEqual(m['selection_by_class']['4']['eligible_group_count'],2)
  self.assertEqual(m,select_panel(clips,{'00000'},list(reversed(anns)),cats,mapping,count=8)[0])
if __name__=='__main__':unittest.main()
