import unittest
from semantic.coco_cohort import select_cohort
class CohortCases(unittest.TestCase):
 def test_two_roles_quota_disjoint_reproducible(self):
  images=[dict(id=i)for i in range(50)];anns=[dict(image_id=i,category_id=2,iscrowd=0,area=1)for i in range(40)]+[dict(image_id=i,category_id=3,iscrowd=0,area=1)for i in range(30,50)]
  data=dict(images=images,annotations=anns);m=select_cohort(data,{2:2,3:3},count=8,quota=2)
  a={im['id']for im in m['candidate']};b={im['id']for im in m['reserve']};self.assertEqual(len(a),8);self.assertEqual(len(b),8);self.assertFalse(a&b)
  self.assertEqual(m,select_cohort(dict(images=list(reversed(images)),annotations=list(reversed(anns))),{2:2,3:3},count=8,quota=2))
  for role in ('candidate','reserve'):self.assertTrue(all(n>=2 for n in m['role_class_image_counts'][role].values()))
if __name__=='__main__':unittest.main()
