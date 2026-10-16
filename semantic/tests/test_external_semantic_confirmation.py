import unittest
from semantic.external_semantic_confirmation import contrast


class ConditionalContrastTests(unittest.TestCase):
    def test_empty_video_is_retained_as_null_not_zero(self):
        def arm(value):return {'runs':[{'calibrated':{'macro_f1_present':value},'per_video':{'a':{'macro_f1_present':value},'b':None}}for _ in range(3)]}
        result=contrast('left','right',{'left':arm(.8),'right':arm(.5)},[{'group':'a'},{'group':'b'}],{'bootstrap_seed':51,'bootstrap_repetitions':20})
        self.assertAlmostEqual(result['per_video_three_seed_mean_differences']['a'],.3)
        self.assertIsNone(result['per_video_three_seed_mean_differences']['b'])
        self.assertEqual(result['conditional_video_bootstrap']['video_count'],1)
        self.assertAlmostEqual(result['conditional_video_bootstrap']['mean_per_video_difference'],.3)


if __name__=='__main__':unittest.main()
