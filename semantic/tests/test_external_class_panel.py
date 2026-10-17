import unittest
from semantic.external_class_panel import select_panel

class PanelSelectionTests(unittest.TestCase):
    def test_excluded_person_and_nonpositive_never_select(self):
        clips=[{'group':str(i),'frames':[{'id':i}]}for i in range(6)]
        cats=[{'id':10,'name':'person'},{'id':20,'name':'bird'},{'id':30,'name':'cat'}]
        ovis=[{'id':1,'name':'Person'},{'id':2,'name':'Bird'},{'id':3,'name':'Cat'}]
        anns=[{'image_id':0,'category_id':20,'area':2},{'image_id':1,'category_id':10,'area':8},
            {'image_id':2,'category_id':20,'area':0},{'image_id':3,'category_id':20,'area':8,'iscrowd':1},
            {'image_id':4,'category_id':20,'area':4},{'image_id':5,'category_id':30,'area':3}]
        result,_=select_panel(clips,{'0'},anns,cats,ovis)
        self.assertEqual(result['selection_by_class']['2']['selected_group'],'4')
        self.assertEqual(result['selection_by_class']['3']['selected_group'],'5')
        self.assertNotIn('1',result['selection_by_class'])
        self.assertEqual([c['group']for c in result['panel']],['4','5'])

if __name__=='__main__':unittest.main()
