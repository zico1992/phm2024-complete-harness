import unittest
import pandas as pd
from operational.review import review_queue


class ReviewPriorityContracts(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'id':['signal-gap','signal','invalid','support','routine'],'fault_flag':[1,1,None,0,0],
            'valid_input':[True,True,False,True,True],'reliability_status':['review','within_reference','rejected','review','within_reference'],
            'review_recommended':[True,True,True,True,False],'p_faulty':[.9,.8,None,.01,.01]})

    def test_priority_uses_predictions_and_support_not_outcomes(self):
        d=self.frame();before=review_queue(d,{})
        after=review_queue(d.assign(faulty=[0,0,1,1,1],trq_margin=-999),{})
        self.assertEqual(list(before.review_priority),['P1','P2','P2','P3','P4'])
        self.assertEqual(list(before.review_priority),list(after.review_priority))

    def test_completed_review_closes_queue_but_does_not_change_signal(self):
        d=self.frame();out=review_queue(d,{'signal':{'disposition':'review_complete','reviewer':'engineer','at_utc':'2026-01-01T00:00:00Z','note':'test'}})
        self.assertFalse(out.loc[1,'open_review'])
        self.assertEqual(out.loc[1,'fault_flag'],1)
        self.assertEqual(out.loc[1,'review_state'],'complete')
        self.assertEqual(out.loc[1,'latest_reviewer'],'engineer')
        self.assertTrue(out.loc[0,'open_review'])
