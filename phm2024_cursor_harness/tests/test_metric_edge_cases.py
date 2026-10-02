import unittest
import warnings
import numpy as np
import pandas as pd
from phm.evaluate import metrics


class MetricEdgeCases(unittest.TestCase):
    def test_single_observation_has_no_r2_and_fixed_confusion_shape(self):
        y=pd.DataFrame({'faulty':[0],'trq_margin':[1.]})
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            m=metrics(y,np.array([.1]),np.array([1.1]),np.array([1.]),.5)
        self.assertIsNone(m['regression']['r2'])
        self.assertEqual(m['classification']['confusion_matrix'],[[1,0],[0,0]])
        self.assertEqual(m['classification']['balanced_accuracy'],1.)
        self.assertIsNone(m['classification']['roc_auc'])

    def test_single_truth_class_counts_false_alerts_without_warning(self):
        y=pd.DataFrame({'faulty':[0,0],'trq_margin':[1.,2.]})
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            m=metrics(y,np.array([.1,.8]),np.array([1.1,1.9]),np.array([1.,1.]),.5)
        self.assertEqual(m['classification']['confusion_matrix'],[[1,1],[0,0]])
        self.assertEqual(m['classification']['false_positives'],1)
        self.assertEqual(m['classification']['balanced_accuracy'],.5)


if __name__=='__main__':unittest.main()
