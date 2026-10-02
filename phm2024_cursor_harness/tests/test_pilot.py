import json
from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from operational.core import Predictor, sha, operational_input, labelled_join, evaluate_shadow
from phm.data import RAW

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'pilot_artifacts/sample_demo'


class PilotContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from operational.runtime import check_baseline_environment
        try: check_baseline_environment(BASE)
        except ValueError as exc: raise unittest.SkipTest(f'Bundled sample runtime required: {exc}')
        cls.predictor = Predictor(BASE)
        cls.x = pd.read_csv(ROOT/'data_demo/challenge_holdout_X.csv').head(12)

    def test_frozen_predictions_equal_baseline(self):
        before = sha(BASE/'baseline.joblib')
        p,mu,sigma = self.predictor.bundle['calibrator'].predict(self.predictor.bundle['model'],self.x)
        out = self.predictor.predict(self.x)
        np.testing.assert_allclose(out.p_faulty,p,rtol=0,atol=0)
        np.testing.assert_allclose(out.margin_mean,mu,rtol=0,atol=0)
        np.testing.assert_allclose(out.margin_sigma,sigma,rtol=0,atol=0)
        self.assertEqual(before,sha(BASE/'baseline.joblib'))
        self.assertTrue((out.margin_high_90>out.margin_low_90).all())

    def test_invalid_rows_do_not_receive_predictions(self):
        x=self.x.copy()
        x.loc[0,'ng']=0
        x.loc[1,'mgt']=np.nan
        out=self.predictor.predict(x)
        self.assertTrue(out.loc[:1,'p_faulty'].isna().all())
        self.assertTrue(out.loc[:1,'reliability_status'].eq('rejected').all())
        self.assertTrue(out.loc[:1,'review_recommended'].all())
        self.assertTrue(out.loc[2:,'p_faulty'].notna().all())

    def test_duplicate_observation_id_rejected(self):
        x=self.x.copy();x.loc[1,'id']=x.loc[0,'id']
        with self.assertRaises(ValueError): self.predictor.predict(x)

    def test_extra_label_columns_cannot_change_prediction(self):
        a=self.predictor.predict(self.x)
        x=self.x.assign(faulty=1,trq_margin=1e9,inspection_result='confirmed')
        b=self.predictor.predict(x)
        np.testing.assert_array_equal(a.p_faulty,b.p_faulty)
        np.testing.assert_array_equal(a.margin_mean,b.margin_mean)

    def test_joint_novelty_inside_every_univariate_range(self):
        # Diagonal training support: marginal ranges alone cannot detect a joint gap.
        train=pd.DataFrame({'a':np.linspace(0,10,100),'b':np.linspace(0,10,100)})
        scaler=RobustScaler().fit(train)
        nn=NearestNeighbors(n_neighbors=1).fit(scaler.transform(train))
        point=pd.DataFrame({'a':[1.],'b':[9.]})
        self.assertTrue(((point.iloc[0]>=train.min())&(point.iloc[0]<=train.max())).all())
        self.assertGreater(nn.kneighbors(scaler.transform(point))[0][0,0],1.)

    def test_seen_measurements_excluded_from_metrics(self):
        guard=self.predictor.guard['spaces']['joint']
        # Reference is fitting rows and stored as robust coordinates.
        original=guard['scaler'].inverse_transform(guard['nn']._fit_X[:2])
        x=pd.DataFrame(original,columns=RAW).assign(id=['fit-a','fit-b'])
        # Float inverse-transform may round; use original source IDs for exact fingerprints.
        source=pd.read_csv(ROOT/'data_demo/fitting_rows_contract_test.csv')
        x=source[source.id.isin(guard['reference_ids'][:2])].copy()
        out=self.predictor.predict(x).assign(faulty=0,trq_margin=0)
        report=evaluate_shadow(out,self.predictor.manifest['threshold'])
        self.assertEqual(report['seen_duplicate_labelled'],len(out))
        self.assertEqual(report['evaluated_independent_feature_rows'],0)
        self.assertIsNone(report['metrics'])

    def test_partial_labels_and_invalid_rows_counted(self):
        out=self.predictor.predict(self.x.head(3))
        out['faulty']=[0,1,np.nan];out['trq_margin']=[1.,-1.,np.nan]
        out.loc[1,'valid_input']=False
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            report=evaluate_shadow(out,self.predictor.manifest['threshold'])
        self.assertEqual(report['labelled'],2)
        self.assertEqual(report['invalid'],1)
        self.assertEqual(report['evaluated_independent_feature_rows'],1)
        json.dumps(report,allow_nan=False)

    def test_intake_requires_confirmed_contract_and_temporal_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            template=json.loads((ROOT/'operational/templates/data_contract.json').read_text())
            c=tmp/'contract.json';c.write_text(json.dumps(template))
            with self.assertRaisesRegex(ValueError,'confirmed'): operational_input('unused.csv',c,self.predictor)
            template.update(status='confirmed',confirmed_by='test owner',confirmed_at='2026-01-01T00:00:00Z',baseline_sha256=self.predictor.manifest['baseline_sha256'],baseline_physics=self.predictor.manifest['physics'])
            for raw in RAW:
                template['measurements'][raw].update(meaning='test baseline measurement',source_unit='test-unit',baseline_unit='C' if raw in ['oat','mgt'] else 'test-unit')
            c.write_text(json.dumps(template))
            x=self.x.head(2).drop(columns='id').assign(observation_id=['a','b'],engine_id='E01',flight_id='F01',timestamp_utc='2026-01-01T00:00:00Z',sensor_quality='ok')
            file=tmp/'observations.csv';x.to_csv(file,index=False)
            self.assertEqual(len(operational_input(file,c,self.predictor)),2)
            x.timestamp_utc='2026-01-01 00:00:00';x.to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError,'offset'): operational_input(file,c,self.predictor)

    def test_labels_preceding_observation_rejected(self):
        out=self.predictor.predict(self.x.head(1)).assign(timestamp_utc='2026-01-02T00:00:00Z')
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'labels.csv'
            pd.DataFrame({'observation_id':out.id,'faulty':[1],'trq_margin':[0.], 'label_source':['inspection'],'label_timestamp_utc':['2026-01-01T00:00:00Z']}).to_csv(file,index=False)
            with self.assertRaisesRegex(ValueError,'precedes'): labelled_join(out,file,'shadow')

    def test_artifact_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            for name in ['manifest.json','baseline.joblib','reliability.joblib','policy.json']:
                shutil.copy(BASE/name,Path(tmp)/name)
            with (Path(tmp)/'baseline.joblib').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'): Predictor(tmp)


if __name__=='__main__': unittest.main()
