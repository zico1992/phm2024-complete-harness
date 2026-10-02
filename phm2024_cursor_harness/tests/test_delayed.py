from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from pilot import execute
from operational.core import sha
from operational.delayed import delayed_evaluate,verify_delayed
from operational.runtime import check_baseline_environment

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'pilot_artifacts/sample_demo'


class DelayedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:check_baseline_environment(BASE)
        except ValueError as exc:raise unittest.SkipTest(f'Bundled sample runtime required: {exc}')

    def test_partial_then_cumulative_labels_preserve_source_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'predictions'
            execute(SimpleNamespace(command='replay',baseline=BASE,input=ROOT/'data_demo/challenge_holdout_X.csv',labels=None,output=source))
            before={p.name:sha(p) for p in source.iterdir() if p.is_file()}
            labels=pd.read_csv(ROOT/'data_demo/challenge_holdout_y.csv')
            labels.head(10).to_csv(tmp/'partial.csv',index=False)
            labels.to_csv(tmp/'cumulative.csv',index=False)
            with patch('operational.runtime.joblib.load',side_effect=AssertionError('delayed labels must not load model')):
                one=delayed_evaluate(BASE,source,tmp/'partial.csv',tmp/'one')
                self.assertEqual(one['labelled'],10)
                self.assertEqual(verify_delayed(BASE,tmp/'one')['status'],'passed')
                two=delayed_evaluate(BASE,source,tmp/'cumulative.csv',tmp/'two')
                self.assertEqual(two['labelled'],2999)
                self.assertEqual(verify_delayed(BASE,tmp/'two')['status'],'passed')
            self.assertEqual(before,{p.name:sha(p) for p in source.iterdir() if p.is_file()})
            self.assertEqual(sha(source/'predictions.csv'),sha(tmp/'one/predictions.csv'))
            self.assertEqual(sha(source/'predictions.csv'),sha(tmp/'two/predictions.csv'))

    def test_operational_shadow_can_save_before_labels_arrive(self):
        from operational.core import Predictor,now
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);predictor=Predictor(BASE)
            x=pd.read_csv(ROOT/'data_demo/challenge_holdout_X.csv').head(8)
            y=pd.read_csv(ROOT/'data_demo/challenge_holdout_y.csv')
            xy=x.merge(y,on='id',validate='one_to_one')
            observations=xy.drop(columns=['faulty','trq_margin']).assign(observation_id=lambda d:d.id.astype(str),engine_id='UNIT-TEST-ENGINE',flight_id='UNIT-TEST-FLIGHT',timestamp_utc='2026-01-01T00:00:00Z',sensor_quality='ok').drop(columns='id')
            observations.to_csv(tmp/'observations.csv',index=False)
            contract=json.loads((ROOT/'operational/templates/data_contract.json').read_text())
            contract.update(status='confirmed',confirmed_by='unit test fixture only',confirmed_at='2026-01-01T00:00:00Z',baseline_sha256=predictor.manifest['baseline_sha256'],baseline_physics=predictor.manifest['physics'])
            for c,rule in contract['measurements'].items():
                rule.update(meaning='unit test mapping',source_unit='C' if c in ['oat','mgt'] else 'fixture',baseline_unit='C' if c in ['oat','mgt'] else 'fixture')
            (tmp/'contract.json').write_text(json.dumps(contract))
            result=execute(SimpleNamespace(command='shadow',baseline=BASE,input=tmp/'observations.csv',contract=tmp/'contract.json',labels=None,output=tmp/'source'))
            self.assertEqual(result['labelled'],0)
            labels=xy[['id','faulty','trq_margin']].rename(columns={'id':'observation_id'}).assign(label_source='unit test, not operational evidence',label_timestamp_utc=now())
            labels.to_csv(tmp/'labels.csv',index=False)
            with patch('operational.runtime.joblib.load',side_effect=AssertionError('no model reload')):
                delayed_evaluate(BASE,tmp/'source',tmp/'labels.csv',tmp/'evaluation')
                self.assertEqual(verify_delayed(BASE,tmp/'evaluation')['status'],'passed')
            report=json.loads((tmp/'evaluation/evaluation.json').read_text())
            self.assertEqual(report['label_timing']['labelled_after_prediction_commit'],8)

    def test_tampered_predictions_and_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'predictions'
            execute(SimpleNamespace(command='replay',baseline=BASE,input=ROOT/'data_demo/challenge_holdout_X.csv',labels=None,output=source))
            labels=ROOT/'data_demo/challenge_holdout_y.csv'
            delayed_evaluate(BASE,source,labels,tmp/'evaluation')
            with (tmp/'evaluation/labels.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'[Ll]abel.*changed'):verify_delayed(BASE,tmp/'evaluation')
            with (source/'predictions.csv').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'prediction hash'):delayed_evaluate(BASE,source,labels,tmp/'other')
            self.assertFalse((tmp/'other').exists())
