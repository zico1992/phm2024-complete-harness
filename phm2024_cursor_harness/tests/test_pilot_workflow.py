"""Temporary synthetic metadata tests; never operational validation evidence."""
from pathlib import Path
from types import SimpleNamespace
import json
import socket
import sqlite3
from unittest.mock import patch
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import pandas as pd
from pilot import execute
from operational.core import Predictor,sha
from operational.server import review_database
from operational.delayed import delayed_evaluate
from operational.review import review_location, latest_reviews
from verify_pilot import verify

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'pilot_artifacts/sample_demo'


class DatabaseContracts(unittest.TestCase):
    def test_database_connections_close_after_commit_and_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'reviews.sqlite'
            connect=sqlite3.connect
            opened=[]
            def tracked_connect(*args,**kwargs):
                db=connect(*args,**kwargs);opened.append(db);return db
            with patch('operational.server.sqlite3.connect',side_effect=tracked_connect):
                with review_database(file) as db:
                    db.execute('CREATE TABLE events (value INTEGER)')
                    db.execute('INSERT INTO events VALUES (1)')
                with self.assertRaisesRegex(RuntimeError,'rollback fixture'):
                    with review_database(file) as db:
                        db.execute('INSERT INTO events VALUES (2)')
                        raise RuntimeError('rollback fixture')
                with review_database(file) as db:
                    self.assertEqual(db.execute('SELECT value FROM events').fetchall(),[(1,)])
            # Keep references alive: closure must be explicit, not left to GC.
            for db in opened:
                with self.assertRaises(sqlite3.ProgrammingError): db.execute('SELECT 1')
            moved=file.with_name('released.sqlite')
            file.rename(moved);moved.unlink()


class WorkflowContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from operational.runtime import check_baseline_environment
        try: check_baseline_environment(BASE)
        except ValueError as exc: raise unittest.SkipTest(f'Bundled sample runtime required: {exc}')

    def test_complete_labelled_shadow_on_temporary_contract_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);predictor=Predictor(BASE)
            x=pd.read_csv(ROOT/'data_demo/challenge_holdout_X.csv').head(8)
            y=pd.read_csv(ROOT/'data_demo/challenge_holdout_y.csv')
            xy=x.merge(y,on='id',validate='one_to_one')
            observations=xy.drop(columns=['faulty','trq_margin']).assign(observation_id=lambda d:d.id.astype(str),engine_id='SYNTHETIC-TEST-ENGINE',flight_id='SYNTHETIC-TEST-FLIGHT',timestamp_utc='2026-01-01T00:00:00Z',sensor_quality='ok').drop(columns='id')
            labels=xy[['id','faulty','trq_margin']].rename(columns={'id':'observation_id'}).assign(label_source='unit-test fixture, not operational',label_timestamp_utc='2026-01-02T00:00:00Z')
            observations.to_csv(tmp/'observations.csv',index=False);labels.to_csv(tmp/'labels.csv',index=False)
            contract=json.loads((ROOT/'operational/templates/data_contract.json').read_text())
            contract.update(status='confirmed',confirmed_by='unit test only',confirmed_at='2026-01-01T00:00:00Z',baseline_sha256=predictor.manifest['baseline_sha256'],baseline_physics=predictor.manifest['physics'])
            for c,rule in contract['measurements'].items():
                rule.update(meaning='synthetic test of baseline mapping',source_unit='C' if c in ['oat','mgt'] else 'fixture',baseline_unit='C' if c in ['oat','mgt'] else 'fixture')
            (tmp/'contract.json').write_text(json.dumps(contract))
            args=SimpleNamespace(command='shadow',baseline=str(BASE),input=str(tmp/'observations.csv'),contract=str(tmp/'contract.json'),labels=str(tmp/'labels.csv'),output=str(tmp/'shadow'))
            result=execute(args)
            self.assertEqual(result['mode'],'shadow');self.assertEqual(result['labelled'],8)
            v=verify(BASE,tmp/'shadow',tmp/'observations.csv',tmp/'contract.json',tmp/'labels.csv')
            self.assertEqual(v['status'],'passed')
            ledger=json.loads((tmp/'shadow/run_manifest.json').read_text())
            self.assertFalse(ledger['labels_used_for_predictions'])
            self.assertEqual(sha(BASE/'baseline.joblib'),predictor.manifest['baseline_sha256'])

    def test_local_api_predictions_reviews_and_session_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            execute(SimpleNamespace(command='replay',baseline=str(BASE),input=str(ROOT/'data_demo/challenge_holdout_X.csv'),labels=None,output=str(tmp/'replay')))
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            proc=subprocess.Popen([sys.executable,str(ROOT/'pilot.py'),'serve','--baseline',str(BASE),'--run-dir',str(tmp/'replay'),'--port',str(port)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            try:
                url=f'http://127.0.0.1:{port}'
                deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    try:
                        with urlopen(url+'/api/summary',timeout=.5) as r:summary=json.load(r)
                        break
                    except OSError:
                        if proc.poll() is not None: self.fail(proc.stderr.read().decode())
                        time.sleep(.05)
                else:self.fail('server did not start')
                with urlopen(url+'/api/observations?filter=fn') as r:
                    page=json.load(r)
                self.assertEqual(len(page['rows']),0)  # no labels means no known FN
                with urlopen(url) as r:
                    self.assertIn('Review queue',r.read().decode())
                    self.assertIn("script-src 'self'",r.headers['Content-Security-Policy'])
                sample=pd.read_csv(ROOT/'data_demo/challenge_holdout_X.csv').head(1).to_dict('records')[0]
                def post(path,body,token=summary['session_token'],origin=url):
                    req=Request(url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Session-Token':token,'Origin':origin})
                    try:
                        with urlopen(req) as r:return json.load(r)
                    except HTTPError as exc:
                        exc.close()
                        raise
                result=post('/api/predict',{'observations':[sample]})
                self.assertIsNotNone(result['predictions'][0]['p_faulty'])
                review={'id':str(sample['id']),'reviewer':'contract test','disposition':'monitor','note':'Temporary test, not an engineering decision.'}
                self.assertEqual(post('/api/review',review)['status'],'review recorded')
                review['disposition']='review_complete'
                post('/api/review',review)
                with urlopen(url+'/api/observation?id='+review['id']) as r:
                    item=json.load(r)['rows'][0]
                self.assertEqual(item['latest_disposition'],'review_complete')
                self.assertFalse(item['open_review'])
                with urlopen(url+'/api/reviews') as r:self.assertEqual(len(json.load(r)['rows']),2)
                with self.assertRaises(HTTPError) as context:post('/api/review',review,token='wrong')
                self.assertEqual(context.exception.code,403)
                with self.assertRaises(HTTPError) as context:post('/api/review',review,origin='https://unrelated.example')
                self.assertEqual(context.exception.code,403)
                with self.assertRaises(HTTPError) as context:post('/api/predict',{'observations':[{**sample,'faulty':1}]})
                self.assertEqual(context.exception.code,400)
                # Windows refuses this rename if a server connection still holds the file.
                database=tmp/'replay/reviews.sqlite'
                moved=database.with_name('released.sqlite')
                database.rename(moved);moved.rename(database)
                # A label snapshot shares the source prediction run's review history.
                delayed_evaluate(BASE,tmp/'replay',ROOT/'data_demo/challenge_holdout_y.csv',tmp/'later')
                ledger=json.loads((tmp/'later/run_manifest.json').read_text())
                dbpath,_=review_location(tmp/'later',ledger)
                self.assertEqual(dbpath.resolve(),database.resolve())
                with review_database(dbpath) as db:
                    latest=latest_reviews(db,ledger['baseline']['baseline_sha256'],ledger['prediction_sha256'])
                self.assertEqual(latest[review['id']]['disposition'],'review_complete')
            finally:
                try:
                    if proc.poll() is None: proc.terminate()
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill();proc.wait(timeout=5)
                finally:
                    proc.stderr.close()


if __name__=='__main__':unittest.main()
