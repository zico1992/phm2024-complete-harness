"""Loopback-only local review interface. SQLite append-only audit, no external actions."""
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager, closing
import json
import secrets
import sqlite3
import pandas as pd
from phm.data import RAW
from operational.core import Predictor, sha
from operational.review import PRIORITY_POLICY, review_location, latest_reviews, review_queue


@contextmanager
def review_database(path):
    """Commit/rollback the transaction AND release the SQLite file handle."""
    with closing(sqlite3.connect(path)) as db:
        with db:
            yield db


def records(frame):
    return json.loads(frame.to_json(orient='records'))


def serve(run_dir, baseline, port):
    root = Path(run_dir).resolve()
    predictor = Predictor(baseline)
    ledger = json.loads((root/'run_manifest.json').read_text())
    if ledger['baseline']['baseline_sha256'] != predictor.manifest['baseline_sha256']:
        raise ValueError('Run and baseline hashes differ')
    if ledger['prediction_sha256'] != sha(root/'predictions.csv'):
        raise ValueError('Prediction file changed after run')
    # Evaluation-only columns are joined for viewing, never used for new prediction.
    data = pd.read_csv(root/'predictions.csv',dtype={'id':str,'engine_id':str,'flight_id':str})
    evaluated = root/'evaluated_predictions.csv'
    if evaluated.exists():
        if ledger.get('evaluated_predictions_sha256') != sha(evaluated):
            raise ValueError('Evaluated predictions changed')
        e = pd.read_csv(evaluated,dtype={'id':str})
        labels = [c for c in ['faulty','trq_margin','label_source','label_timestamp_utc'] if c in e]
        data = data.merge(e[['id']+labels],on='id',how='left',validate='one_to_one')
    if ledger.get('evaluation_sha256') != sha(root/'evaluation.json'):
        raise ValueError('Evaluation changed')
    evaluation = json.loads((root/'evaluation.json').read_text())
    token = secrets.token_urlsafe(32)
    origin = f'http://127.0.0.1:{port}'
    dbpath,legacy_run_hashes = review_location(root,ledger)
    prediction_scope = ledger['prediction_sha256']
    with review_database(dbpath) as db:
        db.execute('CREATE TABLE IF NOT EXISTS audit (sequence INTEGER PRIMARY KEY, at_utc TEXT, observation_id TEXT, reviewer TEXT, disposition TEXT, note TEXT, baseline_sha256 TEXT, run_sha256 TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS interface_predictions (sequence INTEGER PRIMARY KEY, at_utc TEXT, mode TEXT, payload TEXT, result TEXT, baseline_sha256 TEXT)')
    with review_database(dbpath) as db:
        columns={row[1] for row in db.execute('PRAGMA table_info(audit)')}
        if 'prediction_sha256' not in columns:
            db.execute('ALTER TABLE audit ADD COLUMN prediction_sha256 TEXT')
        for legacy_hash in legacy_run_hashes:
            db.execute('UPDATE audit SET prediction_sha256=? WHERE prediction_sha256 IS NULL AND baseline_sha256=? AND run_sha256=?',
                (prediction_scope,predictor.manifest['baseline_sha256'],legacy_hash))

    def queue():
        with review_database(dbpath) as db:
            latest=latest_reviews(db,predictor.manifest['baseline_sha256'],prediction_scope)
        return review_queue(data,latest)

    runhash = sha(root/'run_manifest.json')
    web = Path(__file__).parent/'web'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def respond(self,code,obj):
            body = json.dumps(obj,allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self):
            return self.headers.get('Host')==f'127.0.0.1:{port}'

        def do_GET(self):
            if not self.host_ok():
                return self.respond(403,{'error':'Use the printed loopback URL'})
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path=='/api/summary':
                q=queue()
                return self.respond(200,{'manifest':ledger,'evaluation':evaluation,'session_token':token,
                    'counts':{'total':len(data),'review':int(data.review_recommended.sum()),
                              'rejected':int((~data.valid_input).sum()),'fault_flags':int((data.fault_flag==1).sum()),
                              'open_review':int(q.open_review.sum()),'unreviewed':int((q.review_recommended & q.review_state.eq('unreviewed')).sum())},
                    'policy':predictor.policy,'priority_policy':PRIORITY_POLICY,'runtime':predictor.runtime})
            if parsed.path=='/api/observations':
                d = queue()
                scope=query.get('scope',['all'])[0]
                if scope=='open':d=d[d.open_review]
                elif scope=='unreviewed':d=d[d.review_recommended & d.review_state.eq('unreviewed')]
                elif scope=='complete':d=d[d.review_state.eq('complete')]
                elif scope!='all':return self.respond(400,{'error':'Unknown review scope'})
                text = query.get('q',[''])[0].strip().lower()
                if text:
                    mask = d.id.str.lower().str.contains(text,regex=False)
                    for col in ['engine_id','flight_id']:
                        if col in d:
                            mask |= d[col].fillna('').str.lower().str.contains(text,regex=False)
                    d = d[mask]
                filter_ = query.get('filter',['all'])[0]
                if filter_=='review': d=d[d.review_recommended]
                if filter_=='rejected': d=d[~d.valid_input]
                if filter_=='fn': d=d[(d.faulty==1)&(d.fault_flag==0)] if 'faulty' in d else d.iloc[:0]
                if filter_=='fp': d=d[(d.faulty==0)&(d.fault_flag==1)] if 'faulty' in d else d.iloc[:0]
                try:
                    offset=max(0,int(query.get('offset',['0'])[0]))
                except ValueError:
                    return self.respond(400,{'error':'invalid offset'})
                sort=query.get('sort',['priority'])[0]
                if sort=='priority':d=d.sort_values(['priority_rank','p_faulty','id'],ascending=[True,False,True],na_position='last',kind='stable')
                elif sort=='id':d=d.sort_values('id',kind='stable')
                else:return self.respond(400,{'error':'Unknown sort order'})
                return self.respond(200,{'total':len(d),'rows':records(d.iloc[offset:offset+100])})
            if parsed.path=='/api/observation':
                q=queue();q=q[q.id==query.get('id',[''])[0]]
                return self.respond(200,{'rows':records(q)})
            if parsed.path=='/api/history':
                engine = query.get('engine',[''])[0]
                if 'engine_id' not in data:
                    return self.respond(200,{'rows':[],'note':'No engine IDs or timestamps in challenge CSVs'})
                g = data[data.engine_id==engine].sort_values('timestamp_utc').tail(200)
                return self.respond(200,{'rows':records(g),'note':'Most recent 200 observations; no persistence threshold applied'})
            if parsed.path=='/api/reviews':
                with review_database(dbpath) as db:
                    db.row_factory=sqlite3.Row
                    rows = [dict(r) for r in db.execute('SELECT * FROM audit WHERE baseline_sha256=? AND prediction_sha256=? ORDER BY sequence DESC LIMIT 200',(predictor.manifest['baseline_sha256'],prediction_scope))]
                return self.respond(200,{'rows':rows})
            if parsed.path in ['/','/app.js','/style.css']:
                file = web/({'/':'index.html','/app.js':'app.js','/style.css':'style.css'}[parsed.path])
                body=file.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type',{'/':'text/html; charset=utf-8','/app.js':'text/javascript; charset=utf-8','/style.css':'text/css; charset=utf-8'}[parsed.path])
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
                self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return self.respond(404,{'error':'not found'})

        def do_POST(self):
            if not self.host_ok() or self.headers.get('Origin') not in [None,origin] or self.headers.get('X-Session-Token')!=token:
                return self.respond(403,{'error':'Invalid local session token or origin'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size<1 or size>1000000:
                    raise ValueError('Request size must be 1..1000000 bytes')
                body=json.loads(self.rfile.read(size))
                if self.path=='/api/review':
                    obs=str(body.get('id',''))
                    reviewer=str(body.get('reviewer','')).strip()
                    disposition=body.get('disposition')
                    note=str(body.get('note',''))
                    if obs not in set(data.id) or not reviewer or len(reviewer)>100 or len(note)>4000 or disposition not in ['needs_inspection','monitor','data_issue','review_complete']:
                        raise ValueError('Provide an existing ID, reviewer and valid disposition')
                    with review_database(dbpath) as db:
                        db.execute('INSERT INTO audit (at_utc,observation_id,reviewer,disposition,note,baseline_sha256,run_sha256,prediction_sha256) VALUES (?,?,?,?,?,?,?,?)',
                            (datetime.now(timezone.utc).isoformat(),obs,reviewer,disposition,note,predictor.manifest['baseline_sha256'],runhash,prediction_scope))
                    return self.respond(201,{'status':'review recorded','ground_truth_changed':False})
                if self.path=='/api/predict':
                    observations=body.get('observations')
                    if not isinstance(observations,list) or not 1<=len(observations)<=100:
                        raise ValueError('Provide 1..100 observations')
                    if any(set(o)-set(['id']+RAW) for o in observations):
                        raise ValueError('Local measurement demo accepts id and raw measurements only; use shadow CLI for operational metadata')
                    result=records(predictor.predict(pd.DataFrame(observations),'replay'))
                    with review_database(dbpath) as db:
                        db.execute('INSERT INTO interface_predictions (at_utc,mode,payload,result,baseline_sha256) VALUES (?,?,?,?,?)',
                            (datetime.now(timezone.utc).isoformat(),'measurement_demo',json.dumps(observations),json.dumps(result),predictor.manifest['baseline_sha256']))
                    return self.respond(200,{'mode':'measurement_demo','predictions':result})
                return self.respond(404,{'error':'not found'})
            except (ValueError,KeyError,TypeError) as exc:
                return self.respond(400,{'error':str(exc)})
    print(f'Review dashboard: {origin} (Ctrl+C to stop). Local pilot; no automatic maintenance actions.',flush=True)
    with HTTPServer(('127.0.0.1',port),Handler) as server:
        server.serve_forever()
