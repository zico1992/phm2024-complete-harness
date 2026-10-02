"""Transparent provisional queue ordering. Outcome labels never determine priority."""
from pathlib import Path
import json
import pandas as pd
from .core import sha

PRIORITY_POLICY={
    'P1':'Fault flag plus a reliability warning: engineering review of signal and support',
    'P2':'Fault flag with no support warning, or rejected measurements needing correction',
    'P3':'Reliability warning without a fault flag',
    'P4':'No implemented review trigger',
    'status':'provisional queue ordering, not fault severity or an approved maintenance rule'}


def review_location(root,ledger):
    """Keep review history attached to the original prediction run across label snapshots."""
    root=Path(root).resolve();visited=set();run_hashes=[sha(root/'run_manifest.json')]
    while ledger.get('kind')=='delayed_evaluation':
        if str(root) in visited:raise ValueError('Cyclic delayed-evaluation ancestry')
        visited.add(str(root));parent=Path(ledger['source_run_dir'])
        if ledger.get('source_reference_type')=='relative_to_snapshot':parent=(root/parent).resolve()
        for name,value in ledger['source_files_sha256'].items():
            if sha(parent/name)!=value:raise ValueError('Delayed evaluation source changed or moved; restore its recorded source for shared review history')
        parent_ledger=json.loads((parent/'run_manifest.json').read_text())
        if parent_ledger['prediction_sha256']!=ledger['prediction_sha256']:raise ValueError('Prediction ancestry differs')
        root=parent.resolve();ledger=parent_ledger;run_hashes.append(sha(root/'run_manifest.json'))
    return root/'reviews.sqlite',run_hashes


def latest_reviews(db,baseline_sha,prediction_sha):
    db.row_factory=__import__('sqlite3').Row
    rows=db.execute('''SELECT * FROM audit WHERE sequence IN
        (SELECT MAX(sequence) FROM audit WHERE baseline_sha256=? AND prediction_sha256=? GROUP BY observation_id)''',
        (baseline_sha,prediction_sha)).fetchall()
    return {r['observation_id']:dict(r) for r in rows}


def review_queue(frame,latest):
    d=frame.copy()
    fault=d.fault_flag.fillna(0).eq(1)
    rejected=~d.valid_input
    warning=d.reliability_status.ne('within_reference')
    rank=pd.Series(4,index=d.index,dtype=int)
    rank.loc[warning]=3;rank.loc[fault|rejected]=2;rank.loc[fault&warning]=1
    d['priority_rank']=rank;d['review_priority']='P'+rank.astype(str)
    d['priority_reason']=d.review_priority.map(PRIORITY_POLICY)
    for col,key in [('latest_disposition','disposition'),('latest_reviewer','reviewer'),('latest_review_at','at_utc'),('latest_review_note','note')]:
        d[col]=d.id.map(lambda obs:latest.get(str(obs),{}).get(key,''))
    d['review_state']=d.latest_disposition.map(lambda value:'complete' if value=='review_complete' else ('unreviewed' if not value else 'follow_up'))
    d['open_review']=d.review_recommended & d.review_state.ne('complete')
    return d
