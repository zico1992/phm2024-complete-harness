"""Evaluate immutable saved predictions against a later, versioned label export."""
from pathlib import Path
import json
import shutil
import os
import pandas as pd
from .core import sha, now, new_dir, labelled_join, evaluate_shadow
from .runtime import check_baseline_environment
from phm.data import write_json


def validated_saved_run(root,manifest):
    root=Path(root)
    ledger=json.loads((root/'run_manifest.json').read_text())
    complete=json.loads((root/'completion.json').read_text())
    for key in ['baseline_sha256','reliability_sha256','policy_sha256','threshold']:
        if ledger['baseline'][key]!=manifest[key]: raise ValueError(f'Source run baseline differs: {key}')
    if complete['baseline_sha256']!=manifest['baseline_sha256']: raise ValueError('Completion baseline differs')
    prediction_hash=sha(root/'predictions.csv')
    if prediction_hash!=ledger['prediction_sha256'] or prediction_hash!=complete['prediction_sha256']:
        raise ValueError('Saved prediction hash changed')
    if sha(root/'evaluation.json')!=ledger['evaluation_sha256']: raise ValueError('Saved evaluation changed')
    if ledger.get('evaluated_predictions_sha256') and sha(root/'evaluated_predictions.csv')!=ledger['evaluated_predictions_sha256']:
        raise ValueError('Saved evaluated predictions changed')
    d=pd.read_csv(root/'predictions.csv',dtype={'id':str,'engine_id':str,'flight_id':str})
    if d.id.isna().any() or d.id.duplicated().any() or len(d)!=complete['n'] or len(d)!=ledger['n']:
        raise ValueError('Invalid saved prediction IDs/count')
    if not d.baseline_sha256.eq(manifest['baseline_sha256']).all():raise ValueError('Row baseline identity differs')
    return ledger,d


def check_frozen_files(root,manifest):
    for file,key in [('baseline.joblib','baseline_sha256'),('reliability.joblib','reliability_sha256'),('policy.json','policy_sha256')]:
        if sha(Path(root)/file)!=manifest[key]:raise ValueError(f'Frozen artifact changed: {file}')


def delayed_evaluate(baseline,source,labels,output):
    manifest=json.loads((Path(baseline)/'manifest.json').read_text())
    runtime=check_baseline_environment(baseline)
    check_frozen_files(baseline,manifest)
    ledger,predictions=validated_saved_run(source,manifest)
    # No deserialization, model inference, fitting or calibration is allowed here.
    joined=labelled_join(predictions,labels,ledger['mode'])
    report=evaluate_shadow(joined,manifest['threshold'])
    committed=ledger.get('predictions_saved_at',ledger['created_at'])
    if ledger['mode']=='shadow':
        known=joined.faulty.notna()
        times=pd.to_datetime(joined.loc[known,'label_timestamp_utc'],utc=True,format='ISO8601')
        later=int((times>pd.Timestamp(committed)).sum())
        report['label_timing']={'labelled_after_prediction_commit':later,'labelled_on_or_before_prediction_commit':int(known.sum())-later,
            'note':'Confirmation timestamps only; this is not proof of prospective label availability or fault onset.'}
    report.update(baseline_name=manifest['name'],mode=ledger['mode'],evaluation_kind='delayed_labels',
        note='Metrics use this label export only. Supply a cumulative export for cumulative coverage; review notes are never labels.')
    out=new_dir(output)
    before={name:sha(Path(source)/name) for name in ['predictions.csv','run_manifest.json','completion.json','evaluation.json']}
    shutil.copyfile(Path(source)/'predictions.csv',out/'predictions.csv')
    shutil.copyfile(labels,out/'labels.csv')
    joined.to_csv(out/'evaluated_predictions.csv',index=False)
    write_json(out/'evaluation.json',report)
    snapshot={**ledger,'kind':'delayed_evaluation','created_at':now(),'predictions_saved_at':committed,
        'source_run_dir':os.path.relpath(Path(source).resolve(),out.resolve()),'source_reference_type':'relative_to_snapshot','source_files_sha256':before,
        'label_sha256':sha(out/'labels.csv'),'evaluated_predictions_sha256':sha(out/'evaluated_predictions.csv'),
        'evaluation_sha256':sha(out/'evaluation.json'),'evaluation_runtime':runtime,
        'labels_used_for_predictions':False,'workflow':ledger['workflow']+'; later label evaluation snapshot'}
    write_json(out/'run_manifest.json',snapshot)
    write_json(out/'completion.json',{'completed_at':now(),'mode':ledger['mode'],'n':len(predictions),
        'prediction_sha256':sha(out/'predictions.csv'),'baseline_sha256':manifest['baseline_sha256'],'kind':'delayed_evaluation'})
    for name,value in before.items():
        if sha(Path(source)/name)!=value:raise ValueError('Source run changed during evaluation')
    check_frozen_files(baseline,manifest)
    return {'output':str(out),'n':len(predictions),'labelled':report['labelled'],'predictions_recomputed':False}


def verify_delayed(baseline,run_dir,labels=None):
    manifest=json.loads((Path(baseline)/'manifest.json').read_text())
    check_baseline_environment(baseline);check_frozen_files(baseline,manifest)
    ledger,d=validated_saved_run(run_dir,manifest)
    if ledger.get('kind')!='delayed_evaluation':raise ValueError('Not a delayed-evaluation snapshot')
    root=Path(run_dir)
    label_path=Path(labels) if labels else root/'labels.csv'
    if sha(label_path)!=ledger['label_sha256']:raise ValueError('Label export changed')
    if sha(root/'labels.csv')!=ledger['label_sha256']:raise ValueError('Snapshot labels changed')
    expected=evaluate_shadow(labelled_join(d,label_path,ledger['mode']),manifest['threshold'])
    actual=json.loads((root/'evaluation.json').read_text())
    for key,value in expected.items():
        if value!=actual.get(key):raise ValueError(f'Delayed evaluation differs: {key}')
    result={'status':'passed','kind':'delayed_evaluation','n':len(d),'labelled':expected['labelled'],
        'baseline_sha256':manifest['baseline_sha256'],'predictions_recomputed':False,
        'verification_note':'Verifies stored prediction integrity and label metrics; use verify_pilot.py with original inputs for model reproduction.'}
    write_json(root/'pilot_verification.json',result)
    return result
