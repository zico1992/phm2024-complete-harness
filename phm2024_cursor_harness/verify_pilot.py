"""Reproduce persisted predictions using the exact frozen artifact and verify run integrity."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from operational.core import Predictor, operational_input, labelled_join, evaluate_shadow, sha
from phm.data import load_x, write_json


def verify(baseline,run_dir,input_path,contract=None,labels=None):
    p=Predictor(baseline);root=Path(run_dir)
    ledger=json.loads((root/'run_manifest.json').read_text())
    complete=json.loads((root/'completion.json').read_text())
    if ledger['baseline']['baseline_sha256']!=p.manifest['baseline_sha256']:
        raise ValueError('Wrong baseline')
    if sha(input_path)!=ledger['input_sha256']:
        raise ValueError('Input hash differs')
    if sha(root/'predictions.csv')!=complete['prediction_sha256'] or complete['prediction_sha256']!=ledger['prediction_sha256']:
        raise ValueError('Prediction file changed')
    if ledger['mode']=='shadow':
        if not contract or sha(contract)!=ledger['contract_sha256']:
            raise ValueError('Matching operational contract required')
        x=operational_input(input_path,contract,p)
    else:x=load_x(input_path)
    generated=p.predict(x,ledger['mode'])
    saved=pd.read_csv(root/'predictions.csv',dtype={'id':str})
    if list(generated.id)!=list(saved.id): raise ValueError('Prediction IDs/order differ')
    for col in ['p_faulty','margin_mean','margin_sigma','joint_distance','environment_distance']:
        np.testing.assert_allclose(generated[col],saved[col],rtol=1e-12,atol=1e-12,equal_nan=True)
    for col in ['valid_input','regime','training_or_calibration_duplicate','reliability_status','review_recommended']:
        if list(generated[col])!=list(saved[col]):raise ValueError(f'{col} changed')
    if ledger.get('evaluation_sha256')!=sha(root/'evaluation.json'):raise ValueError('Evaluation hash changed')
    if labels:
        if ledger.get('evaluated_predictions_sha256')!=sha(root/'evaluated_predictions.csv'):raise ValueError('Evaluated predictions changed')
        if sha(labels)!=ledger.get('label_sha256'):raise ValueError('Label hash differs')
        evaluated=labelled_join(generated,labels,ledger['mode'])
        expected=evaluate_shadow(evaluated,p.manifest['threshold'])
        actual=json.loads((root/'evaluation.json').read_text())
        if expected['evaluated_independent_feature_rows']!=actual['evaluated_independent_feature_rows']:raise ValueError('Evaluation denominator changed')
        if expected['metrics']:
            for family in ['classification','regression']:
                for key,value in expected['metrics'][family].items():
                    other=actual['metrics'][family][key]
                    if isinstance(value,(float,int)) and value is not None:
                        np.testing.assert_allclose(value,other,rtol=1e-10,atol=1e-10)
                    elif value!=other:raise ValueError(f'Metric changed: {key}')
    p.assert_frozen()
    result={'status':'passed','n':len(saved),'baseline_sha256':p.manifest['baseline_sha256'],'mode':ledger['mode'],'labels_verified':bool(labels)}
    write_json(root/'pilot_verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True);parser.add_argument('--run-dir',required=True);parser.add_argument('--input',required=True);parser.add_argument('--contract');parser.add_argument('--labels')
    a=parser.parse_args()
    print(json.dumps(verify(a.baseline,a.run_dir,a.input,a.contract,a.labels),indent=2))
