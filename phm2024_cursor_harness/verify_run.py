"""Validate a completed experiment against its original data and saved artifact."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from phm.data import load_x, write_json, manifests
from phm.evaluate import submission_action
from operational.runtime import check_model_environment, load_trusted


def verify(root, data_dir):
    root=Path(root);data_dir=Path(data_dir)
    completion=json.loads((root/'completion.json').read_text())
    assert completion['stage']=='complete'
    expected=json.loads((root/'input_manifest.json').read_text())
    assert expected==manifests(data_dir),'Input contents changed since experiment'
    membership=pd.read_csv(root/'partition_manifest.csv')
    assert len(membership)==completion['experiment_rows']
    assert not membership.id.duplicated().any()
    assert membership.groupby('duplicate_group').partition.nunique().max()==1
    runtime=check_model_environment(root/'selected_model.joblib')
    bundle=load_trusted(root/'selected_model.joblib')
    fitting=set(membership.loc[membership.partition.isin(['train','selection']),'id'])
    calibration=set(membership.loc[membership.partition=='calibration','id'])
    final=set(membership.loc[membership.partition=='final','id'])
    assert set(bundle['training_ids'])==fitting
    assert set(bundle['calibration_ids'])==calibration
    assert fitting.isdisjoint(calibration) and fitting.isdisjoint(final) and calibration.isdisjoint(final)
    assert bundle['spec']==json.loads((root/'selected_spec.json').read_text())
    hp=pd.read_csv(root/'holdout_predictions.csv')
    assert set(hp.id)==final
    x=load_x(data_dir/'X_train.csv').set_index('id').loc[hp.id].reset_index()
    with threadpool_limits(limits=bundle['config']['threads']):
        p,mu,sigma=bundle['calibrator'].predict(bundle['model'],x)
        np.testing.assert_allclose(p,hp.p_faulty,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(mu,hp.margin_mean,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(sigma,hp.margin_sigma,rtol=1e-6,atol=1e-8)
        for name in ['test','validation']:
            filename='submission.jso' if name=='test' else 'validation_submission.jso'
            x=load_x(data_dir/f'X_{name}.csv')
            s=json.loads((root/'submissions'/filename).read_text())
            assert set(s)==set(x.id.astype(str)) and len(s)==len(x)
            p,mu,sigma=bundle['calibrator'].predict(bundle['model'],x)
            label,conf=submission_action(p)
            for j,i in enumerate(x.id):
                a=s[str(i)]
                assert set(a)=={'class','class_conf','pdf_type','pdf_args'}
                assert a['class'] in [0,1] and 0<=a['class_conf']<=1 and a['pdf_type']=='norm'
                assert set(a['pdf_args'])=={'loc','scale'}
                assert np.isfinite(a['pdf_args']['loc']) and np.isfinite(a['pdf_args']['scale'])
                assert a['pdf_args']['scale']>0
                assert a['class']==label[j]
                np.testing.assert_allclose([a['class_conf'],a['pdf_args']['loc'],a['pdf_args']['scale']],
                                            [conf[j],mu[j],sigma[j]],rtol=1e-6,atol=1e-8)
    result={'status':'passed','runtime_preflight':runtime,'experiment_rows':len(membership),'holdout_rows':len(hp),
        'checks':['data hashes','duplicate group separation','fitting/calibration/holdout id separation',
                  'saved artifact reproduces holdout predictions','external submission id coverage',
                  'legal schema and finite positive PDF scale','submissions reproduce saved artifact']}
    write_json(root/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--data-dir',default='data')
    args=parser.parse_args()
    verify(args.run_dir,args.data_dir)
