import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from phm.data import load_data, RAW, ENV, write_json
from phm.evaluate import metrics


def verify(root,data_dir):
    root=Path(root);d,_=load_data(data_dir)
    provenance=json.loads((root/'provenance.json').read_text());k=provenance['regime']
    cluster=joblib.load(root/'stress_cluster.joblib')
    membership=pd.read_csv(root/'cluster_membership.csv').set_index('id')
    pred=pd.read_csv(root/f'regime{k}_predictions.csv')
    with threadpool_limits(limits=provenance['config']['threads']):
        g=cluster['cluster'].predict(cluster['scaler'].transform(d[ENV]))
        assert np.array_equal(g,membership.loc[d.id].stress_regime.to_numpy())
        expected=set(membership[(membership.partition=='final')&(membership.stress_regime==k)].index)
        assert set(pred.id)==expected and not pred.id.duplicated().any()
        truth=d.set_index('id').loc[pred.id]
        np.testing.assert_allclose(truth[RAW].to_numpy(),pred[RAW].to_numpy(),rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(truth.trq_margin,pred.actual_margin,rtol=1e-10,atol=1e-10)
        assert np.array_equal(truth.faulty.to_numpy(),pred.actual_faulty.to_numpy())
        stress=joblib.load(root/'reproduced_stress_model.joblib')
        fit_ids=set(stress['training_ids']);cal_ids=set(stress['calibration_ids'])
        assert fit_ids.isdisjoint(cal_ids) and fit_ids.isdisjoint(expected) and cal_ids.isdisjoint(expected)
        assert not (membership.loc[list(fit_ids)].stress_regime==k).any()
        assert not (membership.loc[list(cal_ids)].stress_regime==k).any()
        fit_groups=set(membership.loc[list(fit_ids)].duplicate_group)
        assert fit_groups.isdisjoint(set(membership.loc[list(expected)].duplicate_group))
        p,mu,sigma=stress['calibrator'].predict(stress['model'],pred)
        np.testing.assert_allclose(p,pred.p_faulty,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(mu,pred.margin_mean,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(sigma,pred.margin_sigma,rtol=1e-6,atol=1e-8)
        assert (sigma>0).all() and np.isfinite(np.column_stack([p,mu,sigma])).all()
        for c in ['all_raw_nearest_fit_id','environment_nearest_fit_id']:
            assert set(pred[c])<=fit_ids
        recomputed=metrics(pd.DataFrame({'faulty':pred.actual_faulty,'trq_margin':pred.actual_margin}),
                           p,mu,sigma,stress['calibrator'].threshold)
        recorded=json.loads((root/'reproduced_stress_metrics.json').read_text())
        for section,key in [('classification','accuracy'),('classification','recall_faulty'),
                            ('regression','mae'),('regression','interval_90_coverage')]:
            np.testing.assert_allclose(recomputed[section][key],recorded[section][key],rtol=1e-8,atol=1e-10)
        if (root/'random_removal_control.joblib').exists():
            control=joblib.load(root/'random_removal_control.joblib')
            assert set(control['training_ids']).isdisjoint(expected)
            assert set(control['calibration_ids']).isdisjoint(expected)
            assert (membership.loc[control['training_ids']].stress_regime==k).any()
            cp,cm,cs=control['calibrator'].predict(control['model'],pred)
            saved=pd.read_csv(root/'random_removal_control_predictions.csv').set_index('id').loc[pred.id]
            np.testing.assert_allclose(np.column_stack([cp,cm,cs]),saved[['p_faulty','margin_mean','margin_sigma']].to_numpy(),rtol=1e-6,atol=1e-8)
    result={'status':'passed','prediction_rows':len(pred),
        'checks':['persisted cluster reproduces membership','source measurements and targets match exports',
                  'fitting/calibration/evaluation are separated','excluded regime absent from stressed fitting and calibration',
                  'duplicate groups remain separated','saved stressed model reproduces predictions and metrics',
                  'nearest reference IDs are fitting rows','random-control artifact reproduces predictions'],
        'original_experiment_identity':'Not verified or claimed; these are instrumented reproduction artifacts.'}
    write_json(root/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--data-dir',default='data')
    a=p.parse_args();verify(a.output_dir,a.data_dir)
