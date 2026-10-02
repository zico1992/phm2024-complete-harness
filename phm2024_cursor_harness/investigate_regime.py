"""Re-run a frozen regime-exclusion experiment and persist its diagnostic evidence.

Original harness versions save aggregate stress metrics only. This exporter records
reproduction provenance; a reconstructed cluster is never claimed to be the original
fitted cluster. No model/feature/threshold search is performed on the holdout.
"""
import argparse
import json
import logging
import platform
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

from phm.data import RAW, ENV, load_data, split_groups, write_json, manifests
from phm.models import JointModel, Calibrator
from phm.evaluate import metrics, submission_action


def save_csv(frame,path,**kwargs):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    frame.to_csv(temporary,**kwargs)
    temporary.replace(path)


def save_model(obj,path):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    joblib.dump(obj,temporary,compress=3)
    joblib.load(temporary)  # Check serialization before exposing a final artifact.
    temporary.replace(path)


def summarize(s):
    faulty=s.actual_faulty.eq(1)
    healthy=~faulty
    fn=s.false_negative.sum();fp=s.false_positive.sum()
    return dict(rows=len(s),faulty_rows=int(faulty.sum()),healthy_rows=int(healthy.sum()),
        false_negatives=int(fn),false_positives=int(fp),
        fault_recall=float(1-fn/faulty.sum()) if faulty.any() else None,
        false_positive_rate=float(fp/healthy.sum()) if healthy.any() else None,
        accuracy=float(1-s.classification_error.mean()),
        margin_mae=float(s.absolute_margin_error.mean()),
        coverage90=float(s.covered90.mean()),mean_sigma=float(s.margin_sigma.mean()))


def compare_metrics(out,d,baseline,stressed,original):
    def row(name,m):
        c,r=m['classification'],m['regression']
        return {'experiment':name,'rows':m['n'],'accuracy':c['accuracy'],
            'fault_recall':c['recall_faulty'],'false_negatives':c['false_negatives'],
            'false_positives':c['false_positives'],'mae':r['mae'],
            'coverage90':r['interval_90_coverage'],'interval90_mean_width':r['interval_90_mean_width']}
    rows=[row('main_model_same_reproduced_regime_rows',baseline),
          row('reproduced_exclusion_refit',stressed)]
    if original:
        rows.append(row('original_exclusion_aggregate_different_membership_possible',original['metrics']))
    save_csv(pd.DataFrame(rows),out/'metric_comparison.csv',index=False)


def investigate(args):
    start=time.time();out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be new/empty')
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(out/'investigation.log')])
    run=Path(args.run_dir)
    cfg=json.loads((run/'config.json').read_text())
    spec=json.loads((run/'selected_spec.json').read_text())
    original_stress=json.loads((run/'stress_tests.json').read_text())
    original=next((r for r in original_stress.get('results',[]) if r['regime']==args.regime),None)
    hp=pd.read_csv(run/'holdout_predictions.csv')
    d,_=load_data(args.data_dir)
    if cfg['max_rows'] and len(d)>cfg['max_rows']:
        d=d.sample(cfg['max_rows'],random_state=cfg['seed']).reset_index(drop=True)
    parts=split_groups(d,cfg['seed'])
    if set(hp.id)!=set(d.iloc[parts['final']].id):
        raise ValueError('Holdout IDs differ from reconstructed split; provide original partition manifest/code')
    truth=d.set_index('id').loc[hp.id]
    if not np.array_equal(truth.faulty.to_numpy(),hp.actual_faulty.to_numpy()):
        raise ValueError('Original holdout class labels differ from source targets')
    np.testing.assert_allclose(truth.trq_margin.to_numpy(),hp.actual_margin.to_numpy(),rtol=1e-10,atol=1e-10)
    d['experiment_kind']='instrumented_reproduction'
    for name,ix in parts.items():d.loc[d.index[ix],'partition']=name
    logging.info('Reconstructing stress clustering from training partition only')
    if args.artifact_dir:
        saved_cluster=joblib.load(Path(args.artifact_dir)/'stress_cluster.joblib')
        scaler,km=saved_cluster['scaler'],saved_cluster['cluster']
    else:
        scaler=RobustScaler().fit(d.iloc[parts['train']][ENV])
        km=MiniBatchKMeans(n_clusters=cfg['physics']['regimes'],n_init=3,batch_size=2048,
                          random_state=cfg['seed']).fit(scaler.transform(d.iloc[parts['train']][ENV]))
    g=km.predict(scaler.transform(d[ENV]));d['stress_regime']=g
    centers=scaler.inverse_transform(km.cluster_centers_)
    center_table=pd.DataFrame(centers,columns=ENV).assign(regime=np.arange(len(centers)))
    save_csv(center_table,out/'all_cluster_centers.csv',index=False)
    counts=d.groupby(['stress_regime','partition']).size().unstack(fill_value=0)
    save_csv(counts,out/'cluster_partition_counts.csv')
    save_csv(d[['id','duplicate_group','partition','stress_regime','experiment_kind']],out/'cluster_membership.csv',index=False)
    build=np.concatenate([parts['train'],parts['selection']])
    tr=build[g[build]!=args.regime]
    cal=parts['calibration'][g[parts['calibration']]!=args.regime]
    test=parts['final'][g[parts['final']]==args.regime]
    assert len(test)>0 and args.regime<len(centers)
    assert not np.isin(g[tr],args.regime).any() and not np.isin(g[cal],args.regime).any()
    actual_counts={'fit_rows':len(tr),'calibration_rows':len(cal),'holdout_rows':len(test)}
    expected_counts={'fit_rows':original['fit_rows'],'calibration_rows':original['calibration_rows'],
                     'holdout_rows':original['metrics']['n']} if original else None
    provenance={'status':'new_instrumented_reproduction','regime':args.regime,
        'original_cluster_recovery':'Not established: original fitted stress cluster/model and row-level stress predictions were not saved.',
        'original_counts':expected_counts,'reproduced_counts':actual_counts,
        'counts_match_original':actual_counts==expected_counts,
        'warning':'Matching counts alone would not prove identical membership or predictions.',
        'python':sys.version,'platform':platform.platform(),'sklearn':sklearn.__version__,
        'numpy':np.__version__,'pandas':pd.__version__,'config':cfg,'selected_spec':spec,
        'input_manifest':manifests(args.data_dir)}
    write_json(out/'provenance.json',provenance)
    definition={'regime':args.regime,'features':ENV,'feature_semantics':cfg['physics'],
        'scaler':'RobustScaler: (x-center)/scale','scaler_center':scaler.center_.tolist(),
        'scaler_scale':scaler.scale_.tolist(),'all_centers_in_scaled_space':km.cluster_centers_.tolist(),
        'selected_center_in_raw_units':dict(zip(ENV,centers[args.regime].tolist())),
        'membership_rule':'argmin Euclidean distance to the stored centers after the stored scaling',
        'parameters':{'n_clusters':cfg['physics']['regimes'],'n_init':3,'batch_size':2048,'random_state':cfg['seed']},
        'note':'Cluster numbers are arbitrary operating-condition labels, not engine IDs. PA units/meaning remain unresolved.'}
    write_json(out/'stress_cluster_definition.json',definition)
    save_model({'scaler':scaler,'cluster':km,'definition':definition},out/'stress_cluster.joblib')
    measurements=d[g==args.regime].copy()
    measurements['excluded_from_reproduced_fit']=measurements.partition.isin(['train','selection'])
    measurements['excluded_from_reproduced_calibration']=measurements.partition.eq('calibration')
    save_csv(measurements,out/f'regime{args.regime}_measurements.csv',index=False)
    write_json(out/'measurement_summary.json',measurements[RAW].describe(percentiles=[.05,.25,.5,.75,.95]).to_dict())
    logging.info('Regime %s counts: original=%s; reproduced=%s',args.regime,expected_counts,actual_counts)
    logging.info('Fitting frozen selected architecture with regime removed; no tuning')
    if args.artifact_dir:
        saved_model=joblib.load(Path(args.artifact_dir)/'reproduced_stress_model.joblib')
        assert saved_model['config']==cfg and saved_model['spec']==spec
        assert set(saved_model['training_ids'])==set(d.iloc[tr].id)
        assert set(saved_model['calibration_ids'])==set(d.iloc[cal].id)
        model,calibration=saved_model['model'],saved_model['calibrator']
    else:
        model=JointModel(spec['mode'],spec['architecture'],cfg,cfg['seed'],spec.get('keep')).fit(
            d.iloc[tr],d.iloc[tr],d.iloc[tr].duplicate_group.to_numpy())
        calibration=Calibrator().fit(model,d.iloc[cal],d.iloc[cal])
    save_model({'model':model,'calibrator':calibration,'spec':spec,'config':cfg,
        'training_ids':d.iloc[tr].id.to_numpy(),'calibration_ids':d.iloc[cal].id.to_numpy(),
        'excluded_regime':args.regime,'provenance':provenance},out/'reproduced_stress_model.joblib')
    evaluation=d.iloc[test]
    p,mu,sigma=calibration.predict(model,evaluation)
    sm=metrics(evaluation,p,mu,sigma,calibration.threshold)
    write_json(out/'reproduced_stress_metrics.json',sm)
    pred=evaluation[['id']+RAW+['faulty','trq_margin']].rename(columns={'faulty':'actual_faulty','trq_margin':'actual_margin'}).copy()
    pred['stress_regime']=args.regime
    pred['experiment_kind']='instrumented_reproduction'
    pred['p_faulty']=p;pred['class_f2_threshold']=(p>=calibration.threshold).astype(int)
    pred['classification_threshold']=calibration.threshold
    pred['class_submission'],pred['class_conf']=submission_action(p)
    pred['margin_mean']=mu;pred['margin_sigma']=sigma
    pred['margin_residual']=pred.actual_margin-mu
    pred['absolute_margin_error']=pred.margin_residual.abs()
    pred['standardized_margin_error']=pred.margin_residual/sigma
    pred['margin_lower90']=mu-norm.ppf(.95)*sigma;pred['margin_upper90']=mu+norm.ppf(.95)*sigma
    pred['covered90']=pred.standardized_margin_error.abs()<=norm.ppf(.95)
    pred['false_negative']=pred.actual_faulty.eq(1)&pred.class_f2_threshold.eq(0)
    pred['false_positive']=pred.actual_faulty.eq(0)&pred.class_f2_threshold.eq(1)
    pred['classification_error']=pred.actual_faulty.ne(pred.class_f2_threshold)
    original_main=hp.set_index('id').loc[pred.id]
    for c in ['p_faulty','class_f2_threshold','margin_mean','margin_sigma']:
        pred[f'main_model_{c}']=original_main[c].to_numpy()
    pred['main_model_absolute_margin_error']=np.abs(pred.actual_margin-pred.main_model_margin_mean)
    pred['main_model_classification_error']=pred.actual_faulty.ne(pred.main_model_class_f2_threshold)
    baseline=metrics(evaluation,pred.main_model_p_faulty.to_numpy(),pred.main_model_margin_mean.to_numpy(),
                     pred.main_model_margin_sigma.to_numpy(),json.loads((run/'final_metrics.json').read_text())['classification']['threshold'])
    write_json(out/'main_model_same_rows_metrics.json',baseline)
    compare_metrics(out,d,baseline,sm,original)
    if not args.skip_random_control:
        logging.info('Fitting a size-matched random-group removal control, retaining all regimes')
        a,_=next(GroupShuffleSplit(n_splits=1,train_size=len(tr)/len(build),
                    random_state=cfg['seed']+1000).split(build,groups=d.iloc[build].duplicate_group))
        control_tr=build[a]
        a,_=next(GroupShuffleSplit(n_splits=1,train_size=len(cal)/len(parts['calibration']),
                    random_state=cfg['seed']+1001).split(parts['calibration'],groups=d.iloc[parts['calibration']].duplicate_group))
        control_cal=parts['calibration'][a]
        if args.artifact_dir:
            saved_control=joblib.load(Path(args.artifact_dir)/'random_removal_control.joblib')
            assert saved_control['config']==cfg and saved_control['spec']==spec
            assert set(saved_control['training_ids'])==set(d.iloc[control_tr].id)
            assert set(saved_control['calibration_ids'])==set(d.iloc[control_cal].id)
            control_model,control_calibrator=saved_control['model'],saved_control['calibrator']
        else:
            control_model=JointModel(spec['mode'],spec['architecture'],cfg,cfg['seed'],spec.get('keep')).fit(
                d.iloc[control_tr],d.iloc[control_tr],d.iloc[control_tr].duplicate_group.to_numpy())
            control_calibrator=Calibrator().fit(control_model,d.iloc[control_cal],d.iloc[control_cal])
        cp,cm,cs=control_calibrator.predict(control_model,evaluation)
        control_metrics=metrics(evaluation,cp,cm,cs,control_calibrator.threshold)
        write_json(out/'random_removal_control_metrics.json',{'fit_rows':len(control_tr),
            'calibration_rows':len(control_cal),'metrics':control_metrics,
            'note':'Approximately size matched through whole duplicate-group sampling. One frozen-model seed, no tuning.'})
        save_csv(pd.DataFrame({'id':evaluation.id,'p_faulty':cp,'class_f2_threshold':(cp>=control_calibrator.threshold).astype(int),
            'margin_mean':cm,'margin_sigma':cs,'experiment_kind':'instrumented_reproduction_control'}),out/'random_removal_control_predictions.csv',index=False)
        save_model({'model':control_model,'calibrator':control_calibrator,'training_ids':d.iloc[control_tr].id.to_numpy(),
            'calibration_ids':d.iloc[control_cal].id.to_numpy(),'config':cfg,'spec':spec},out/'random_removal_control.joblib')
        c,r=control_metrics['classification'],control_metrics['regression']
        table=pd.read_csv(out/'metric_comparison.csv')
        table=pd.concat([table,pd.DataFrame([{'experiment':'size_matched_random_group_removal_control',
            'rows':control_metrics['n'],'accuracy':c['accuracy'],'fault_recall':c['recall_faulty'],
            'false_negatives':c['false_negatives'],'false_positives':c['false_positives'],
            'mae':r['mae'],'coverage90':r['interval_90_coverage'],
            'interval90_mean_width':r['interval_90_mean_width']}])],ignore_index=True)
        save_csv(table,out/'metric_comparison.csv',index=False)
    logging.info('Calculating training coverage and error concentrations')
    fit=d.iloc[tr]
    qrows=[]
    for c in RAW:
        lo,hi=float(fit[c].min()),float(fit[c].max())
        pred[f'{c}_outside_fit_range']=(pred[c]<lo)|(pred[c]>hi)
        qrows.append({'feature':c,'fit_min':lo,'fit_p05':fit[c].quantile(.05),
            'fit_median':fit[c].median(),'fit_p95':fit[c].quantile(.95),'fit_max':hi,
            'excluded_holdout_min':pred[c].min(),'excluded_holdout_median':pred[c].median(),
            'excluded_holdout_max':pred[c].max(),'fraction_outside_fit_range':pred[f'{c}_outside_fit_range'].mean()})
    save_csv(pd.DataFrame(qrows),out/'coverage_ranges.csv',index=False)
    # Same reference sample for excluded and retained holdout comparisons.
    reference=fit.sample(min(100000,len(fit)),random_state=cfg['seed'])
    final=d.iloc[parts['final']]
    coverage=[]
    for name,columns in [('environment',ENV),('all_raw',RAW)]:
        ss=RobustScaler().fit(fit[columns]);nn=NearestNeighbors(n_neighbors=1,algorithm='kd_tree').fit(ss.transform(reference[columns]))
        distances,neighbor=nn.kneighbors(ss.transform(final[columns]))
        lookup=pd.Series(distances[:,0],index=final.id)
        pred[f'{name}_nearest_fit_distance']=lookup.loc[pred.id].to_numpy()
        nearest_ids=pd.Series(reference.id.to_numpy()[neighbor[:,0]],index=final.id)
        pred[f'{name}_nearest_fit_id']=nearest_ids.loc[pred.id].to_numpy()
        retained=distances[:,0][g[parts['final']]!=args.regime]
        excluded=pred[f'{name}_nearest_fit_distance'].to_numpy()
        threshold=float(np.quantile(retained,.95))
        coverage.append({'space':name,'reference_fit_sample_rows':len(reference),
            'retained_holdout_median_distance':float(np.median(retained)),
            'retained_holdout_p95_distance':threshold,
            'excluded_holdout_median_distance':float(np.median(excluded)),
            'excluded_fraction_above_retained_p95':float((excluded>threshold).mean()),
            'distance_vs_abs_error_spearman':float(spearmanr(excluded,pred.absolute_margin_error).statistic),
            'distance_vs_classification_error_spearman':float(spearmanr(excluded,pred.classification_error).statistic)})
    write_json(out/'coverage_neighbors.json',coverage)
    condition_rows=[]
    for c in RAW+['environment_nearest_fit_distance','all_raw_nearest_fit_distance']:
        # Exploratory quintiles of the evaluation measurements; no predictive rule is fitted.
        bins=pd.qcut(pred[c],5,duplicates='drop')
        for interval,sub in pred.groupby(bins,observed=True):
            condition_rows.append({'feature':c,'bin':str(interval),'min':float(sub[c].min()),
                                   'max':float(sub[c].max()),**summarize(sub)})
    for name,mask in [('np_lt_ng',pred.np<pred.ng),('np_ge_ng',pred.np>=pred.ng),
                      ('zero_ias',pred.ias==0),('nonzero_ias',pred.ias!=0)]:
        sub=pred[mask]
        if len(sub):condition_rows.append({'feature':'operating_indicator','bin':name,**summarize(sub)})
    save_csv(pd.DataFrame(condition_rows),out/'condition_error_rates.csv',index=False)
    save_csv(pred,out/f'regime{args.regime}_predictions.csv',index=False)
    save_csv(pred[pred.false_negative|pred.false_positive],out/'classification_errors.csv',index=False)
    save_csv(pred.nlargest(200,'absolute_margin_error'),out/'largest_margin_errors.csv',index=False)
    save_csv(pred[pred.false_negative],out/'false_negatives.csv',index=False)
    main_threshold=json.loads((run/'final_metrics.json').read_text())['classification']['threshold']
    comparisons=[('main_model',pred.main_model_p_faulty.to_numpy()),('excluded_refit',p)]
    if not args.skip_random_control:comparisons.append(('random_removal_control',cp))
    fixed=[]
    for name,probs in comparisons:
        label=probs>=main_threshold
        fixed.append({'model':name,'common_threshold':main_threshold,'rows':len(pred),
            'accuracy':float(accuracy_score(pred.actual_faulty,label)),
            'fault_recall':float(recall_score(pred.actual_faulty,label)),
            'false_negatives':int(((pred.actual_faulty==1)&(~label)).sum()),
            'false_positives':int(((pred.actual_faulty==0)&label).sum())})
    save_csv(pd.DataFrame(fixed),out/'fixed_threshold_comparison.csv',index=False)
    oe=pred.oat.quantile([0,.2,.4,.6,.8,1]).to_numpy()
    tq=pred.trq_measured.quantile([.4,.6]).to_numpy()
    pa80=float(pred.pa.quantile(.8));ias60=float(pred.ias.quantile(.6))
    focus={f'OAT_above_{oe[3]:g}_through_{oe[4]:g}':(pred.oat>oe[3])&(pred.oat<=oe[4]),
        f'PA_above_{pa80:g}':pred.pa>pa80,f'IAS_above_{ias60:g}':pred.ias>ias60,
        f'OAT_above_{oe[1]:g}_through_{oe[2]:g}':(pred.oat>oe[1])&(pred.oat<=oe[2]),
        f'torque_above_{tq[0]:g}_through_{tq[1]:g}':(pred.trq_measured>tq[0])&(pred.trq_measured<=tq[1])}
    focused=[]
    for name,mask in focus.items():
        s=pred[mask];faulty=s.actual_faulty.eq(1);healthy=~faulty
        focused.append({'condition':name,**summarize(s),
            'FN_main_same_rows':int((faulty&s.main_model_class_f2_threshold.eq(0)).sum()),
            'FP_main_same_rows':int((healthy&s.main_model_class_f2_threshold.eq(1)).sum()),
            'percent_all_FNs':float(100*s.false_negative.sum()/max(pred.false_negative.sum(),1)),
            'percent_all_FPs':float(100*s.false_positive.sum()/max(pred.false_positive.sum(),1))})
    save_csv(pd.DataFrame(focused),out/'focused_condition_errors.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    b=fit.sample(min(12000,len(fit)),random_state=cfg['seed']);a=pred.sample(min(12000,len(pred)),random_state=cfg['seed'])
    axs[0,0].scatter(b.oat,b.pa,s=2,alpha=.15,label='Retained fitting sample')
    axs[0,0].scatter(a.oat,a.pa,s=3,alpha=.25,label='Excluded regime holdout')
    axs[0,0].set(xlabel='OAT (C assumption)',ylabel='PA (meaning/units unresolved)',title='Joint environmental coverage');axs[0,0].legend(markerscale=3)
    ax=axs[0,1];ax.scatter(a.ias,a.oat,s=3,alpha=.2,color='gray',label='Excluded holdout')
    err=pred[pred.classification_error];ax.scatter(err.ias,err.oat,s=5,alpha=.5,color='crimson',label='Classification errors')
    ax.set(xlabel='IAS (units unresolved)',ylabel='OAT (C assumption)',title='Where classification errors occur');ax.legend(markerscale=2)
    axs[1,0].scatter(a.all_raw_nearest_fit_distance,a.absolute_margin_error,s=3,alpha=.25)
    axs[1,0].set(xlabel='Nearest retained fitting distance (scaled raw inputs)',ylabel='Absolute margin error (percentage points)',title='Training support and regression error')
    axs[1,1].scatter(a.actual_margin,a.margin_mean,s=3,alpha=.25)
    lo,hi=pred.actual_margin.min(),pred.actual_margin.max();axs[1,1].plot([lo,hi],[lo,hi],'--',color='gray')
    axs[1,1].set(xlabel='Actual margin (%)',ylabel='Predicted margin (%)',title='Reproduced exclusion predictions')
    fig.tight_layout();fig.savefig(out/'diagnostics.png',dpi=140);plt.close(fig)
    write_json(out/'completion.json',{'status':'complete','elapsed_seconds':time.time()-start,
        'regime':args.regime,'exported_prediction_rows':len(pred),'exported_measurement_rows':len(measurements),
        'provenance':'new instrumented reproduction, not recovered original fitted experiment'})
    logging.info('Complete: %s',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--regime',type=int,default=5)
    parser.add_argument('--skip-random-control',action='store_true',help='Skip the size-matched removal control refit')
    parser.add_argument('--artifact-dir',help='Reuse trusted frozen investigation cluster/models instead of refitting')
    args=parser.parse_args()
    cfg=json.loads((Path(args.run_dir)/'config.json').read_text())
    with threadpool_limits(limits=cfg['threads']):investigate(args)
