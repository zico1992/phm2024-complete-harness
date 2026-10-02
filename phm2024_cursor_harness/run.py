import argparse
import copy
import json
import logging
from pathlib import Path
import platform
import sys
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits
from phm.data import RAW, ENV, load_data, load_x, split_groups, manifests, write_json
from phm.eda import run_eda, plot_evaluation
from phm.models import JointModel, Calibrator
from operational.runtime import current_versions, check_model_environment, load_trusted
from phm.evaluate import metrics, selection_utility, bootstrap_intervals, submission, submission_action


def validate_config(c):
    p=c['physics']
    if p['temperature_unit'] not in ['C','unknown']:
        raise ValueError('temperature_unit must be C or unknown')
    if p['pa_meaning'] not in ['opaque','pressure_altitude']:
        raise ValueError('pa_meaning must be opaque or pressure_altitude')
    if p['pa_meaning']=='pressure_altitude' and p['pa_unit'] not in ['m','ft']:
        raise ValueError('Altitude features require explicit m or ft')
    if p['ias_unit'] not in ['knots','unknown']:
        raise ValueError('ias_unit must be knots or unknown')
    if p['regimes']<2 or c['stack_folds']<2 or c['scale_floor']<=0:
        raise ValueError('Invalid regimes, stack_folds or scale_floor')
    for candidate in c['candidates']:
        if candidate['mode'] not in ['raw','physics','statistical','combined']:
            raise ValueError('Unknown feature mode')
        if candidate['architecture'] not in ['direct','target_torque','linear']:
            raise ValueError('Unknown architecture')


def fit_model(d, ix, spec, cfg):
    return JointModel(spec['mode'],spec['architecture'],cfg,cfg['seed'],spec.get('keep')).fit(
        d.iloc[ix],d.iloc[ix],d.iloc[ix].duplicate_group.to_numpy())


def predict_files(bundle, data_dir, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name in ['test','validation']:
        x=load_x(Path(data_dir)/f'X_{name}.csv')
        p,mu,sigma=bundle['calibrator'].predict(bundle['model'],x)
        filename='submission.jso' if name=='test' else 'validation_submission.jso'
        write_json(out/filename,submission(x.id,p,mu,sigma))
        cls,conf=submission_action(p)
        pd.DataFrame({'id':x.id,'p_faulty':p,'class_submission':cls,'class_conf':conf,
            'class_f2_threshold':(p>=bundle['calibrator'].threshold).astype(int),
            'margin_mean':mu,'margin_sigma':sigma,
            'margin_lower_90':mu-1.644854*sigma,'margin_upper_90':mu+1.644854*sigma}).to_csv(
                out/f'{name}_predictions.csv',index=False)
    write_json(out/'external_evaluation_status.json',{
        'test':'No ground-truth labels supplied. Predictions only, no accuracy claim.',
        'validation':'No ground-truth labels supplied. Predictions only, no accuracy claim.'})


def feature_selection(model, train, selection, cfg, out):
    s=selection.sample(min(cfg['permutation_rows'],len(selection)),random_state=cfg['seed'])
    f=model.inference_features(s)
    importance=permutation_importance(model.clf,f,s.faulty,scoring='roc_auc',
        n_repeats=cfg['permutation_repeats'],random_state=cfg['seed'],n_jobs=1)
    table=pd.DataFrame({'feature':f.columns,'auc_drop_mean':importance.importances_mean,
                        'auc_drop_std':importance.importances_std}).sort_values('auc_drop_mean',ascending=False)
    table.to_csv(out/'permutation_importance_selection.csv',index=False)
    # MI is descriptive, training-only. Selection decisions use validation permutation ranking.
    t=train.sample(min(2500,len(train)),random_state=cfg['seed'])
    ft=model.builder.transform(t)
    mi_c=mutual_info_classif(ft,t.faulty,random_state=cfg['seed'])
    mi_r=mutual_info_regression(ft,t.trq_margin,random_state=cfg['seed'])
    pd.DataFrame({'feature':ft.columns,'mi_class':mi_c,'mi_margin':mi_r}).to_csv(out/'mutual_information_train.csv',index=False)
    positive=table[(table.feature!='predicted_margin')&(table.auc_drop_mean>0)].feature.tolist()
    # Preserve physically interpretable measured inputs. Selected variant competes with its parent.
    return sorted(set(RAW + positive[:cfg['selection_top_k']]))


def stress_tests(d, parts, selected, cfg, out):
    if cfg['stress_regimes']==0:
        write_json(out/'stress_tests.json',{'status':'disabled by configuration'})
        return
    scaler=RobustScaler().fit(d.iloc[parts['train']][ENV])
    km=MiniBatchKMeans(n_clusters=cfg['physics']['regimes'],n_init=3,batch_size=2048,
                     random_state=cfg['seed']).fit(scaler.transform(d.iloc[parts['train']][ENV]))
    regime=km.predict(scaler.transform(d[ENV]))
    final_counts=pd.Series(regime[parts['final']]).value_counts()
    results=[]
    build=np.concatenate([parts['train'],parts['selection']])
    for k in final_counts.index[:cfg['stress_regimes']]:
        tr=build[regime[build]!=k]
        cal=parts['calibration'][regime[parts['calibration']]!=k]
        test=parts['final'][regime[parts['final']]==k]
        if min(len(tr),len(cal),len(test))<100 or min(d.iloc[tr].faulty.nunique(),d.iloc[cal].faulty.nunique())<2:
            results.append({'regime':int(k),'status':'insufficient rows/classes'});continue
        logging.info('Stress refit: excluding operating regime %s (%s evaluation rows)',k,len(test))
        model=fit_model(d,tr,selected,cfg)
        calibration=Calibrator().fit(model,d.iloc[cal],d.iloc[cal])
        p,mu,sigma=calibration.predict(model,d.iloc[test])
        results.append({'regime':int(k),'fit_rows':len(tr),'calibration_rows':len(cal),
                        'metrics':metrics(d.iloc[test],p,mu,sigma,calibration.threshold)})
    write_json(out/'stress_tests.json',{'note':'Held-out operating regimes are distribution-shift proxies, '
                'not recovered engine identities. Configuration remains frozen.', 'results':results})


def report(out, summary, selected, final_metrics, leaderboard, cfg, n):
    c,r=final_metrics['classification'],final_metrics['regression']
    stress=json.loads((out/'stress_tests.json').read_text())
    stress_rows=[]
    for a in stress.get('results',[]):
        if 'metrics' in a:
            sm=a['metrics'];sc=sm['classification'];sr=sm['regression']
            stress_rows.append(f"| {a['regime']} | {sm['n']} | {sc['accuracy']:.2%} | "
                               f"{sc['recall_faulty']:.2%} | {sr['mae']:.4f} | {sr['interval_90_coverage']:.2%} |")
    stress_table=('| Excluded regime | Holdout rows | Accuracy | Fault recall | Margin MAE | 90% interval coverage |\n'
                  '|---|---:|---:|---:|---:|---:|\n'+'\n'.join(stress_rows)) if stress_rows else 'No completed regime exclusion result in this run.'
    rows='\n'.join(f"| {a['name']} | {a['selection_utility']:.4f} | {a['roc_auc']:.4f} | {a['mae']:.4f} |"
                    for a in leaderboard)
    text=f'''# PHM 2024 experiment report

## Data and interpretation
Original training rows: {summary['rows']:,}. Experiment rows: {n:,}.
Exact repeated feature rows in original data: {summary['duplicate_feature_rows']:,}.
Faulty fraction: {summary['fault_fraction']:.4%}.
IDs are keys only; no time order or asset identifiers are present.
Fault labels are not equivalent to negative torque margins.
PA semantics and units remain ambiguous. Configuration: `{json.dumps(cfg['physics'])}`.
External labels are unavailable, so external test/validation accuracy cannot be measured.

## Selection partition results (not final performance)
Utility = 0.5*(AUC - Brier) - 0.5*normal CRPS / fitting-margin standard deviation.
This utility is NOT the organizer's combined score. All candidates use identical partitions.
Feature ranking and selected-feature ablation use selection rows, never final holdout rows.

| Candidate | Utility (higher better) | ROC AUC | Margin MAE |
|---|---:|---:|---:|
{rows}

Frozen selected configuration: `{json.dumps(selected)}`.

## Locked internal holdout
Rows: {final_metrics['n']:,}. Conventional labels use an F2 threshold selected on calibration rows.

| Metric | Result |
|---|---:|
| Accuracy | {c['accuracy']:.4%} |
| Fault recall | {c['recall_faulty']:.4%} |
| Fault precision | {c['precision_faulty']:.4%} |
| F1 | {c['f1']:.5f} |
| F2 | {c['f2']:.5f} |
| ROC AUC | {c['roc_auc']:.5f} |
| False negatives | {c['false_negatives']} |
| False positives | {c['false_positives']} |
| Brier score | {c['brier']:.5f} |
| Expected calibration error | {c['ece']:.5f} |
| Published classification score | {c['published_classification_score']:.5f} |
| Torque margin MAE (percentage points) | {r['mae']:.5f} |
| Torque margin RMSE (percentage points) | {r['rmse']:.5f} |
| R squared | {r['r2']:.5f} |
| Normal negative log likelihood | {r['normal_nll']:.5f} |
| Normal CRPS | {r['normal_crps']:.5f} |
| 90% interval coverage | {r['interval_90_coverage']:.4%} |
| 90% interval mean width (percentage points) | {r['interval_90_mean_width']:.5f} |

![Evaluation](evaluation.png)

## Operating-regime exclusion results
These are fresh refits of the frozen configuration with a whole operating regime
removed from model fitting and calibration. Evaluation uses final holdout rows in that regime.

{stress_table}

These results show sensitivity to operating-condition shift. They do not establish
unseen-engine performance or guaranteed uncertainty coverage.

## Reliability limits
This is a duplicate-group internal holdout, not an unseen-engine accuracy estimate.
See stress_tests.json for frozen-model operating-regime exclusion experiments.
They approximate operating-condition shift but cannot validate true asset generalization.
Bootstrap intervals group exact duplicates; hidden within-engine dependencies remain unaccounted for.
The model artifact uses train+selection rows, with calibration kept separate. The final holdout is
excluded from fitting, preprocessing, feature selection, calibration and submission generation.
No claimed official regression/combined score: the organizer's regression normalization is unspecified.
Challenge class_conf is score-optimized; calibrated p_faulty is stored separately.
Normal PDFs describe an empirical uncertainty model, not a guaranteed coverage bound under shift.

## Outputs
final_metrics.json, holdout_predictions.csv, bootstrap_intervals.json, stress_tests.json,
selected_model.joblib, selected_spec.json, leaderboard.csv, eda/, submissions/.
External outputs are predictions only. The saved model is the same model evaluated above.
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')


def run(args,cfg):
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty. Use a new run directory to protect prior experiments.')
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log')])
    start=time.time()
    d,external=load_data(args.data_dir)
    write_json(out/'input_manifest.json',manifests(args.data_dir))
    write_json(out/'environment.json',{'python':sys.version,'platform':platform.platform(),
                **current_versions()})
    write_json(out/'config.json',cfg)
    summary=run_eda(d,external,out/'eda',cfg['seed'])
    if args.stage=='eda':return
    if cfg['max_rows'] and len(d)>cfg['max_rows']:
        # Target-independent sample; metadata records the actual experiment population.
        d=d.sample(cfg['max_rows'],random_state=cfg['seed']).reset_index(drop=True)
    parts=split_groups(d,cfg['seed'])
    memberships=[]
    for name,ix in parts.items():
        memberships.append(d.iloc[ix][['id','duplicate_group']].assign(partition=name))
    pd.concat(memberships).to_csv(out/'partition_manifest.csv',index=False)
    train,selection=d.iloc[parts['train']],d.iloc[parts['selection']]
    reference_std=float(train.trq_margin.std())
    leaderboard=[];best_model=None;best_spec=None;best_utility=-np.inf
    def compete(spec,name):
        nonlocal best_model,best_spec,best_utility
        logging.info('Fitting %s on %s rows',name,len(train))
        model=fit_model(d,parts['train'],spec,cfg)
        p,mu,sigma=model.predict(selection)
        m=metrics(selection,p,mu,sigma)
        utility=selection_utility(m,reference_std)
        write_json(out/'candidates'/f'{name}.json',m)
        leaderboard.append({'name':name,'selection_utility':utility,
            'roc_auc':m['classification']['roc_auc'],'brier':m['classification']['brier'],
            'mae':m['regression']['mae'],'crps':m['regression']['normal_crps']})
        pd.DataFrame(leaderboard).sort_values('selection_utility',ascending=False).to_csv(out/'leaderboard.csv',index=False)
        if utility>best_utility:
            best_model,best_spec,best_utility=model,copy.deepcopy(spec),utility
        return model
    for spec in cfg['candidates']:
        compete(spec,f"{spec['mode']}_{spec['architecture']}")
    # Rank the best combined model's engineered features if the best overall is raw.
    parent_spec=copy.deepcopy(best_spec)
    if parent_spec['mode']=='raw':
        parent_spec['mode']='combined'
        parent=compete(parent_spec,f"combined_{parent_spec['architecture']}_selection_parent")
    else:parent=best_model
    keep=feature_selection(parent,train,selection,cfg,out)
    fs_spec={**parent_spec,'keep':keep}
    compete(fs_spec,f"{fs_spec['mode']}_{fs_spec['architecture']}_selected_features")
    selected=copy.deepcopy(best_spec)
    # Freeze BEFORE consulting any holdout labels or stress-test outcomes.
    write_json(out/'selected_spec.json',selected)
    logging.info('Frozen selection: %s',selected)
    build=np.concatenate([parts['train'],parts['selection']])
    final_model=fit_model(d,build,selected,cfg)
    cal=Calibrator().fit(final_model,d.iloc[parts['calibration']],d.iloc[parts['calibration']])
    bundle={'model':final_model,'calibrator':cal,'spec':selected,'config':cfg,
            'training_ids':d.iloc[build].id.to_numpy(),'calibration_ids':d.iloc[parts['calibration']].id.to_numpy()}
    joblib.dump(bundle,out/'selected_model.joblib',compress=3)
    holdout=d.iloc[parts['final']]
    p,mu,sigma=cal.predict(final_model,holdout)
    m=metrics(holdout,p,mu,sigma,cal.threshold)
    write_json(out/'final_metrics.json',m)
    intervals=bootstrap_intervals(holdout,p,mu,sigma,cal.threshold,holdout.duplicate_group,
                                 cfg['bootstrap_repeats'],cfg['seed'])
    write_json(out/'bootstrap_intervals.json',intervals)
    cls,conf=submission_action(p)
    pd.DataFrame({'id':holdout.id,'actual_faulty':holdout.faulty,'actual_margin':holdout.trq_margin,
        'p_faulty':p,'class_f2_threshold':(p>=cal.threshold).astype(int),'class_submission':cls,
        'class_conf':conf,'margin_mean':mu,'margin_sigma':sigma,
        'margin_residual':holdout.trq_margin.to_numpy()-mu}).to_csv(out/'holdout_predictions.csv',index=False)
    # Conditional performance on fixed training quantile bins, not post-hoc thresholds.
    slice_rows=[]
    for c in ['oat','pa','np','ng']:
        edges=np.unique(np.quantile(train[c],[0,.25,.5,.75,1]))
        bins=np.digitize(holdout[c],edges[1:-1])
        for b in np.unique(bins):
            mask=bins==b
            if mask.sum()>=30:
                sm=metrics(holdout.iloc[np.flatnonzero(mask)],p[mask],mu[mask],sigma[mask],cal.threshold)
                slice_rows.append({'feature':c,'bin':int(b),'rows':int(mask.sum()),
                    'recall_faulty':sm['classification']['recall_faulty'],
                    'accuracy':sm['classification']['accuracy'],'mae':sm['regression']['mae'],
                    'interval90_coverage':sm['regression']['interval_90_coverage']})
    pd.DataFrame(slice_rows).to_csv(out/'conditional_performance.csv',index=False)
    plot_evaluation(holdout,p,mu,sigma,m,out)
    stress_tests(d,parts,selected,cfg,out)
    predict_files(bundle,args.data_dir,out/'submissions')
    report(out,summary,selected,m,sorted(leaderboard,key=lambda v:-v['selection_utility']),cfg,len(d))
    write_json(out/'completion.json',{'elapsed_seconds':time.time()-start,'stage':'complete',
                'experiment_rows':len(d),'holdout_rows':len(holdout)})
    logging.info('Complete. Accuracy %.4f, fault recall %.4f, margin MAE %.4f. Report: %s',
        m['classification']['accuracy'],m['classification']['recall_faulty'],m['regression']['mae'],out/'REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='PHM 2024 Cursor experiment harness')
    parser.add_argument('--stage',choices=['run','eda','predict'],default='run')
    parser.add_argument('--data-dir',default='data')
    parser.add_argument('--output-dir',default='runs/quick')
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--model',help='Trusted local selected_model.joblib for predict stage')
    parser.add_argument('--max-rows',type=int,help='Override experiment sample size; 0 means all rows')
    parser.add_argument('--skip-stress',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'));validate_config(cfg)
    if args.max_rows is not None:cfg['max_rows']=args.max_rows or None
    if args.skip_stress:cfg['stress_regimes']=0
    with threadpool_limits(limits=cfg['threads']):
        if args.stage=='predict':
            if not args.model:parser.error('--model required for predict')
            check_model_environment(args.model)
            predict_files(load_trusted(args.model),args.data_dir,args.output_dir)
        else:run(args,cfg)
