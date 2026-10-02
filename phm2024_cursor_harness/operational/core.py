from pathlib import Path
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from phm.data import RAW, ENV, load_x, write_json
from phm.evaluate import metrics
from .runtime import current_versions, check_model_environment, check_baseline_environment, load_trusted


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def new_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output must be new or empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def hashes(frame):
    return pd.util.hash_pandas_object(frame[RAW].astype('float64'), index=False).to_numpy()


def freeze(model_path, training_csv, output, name, reference_limit=50000, environment=None):
    """Build monitoring sidecar on fitting/calibration IDs only; never refit baseline."""
    if reference_limit < 100:
        raise ValueError('reference-limit must be >=100')
    model_path = Path(model_path)
    before = sha(model_path)
    preflight = check_model_environment(model_path,environment)
    bundle = load_trusted(model_path)
    input_manifest = model_path.parent/'input_manifest.json'
    if input_manifest.exists():
        expected = json.loads(input_manifest.read_text()).get('X_train.csv',{}).get('sha256')
        if expected and sha(training_csv) != expected:
            raise ValueError('Training source differs from the baseline input manifest')
    x = load_x(training_csv).set_index('id', drop=False)
    tr = x.loc[bundle['training_ids']].reset_index(drop=True)
    ca = x.loc[bundle['calibration_ids']].reset_index(drop=True)
    if set(tr.id) & set(ca.id) or set(hashes(tr)) & set(hashes(ca)):
        raise ValueError('Training/calibration duplicate leakage')
    guard = {'raw_min': tr[RAW].min().to_dict(), 'raw_max': tr[RAW].max().to_dict(),
             'seen_hashes': np.unique(np.concatenate([hashes(tr), hashes(ca)])), 'spaces': {}}
    rng = np.random.default_rng(42)
    unique = tr.drop_duplicates(RAW)
    ref = unique.iloc[rng.choice(len(unique), min(reference_limit,len(unique)),replace=False)]
    with threadpool_limits(limits=4):
        for label, cols in [('joint', RAW), ('environment', ENV)]:
            scaler = RobustScaler().fit(tr[cols])
            nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=1).fit(scaler.transform(ref[cols]))
            distances = nn.kneighbors(scaler.transform(ca[cols]))[0][:,0]
            guard['spaces'][label] = {'columns':cols, 'scaler':scaler, 'nn':nn,
                                      'threshold':float(np.quantile(distances,.99)), 'reference_ids':ref.id.to_numpy()}
        p, mu, sigma = bundle['calibrator'].predict(bundle['model'],ca)
    guard['wide_interval_threshold'] = float(np.quantile(2*norm.ppf(.95)*sigma,.99))
    # Coverage evidence is monitoring context, not a new calibration or coverage guarantee.
    regime = regime_ids(bundle, ca)
    ypath = Path(training_csv).with_name('y_train.csv')
    evidence = []
    if ypath.exists():
        if input_manifest.exists():
            expected_y = json.loads(input_manifest.read_text()).get('y_train.csv',{}).get('sha256')
            if expected_y and sha(ypath) != expected_y:
                raise ValueError('Calibration labels differ from baseline source manifest')
        y = pd.read_csv(ypath).set_index('id')
        yy = y.loc[ca.id, 'trq_margin'].to_numpy()
        for r in np.unique(regime):
            mask = regime == r
            evidence.append({'regime':int(r),'n':int(mask.sum()),
                             'coverage_90':float((np.abs(yy[mask]-mu[mask])<=norm.ppf(.95)*sigma[mask]).mean())})
    guard['coverage_evidence'] = evidence
    out = new_dir(output)
    shutil.copyfile(model_path, out/'baseline.joblib')
    joblib.dump(guard,out/'reliability.joblib',compress=3)
    policy = {'version':'pilot-v1','support_quantile':.99,'coverage_min_n':100,
              'coverage_warning_below':.85,'threshold_status':'provisional, engineering approval required',
              'decision':'review recommendations only; no maintenance or flight control commands'}
    write_json(out/'policy.json',policy)
    manifest = {'name':name,'created_at':now(),'baseline_sha256':before,
                'reliability_sha256':sha(out/'reliability.joblib'),'policy_sha256':sha(out/'policy.json'),
                'training_csv_sha256':sha(training_csv),'training_n':len(tr),'calibration_n':len(ca),
                'reference_n':len(ref),'spec':bundle['spec'],'physics':bundle['config']['physics'],
                'threshold':float(bundle['calibrator'].threshold),
                'versions':current_versions(),'training_runtime_preflight':preflight,
                'coverage_evidence':evidence,'baseline_engine_ids':'unavailable; unseen-engine independence cannot be verified'}
    write_json(out/'manifest.json',manifest)
    if before != sha(model_path) or before != sha(out/'baseline.joblib'):
        raise ValueError('Baseline hash changed')
    return manifest


def regime_ids(bundle, x):
    b = bundle['model'].builder
    if hasattr(b,'cluster'):
        return b.cluster.predict(b.scaler.transform(x[ENV]))
    return np.full(len(x),-1,dtype=int)


class Predictor:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.assert_frozen()
        self.runtime = check_baseline_environment(self.root)
        self.bundle = load_trusted(self.root/'baseline.joblib')
        self.guard = load_trusted(self.root/'reliability.joblib')
        self.policy = json.loads((self.root/'policy.json').read_text())

    def assert_frozen(self):
        for file, key in [('baseline.joblib','baseline_sha256'),('reliability.joblib','reliability_sha256'),('policy.json','policy_sha256')]:
            if sha(self.root/file) != self.manifest[key]:
                raise ValueError(f'Frozen artifact changed: {file}')

    def predict(self, frame, mode='replay'):
        if mode not in ['replay','shadow']:
            raise ValueError('mode must be replay or shadow')
        if not len(frame) or len(frame)>200000:
            raise ValueError('Batch must contain 1..200000 observations')
        if 'id' not in frame or frame.id.isna().any() or frame.id.astype(str).duplicated().any():
            raise ValueError('Unique, present observation IDs required')
        if set(RAW)-set(frame):
            raise ValueError(f'Missing measurements: {set(RAW)-set(frame)}')
        self.assert_frozen()
        d = frame.copy().reset_index(drop=True)
        d['id'] = d.id.astype(str)
        for c in RAW:
            d[c] = pd.to_numeric(d[c],errors='coerce')
        a = d[RAW].to_numpy(dtype=float)
        valid = np.isfinite(a).all(axis=1) & (d.trq_measured>0) & (d.ng>0) & (d.np>0) & (d.ias>=0)
        if self.manifest['physics']['temperature_unit']=='C':
            valid &= (d.oat>-273.15) & (d.mgt>-273.15)
        if self.manifest['physics']['pa_meaning']=='pressure_altitude':
            factor = .3048 if self.manifest['physics']['pa_unit']=='ft' else 1
            valid &= (d.pa*factor>=-2000)&(d.pa*factor<=11000)
        status = d.get('sensor_quality', pd.Series('unknown',index=d.index)).fillna('unknown').astype(str)
        valid &= ~status.isin(['invalid','failed'])
        result = d[['id']+RAW+[c for c in ['engine_id','flight_id','timestamp_utc','sensor_quality'] if c in d]].copy()
        result['valid_input'] = valid
        for c in ['p_faulty','margin_mean','margin_sigma','margin_low_90','margin_high_90','joint_distance','environment_distance']:
            result[c] = np.nan
        result['fault_flag'] = pd.Series(pd.NA,index=d.index,dtype='Int64')
        result['regime'] = -1
        result['training_or_calibration_duplicate'] = False
        reasons = [[] for _ in range(len(d))]
        for i in np.flatnonzero(~valid):
            reasons[i].append('invalid_measurement_or_sensor_quality')
        ix = np.flatnonzero(valid)
        if len(ix):
            good = d.iloc[ix]
            with threadpool_limits(limits=4):
                p, mu, sigma = self.bundle['calibrator'].predict(self.bundle['model'],good)
                if not (np.isfinite(p).all() and np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma>0).all()):
                    raise ValueError('Baseline produced invalid predictions')
                for c, values in [('p_faulty',p),('margin_mean',mu),('margin_sigma',sigma),('margin_low_90',mu-norm.ppf(.95)*sigma),('margin_high_90',mu+norm.ppf(.95)*sigma)]:
                    result.loc[ix,c] = values
                result.loc[ix,'fault_flag'] = (p>=self.manifest['threshold']).astype(int)
                rs = regime_ids(self.bundle,good)
                result.loc[ix,'regime'] = rs
                result.loc[ix,'training_or_calibration_duplicate'] = np.isin(hashes(good),self.guard['seen_hashes'])
                for label, space in self.guard['spaces'].items():
                    dist = space['nn'].kneighbors(space['scaler'].transform(good[space['columns']]))[0][:,0]
                    result.loc[ix,label+'_distance'] = dist
                    for i in ix[dist>space['threshold']]:
                        reasons[i].append(label+'_coverage_gap')
            for j,i in enumerate(ix):
                outside = [c for c in RAW if good.iloc[j][c]<self.guard['raw_min'][c] or good.iloc[j][c]>self.guard['raw_max'][c]]
                if outside:
                    reasons[i].append('outside_training_range:'+','.join(outside))
                if 2*norm.ppf(.95)*sigma[j]>self.guard['wide_interval_threshold']:
                    reasons[i].append('wide_margin_interval')
                ev = next((e for e in self.guard['coverage_evidence'] if e['regime']==int(rs[j])),None)
                if ev is None or ev['n']<self.policy['coverage_min_n']:
                    reasons[i].append('insufficient_conditional_coverage_evidence')
                elif ev['coverage_90']<self.policy['coverage_warning_below']:
                    reasons[i].append('weak_conditional_interval_coverage')
        if mode=='shadow':
            for i in range(len(d)):
                if status.iloc[i] not in ['ok','verified']:
                    reasons[i].append('sensor_quality_unverified')
        result['reliability_reasons'] = [';'.join(v) for v in reasons]
        result['reliability_status'] = ['rejected' if not valid.iloc[i] else ('review' if reasons[i] else 'within_reference') for i in range(len(d))]
        result['review_recommended'] = (result.fault_flag.fillna(0)==1)|(result.reliability_status!='within_reference')
        result['review_reason'] = ['fault_probability_threshold;'+r if f==1 else r for f,r in zip(result.fault_flag.fillna(0),result.reliability_reasons)]
        result['mode'] = mode
        result['baseline_sha256'] = self.manifest['baseline_sha256']
        self.assert_frozen()
        return result


def operational_input(path, contract_path, predictor):
    contract = json.loads(Path(contract_path).read_text())
    if contract.get('status')!='confirmed' or not contract.get('confirmed_by') or not contract.get('confirmed_at'):
        raise ValueError('Operational data contract must be confirmed by the data owner')
    if contract.get('baseline_sha256') != predictor.manifest['baseline_sha256']:
        raise ValueError('Contract must identify this frozen baseline SHA256')
    if contract.get('baseline_physics') != predictor.manifest['physics']:
        raise ValueError('Operational semantics must match the frozen baseline assumptions')
    d = pd.read_csv(path,dtype={'observation_id':str,'engine_id':str,'flight_id':str})
    required = {'observation_id','engine_id','flight_id','timestamp_utc','sensor_quality'}|set(RAW)
    if required-set(d):
        raise ValueError(f'Missing operational columns: {required-set(d)}')
    for c in ['observation_id','engine_id','flight_id']:
        if d[c].isna().any() or d[c].str.strip().eq('').any():
            raise ValueError(f'{c} must be present')
    if d.observation_id.duplicated().any():
        raise ValueError('Duplicate observation_id')
    stamps = pd.to_datetime(d.timestamp_utc,utc=True,errors='raise',format='ISO8601')
    if not d.timestamp_utc.astype(str).str.contains(r'(?:Z|[+-]\d\d:\d\d)$',regex=True).all():
        raise ValueError('timestamp_utc requires an explicit UTC offset')
    if stamps.isna().any() or (stamps>pd.Timestamp.now(tz='UTC')).any():
        raise ValueError('Missing or future timestamps')
    for c in RAW:
        rule = contract.get('measurements',{}).get(c,{})
        if not all(rule.get(k) for k in ['meaning','source_unit','baseline_unit']):
            raise ValueError(f'{c}: confirm meaning and units')
        if any('CONFIRM' in str(rule[k]).upper() for k in ['meaning','source_unit','baseline_unit']):
            raise ValueError(f'{c}: unresolved measurement definition')
        if c in ['oat','mgt'] and rule['baseline_unit'] != predictor.manifest['physics']['temperature_unit']:
            raise ValueError(f'{c}: baseline temperature unit must match frozen physics config')
        scale, offset = float(rule.get('scale',1)),float(rule.get('offset',0))
        if not np.isfinite([scale,offset]).all() or scale<=0:
            raise ValueError(f'{c}: invalid unit conversion')
        d[c] = pd.to_numeric(d[c],errors='coerce')*scale+offset
    if not d.sensor_quality.isin(['ok','verified','unknown','invalid','failed']).all():
        raise ValueError('Unknown sensor_quality enum')
    d['id'] = d.observation_id
    d['timestamp_utc'] = stamps.astype(str)
    return d.sort_values(['timestamp_utc','engine_id','id']).reset_index(drop=True)


def labelled_join(predictions, label_path, mode):
    y = pd.read_csv(label_path,dtype={'observation_id':str,'id':str})
    if mode=='shadow':
        required = {'observation_id','faulty','trq_margin','label_source','label_timestamp_utc'}
        if required-set(y):
            raise ValueError(f'Missing label fields: {required-set(y)}')
        if y.label_source.isna().any() or y.label_source.astype(str).str.strip().eq('').any():
            raise ValueError('Confirmed label provenance required')
        y['id'] = y.observation_id
        if not y.label_timestamp_utc.astype(str).str.contains(r'(?:Z|[+-]\d\d:\d\d)$',regex=True).all():
            raise ValueError('Label timestamps require an explicit UTC offset')
        times = pd.to_datetime(y.label_timestamp_utc,utc=True,errors='raise',format='ISO8601')
        if times.isna().any() or (times>pd.Timestamp.now(tz='UTC')).any():
            raise ValueError('Missing or future label timestamps')
    if not {'id','faulty','trq_margin'}<=set(y) or y.id.isna().any() or y.id.duplicated().any():
        raise ValueError('Labels require unique IDs, faulty and trq_margin')
    if not y.faulty.isin([0,1]).all() or not np.isfinite(pd.to_numeric(y.trq_margin,errors='coerce')).all():
        raise ValueError('Invalid labels')
    if set(y.id)-set(predictions.id):
        raise ValueError('Label IDs absent from this prediction run')
    cols = ['id','faulty','trq_margin']+[c for c in ['label_source','label_timestamp_utc'] if c in y]
    joined = predictions.merge(y[cols],on='id',how='left',validate='one_to_one')
    if mode=='shadow':
        known = joined.faulty.notna()
        if (pd.to_datetime(joined.loc[known,'label_timestamp_utc'],utc=True,format='ISO8601')<pd.to_datetime(joined.loc[known,'timestamp_utc'],utc=True,format='ISO8601')).any():
            raise ValueError('Label confirmation precedes observation')
    return joined


def evaluate_shadow(d, threshold):
    known = d.faulty.notna() if 'faulty' in d else pd.Series(False,index=d.index)
    eligible = known & d.valid_input & ~d.training_or_calibration_duplicate
    report = {'observations':len(d),'labelled':int(known.sum()),'invalid':int((~d.valid_input).sum()),
              'seen_duplicate_labelled':int((known&d.training_or_calibration_duplicate).sum()),
              'evaluated_independent_feature_rows':int(eligible.sum()),
              'independence_note':'Exact fitting/calibration feature duplicates excluded. Engine independence is unknown for challenge baseline.',
              'metrics':None,'slices':[],'flight_events':None,
              'lead_time':None,'lead_time_note':'Requires confirmed fault onset/event IDs; label confirmation is not onset.',
              'acceptance':'NOT ASSESSED: stakeholder acceptance criteria are not supplied'}
    e = d.loc[eligible].copy()
    if not len(e):
        return clean_numbers(report)
    report['metrics'] = metrics(e,e.p_faulty.to_numpy(),e.margin_mean.to_numpy(),e.margin_sigma.to_numpy(),threshold)
    for col in ['regime','reliability_status','engine_id']:
        if col in e:
            for value,g in e.groupby(col):
                m = metrics(g,g.p_faulty.to_numpy(),g.margin_mean.to_numpy(),g.margin_sigma.to_numpy(),threshold)
                report['slices'].append({'dimension':col,'value':str(value),'n':len(g),
                    'false_negatives':m['classification']['false_negatives'],'false_positives':m['classification']['false_positives'],
                    'recall':m['classification']['recall_faulty'],'mae':m['regression']['mae'],
                    'coverage_90':m['regression']['interval_90_coverage']})
    if {'engine_id','flight_id'}<=set(d):
        # Fully labelled flights only. Include duplicates/invalid in completeness checks.
        rows = []
        skipped = 0
        for key,g in d.groupby(['engine_id','flight_id']):
            if not (g.faulty.notna().all() and g.valid_input.all() and (~g.training_or_calibration_duplicate).all()):
                skipped+=1
                continue
            truth = bool((g.faulty==1).any())
            alert = bool((g.fault_flag==1).any())
            rows.append((truth,alert))
        if rows:
            truth,alert = np.asarray(rows).T
            report['flight_events'] = {'fully_labelled_flights':len(rows),'excluded_incomplete_or_seen_flights':skipped,
                'missed_fault_positive_flights':int((truth&~alert).sum()),'false_alert_flights':int((~truth&alert).sum()),
                'false_alerts_per_100_healthy_flights':float(100*(~truth&alert).sum()/(~truth).sum()) if (~truth).any() else None,
                'definition':'Any threshold-crossing observation per engine/flight; persistence and onset not inferred'}
    return clean_numbers(report)


def clean_numbers(value):
    if isinstance(value,dict):
        return {k:clean_numbers(v) for k,v in value.items()}
    if isinstance(value,list):
        return [clean_numbers(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):
        return None
    return value
