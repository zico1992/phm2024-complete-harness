import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, fbeta_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score)


def classification_score(y, label, confidence):
    y, label, c = np.asarray(y), np.asarray(label), np.asarray(confidence)
    if not np.isin(label, [0, 1]).all() or not np.isfinite(c).all() or ((c < 0) | (c > 1)).any():
        raise ValueError('Invalid submitted class/confidence')
    score = np.where(y == label, c, -c)
    fn = (y == 1) & (label == 0)
    score[fn] = -c[fn] - 4*c[fn]**11
    return score


def submission_action(p):
    """Bayes action for published asymmetric classification score.

    class_conf is a decision confidence, NOT the probability of the chosen class.
    Keep calibrated p_faulty as a separate output for reliability analysis.
    E[S|healthy]=(1-2p)c-4pc^11; E[S|faulty]=(2p-1)c.
    """
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    healthy_c = np.minimum(1., (np.maximum(1-2*p, 0) / np.maximum(44*p, 1e-15)) ** .1)
    healthy_score = (1-2*p)*healthy_c - 4*p*healthy_c**11
    faulty_score = np.maximum(2*p-1, 0)
    label = (faulty_score > healthy_score).astype(int)
    confidence = np.where(label == 1, 1., healthy_c)
    # At complete uncertainty, neither action has positive expected reward.
    confidence = np.where(p == .5, 0., confidence)
    return label, confidence


def reliability(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    bin_id = np.minimum((p*bins).astype(int), bins-1)
    rows = []
    for b in range(bins):
        mask = bin_id == b
        if mask.any():
            rows.append({'bin': b, 'count': int(mask.sum()),
                         'predicted_probability': float(p[mask].mean()),
                         'observed_fault_fraction': float(y[mask].mean())})
    ece = sum(r['count']*abs(r['predicted_probability']-r['observed_fault_fraction']) for r in rows) / len(y)
    return rows, float(ece)


def metrics(y, p, mu, sigma, threshold=.5):
    target = y.trq_margin.to_numpy()
    labels = (p >= threshold).astype(int)
    submitted, confidence = submission_action(p)
    cm = confusion_matrix(y.faulty, labels, labels=[0, 1])
    rows, ece = reliability(y.faulty, p)
    z = (target-mu)/sigma
    # CRPS of normal predictive distributions; smaller is better.
    crps = sigma*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))
    class_m = dict(accuracy=float(accuracy_score(y.faulty, labels)),
        # Average recall over truth classes present; meaningful for one-class slices too.
        balanced_accuracy=float(np.mean([cm[k,k]/cm[k].sum() for k in range(2) if cm[k].sum()>0])),
        precision_faulty=float(precision_score(y.faulty, labels, zero_division=0)),
        recall_faulty=float(recall_score(y.faulty, labels, zero_division=0)),
        f1=float(f1_score(y.faulty, labels, zero_division=0)),
        f2=float(fbeta_score(y.faulty, labels, beta=2, zero_division=0)),
        roc_auc=float(roc_auc_score(y.faulty, p)) if y.faulty.nunique()==2 else None,
        pr_auc=float(average_precision_score(y.faulty, p)) if y.faulty.nunique()==2 else None,
        brier=float(brier_score_loss(y.faulty, p)),
        log_loss=float(log_loss(y.faulty, p, labels=[0,1])), ece=ece,
        false_negatives=int(cm[1,0]), false_positives=int(cm[0,1]),
        confusion_matrix=cm.tolist(), threshold=float(threshold),
        published_classification_score=float(classification_score(y.faulty, submitted, confidence).mean()))
    reg_m = dict(mae=float(mean_absolute_error(target, mu)),
        rmse=float(np.sqrt(mean_squared_error(target, mu))), r2=float(r2_score(target, mu)) if len(target)>=2 else None,
        normal_nll=float(-norm.logpdf(target, loc=mu, scale=sigma).mean()),
        normal_crps=float(crps.mean()), residual_bias=float((target-mu).mean()),
        within_0_5_margin_points=float((np.abs(target-mu)<=.5).mean()),
        interval_90_coverage=float((np.abs(z)<=norm.ppf(.95)).mean()),
        interval_90_mean_width=float((2*norm.ppf(.95)*sigma).mean()),
        interval_95_coverage=float((np.abs(z)<=norm.ppf(.975)).mean()),
        mean_raw_pdf_at_truth=float(norm.pdf(target, loc=mu, scale=sigma).mean()))
    return {'classification': class_m, 'regression': reg_m, 'reliability': rows,
            'n': len(y), 'official_combined_score': None,
            'score_note': 'Classification formula reproduced. Raw PDF, NLL and CRPS are diagnostics; '
                          'organizer regression normalization is unspecified, so no official combined score is claimed.'}


def selection_utility(m, margin_reference_std):
    # Explicit, configurable in code, NOT the official competition score.
    # Classification AUC penalizes poor ranking and Brier poor probability quality.
    # Regression scale-free CRPS balances both targets.
    c, r = m['classification'], m['regression']
    return float(.5*(c['roc_auc']-c['brier']) - .5*r['normal_crps']/max(margin_reference_std, 1e-6))


def bootstrap_intervals(y, p, mu, sigma, threshold, groups, repeats, seed):
    rng = np.random.default_rng(seed)
    codes, unique = pd.factorize(groups)
    members = [np.flatnonzero(codes == i) for i in range(len(unique))]
    vals = {'accuracy': [], 'recall_faulty': [], 'mae': [], 'classification_score': []}
    lab = p >= threshold
    sub, conf = submission_action(p)
    for _ in range(repeats):
        ix = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        yt = y.iloc[ix]
        vals['accuracy'].append(float(accuracy_score(yt.faulty, lab[ix])))
        vals['recall_faulty'].append(float(recall_score(yt.faulty, lab[ix], zero_division=0)))
        vals['mae'].append(float(mean_absolute_error(yt.trq_margin, mu[ix])))
        vals['classification_score'].append(float(classification_score(yt.faulty, sub[ix], conf[ix]).mean()))
    return {k: {'low_95': float(np.quantile(v,.025)), 'high_95': float(np.quantile(v,.975))}
            for k,v in vals.items()} if repeats else {}


def submission(ids, p, mu, sigma):
    label, confidence = submission_action(p)
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma<=0).any():
        raise ValueError('Invalid predictive distribution')
    return {str(int(i)): {'class': int(l), 'class_conf': float(c), 'pdf_type':'norm',
                         'pdf_args': {'loc':float(m), 'scale':float(s)}}
            for i,l,c,m,s in zip(ids,label,confidence,mu,sigma)}
