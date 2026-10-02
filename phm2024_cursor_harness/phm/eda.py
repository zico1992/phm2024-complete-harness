from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import RAW, write_json


def run_eda(d, external, out, seed):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    summary = {'rows': len(d), 'fault_fraction': float(d.faulty.mean()),
        'missing_values': d[RAW+['faulty','trq_margin']].isna().sum().to_dict(),
        'duplicate_feature_rows': int(d.duplicate_group.duplicated().sum()),
        'conflicting_class_groups': int((d.groupby('duplicate_group').faulty.nunique()>1).sum()),
        'negative_margin_is_not_fault_label': pd.crosstab(d.faulty,d.trq_margin<0).to_dict(),
        'feature_definitions': {'pa': 'Ambiguous: official page/forum power available; papers pressure altitude',
                                'np': 'Official net power; values also resemble speed percentage. No shaft-speed claim.'},
        'external_rows': {k:len(v) for k,v in external.items()},
        'np_less_than_ng_fraction': {'train':float((d.np<d.ng).mean()),
              **{k:float((v.np<v.ng).mean()) for k,v in external.items()}}}
    write_json(out/'summary.json',summary)
    d[RAW+['trq_margin']].describe(percentiles=[.01,.05,.5,.95,.99]).to_csv(out/'descriptive_statistics.csv')
    d[RAW+['faulty','trq_margin']].corr().to_csv(out/'correlations.csv')
    d.groupby('faulty').trq_margin.agg(['count','min','median','max']).to_csv(out/'margin_by_class.csv')
    # All external diagnostics are descriptive, never used to fit preprocessing or choose models.
    sample = d.sample(min(30000,len(d)), random_state=seed)
    shifts = []
    for name, x in external.items():
        for c in RAW:
            a,b = sample[c],x[c]
            ks = ks_2samp(a,b)
            shifts.append({'dataset':name,'feature':c,'ks_statistic':float(ks.statistic),
                'ks_pvalue':float(ks.pvalue),
                'wasserstein_over_train_std':float(wasserstein_distance(a,b)/max(a.std(),1e-6)),
                'outside_train_full_range_fraction':float(((b<d[c].min())|(b>d[c].max())).mean())})
    pd.DataFrame(shifts).to_csv(out/'external_shift.csv',index=False)
    fig,axs=plt.subplots(2,4,figsize=(14,7))
    for ax,c in zip(axs.flat,RAW+['trq_margin']):
        for label in [0,1]:
            ax.hist(sample.loc[sample.faulty==label,c],bins=45,density=True,alpha=.5,label=f'faulty={label}')
        ax.set_title(c)
    axs.flat[0].legend();fig.tight_layout();fig.savefig(out/'class_distributions.png',dpi=130);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,c in zip(axs,['oat','pa','np']):
        ax.hist(sample[c],bins=50,density=True,histtype='step',label='train')
        for name,x in external.items():
            ax.hist(x[c],bins=50,density=True,histtype='step',label=name)
        ax.set_title(c)
    axs[0].legend();fig.tight_layout();fig.savefig(out/'external_distributions.png',dpi=130);plt.close(fig)
    return summary


def plot_evaluation(y,p,mu,sigma,m,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    ax=axs[0,0];cm=np.array(m['classification']['confusion_matrix'])
    ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set(xlabel='Predicted',ylabel='Actual',xticks=[0,1],yticks=[0,1],title='Nominal=0, faulty=1')
    r=pd.DataFrame(m['reliability']);ax=axs[0,1]
    ax.plot([0,1],[0,1],'--',color='gray');ax.plot(r.predicted_probability,r.observed_fault_fraction,'o-')
    ax.set(xlabel='Calibrated probability faulty',ylabel='Observed faulty fraction',title='Reliability')
    ax=axs[1,0];ix=np.linspace(0,len(y)-1,min(len(y),5000)).astype(int)
    ax.scatter(y.trq_margin.to_numpy()[ix],mu[ix],s=3,alpha=.3)
    lo,hi=float(y.trq_margin.min()),float(y.trq_margin.max());ax.plot([lo,hi],[lo,hi],'--',color='gray')
    ax.set(xlabel='Actual torque margin (%)',ylabel='Predicted torque margin (%)',title='Regression')
    ax=axs[1,1];ax.hist((y.trq_margin.to_numpy()-mu)/sigma,bins=60,density=True)
    ax.set(xlabel='Standardized residual',title='Distribution check')
    fig.tight_layout();fig.savefig(out/'evaluation.png',dpi=140);plt.close(fig)
