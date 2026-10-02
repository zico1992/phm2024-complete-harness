import numpy as np
import pandas as pd
import unittest
from phm.data import RAW, split_groups, load_data
from phm.features import FeatureBuilder, physics
from phm.evaluate import classification_score, submission_action, submission
from phm.models import JointModel, Calibrator


def cfg():
    return dict(iterations=12,leaves=7,stack_folds=2,scale_floor=.05,
                physics=dict(temperature_unit='C',pa_meaning='opaque',pa_unit='unknown',
                             ias_unit='unknown',regimes=3))


def toy():
    rng=np.random.default_rng(42);n=240
    d=pd.DataFrame({'id':np.arange(n),'trq_measured':rng.uniform(40,90,n),
        'oat':rng.uniform(0,30,n),'mgt':rng.uniform(500,680,n),
        'pa':rng.uniform(-50,1500,n),'ias':rng.uniform(0,130,n),
        'np':rng.uniform(90,100,n),'ng':rng.uniform(90,100,n)})
    d['trq_margin']=100*(d.trq_measured/(60+.2*d.oat+.002*d.pa)-1)
    d['faulty']=(d.trq_margin<-3).astype(int)
    d['duplicate_group']=pd.util.hash_pandas_object(d[RAW],index=False).to_numpy()
    return d


def check_duplicate_groups_never_cross_partitions(toy):
    duplicate=toy.iloc[:40].copy();duplicate.id+=1000
    d=pd.concat([toy,duplicate],ignore_index=True)
    p=split_groups(d,42)
    for a in p:
        for b in p:
            if a!=b:assert set(d.iloc[p[a]].duplicate_group).isdisjoint(d.iloc[p[b]].duplicate_group)


def check_asymmetric_score():
    s=classification_score([0,0,1,1],[0,1,1,0],[1,1,1,1])
    np.testing.assert_allclose(s,[1,-1,1,-5])
    with np.testing.assert_raises(ValueError):classification_score([1],[0],[1.1])


def check_submission_action_maximizes_expected_score():
    p=np.array([0,.01,.1,.4,.5,.8,1])
    l,c=submission_action(p)
    obtained=(1-p)*classification_score(np.zeros(len(p)),l,c)+p*classification_score(np.ones(len(p)),l,c)
    grid=np.linspace(0,1,1001)
    healthy=(1-2*p[:,None])*grid-4*p[:,None]*grid**11
    faulty=(2*p[:,None]-1)*grid
    optimum=np.maximum(healthy.max(axis=1),faulty.max(axis=1))
    assert np.all(obtained>=optimum-1e-5)


def check_features_batch_independent_and_no_label_columns(toy,cfg):
    builder=FeatureBuilder('combined',cfg['physics']).fit(toy.iloc[:150])
    a=builder.transform(toy.iloc[150:151])
    b=builder.transform(toy.iloc[150:])
    np.testing.assert_allclose(a.iloc[0],b.iloc[0])
    assert not set(['id','faulty','trq_margin','duplicate_group']) & set(a.columns)
    altered=toy.copy();altered['faulty']=1-toy.faulty;altered['trq_margin']=999
    np.testing.assert_allclose(builder.transform(toy),builder.transform(altered))


def check_altitude_feature_is_opt_in(toy,cfg):
    assert 'density_ratio_isa_hypothesis' not in physics(toy,cfg['physics'])
    pc={**cfg['physics'],'pa_meaning':'pressure_altitude','pa_unit':'m','ias_unit':'knots'}
    f=physics(toy,pc)
    assert np.isfinite(f.to_numpy()).all()
    assert (f.density_ratio_isa_hypothesis>0).all()


def check_sea_level_atmosphere_reference(toy,cfg):
    x=toy.iloc[:1].copy();x.oat=15.;x.pa=0.;x.ias=0.
    pc={**cfg['physics'],'pa_meaning':'pressure_altitude','pa_unit':'m','ias_unit':'knots'}
    f=physics(x,pc)
    np.testing.assert_allclose(f.density_ratio_isa_hypothesis.iloc[0],1.)
    np.testing.assert_allclose(f.density_altitude_m_hypothesis.iloc[0],0.,atol=1e-8)
    assert f.mach_low_mach_proxy.iloc[0]==0


def check_submission_preserves_non_contiguous_ids():
    s=submission([4,9],np.array([.1,.9]),np.array([-2.,3.]),np.array([.4,.6]))
    assert list(s)==['4','9']
    assert s['4']['pdf_args']['scale']>0
    with np.testing.assert_raises(ValueError):submission([0],np.array([.5]),np.array([1]),np.array([0]))


def check_ids_join_not_row_position(toy,tmp_path):
    toy[['id']+RAW].to_csv(tmp_path/'X_train.csv',index=False)
    toy[['id','faulty','trq_margin']].sample(frac=1,random_state=1).to_csv(tmp_path/'y_train.csv',index=False)
    for name in ['test','validation']:toy[['id']+RAW].iloc[:10].to_csv(tmp_path/f'X_{name}.csv',index=False)
    d,_=load_data(tmp_path)
    np.testing.assert_allclose(d.trq_margin,toy.trq_margin)


def check_joint_fit_predict_and_artifact_roundtrip(toy,cfg,architecture,tmp_path):
    from threadpoolctl import threadpool_limits
    import joblib
    with threadpool_limits(limits=2):
        model=JointModel('combined',architecture,cfg).fit(toy.iloc[:160],toy.iloc[:160],toy.duplicate_group.iloc[:160])
        cal=Calibrator().fit(model,toy.iloc[160:200],toy.iloc[160:200])
        a=cal.predict(model,toy.iloc[200:])
        assert np.isfinite(np.column_stack(a)).all()
        assert (a[2]>0).all()
        joblib.dump((model,cal),tmp_path/'artifact.joblib')
        m,c=joblib.load(tmp_path/'artifact.joblib')
        b=c.predict(m,toy.iloc[200:])
        np.testing.assert_allclose(a,b)


class ContractTests(unittest.TestCase):
    def test_group_splits(self):
        check_duplicate_groups_never_cross_partitions(toy())
    def test_scoring(self):
        check_asymmetric_score()
    def test_bayes_action(self):
        check_submission_action_maximizes_expected_score()
    def test_feature_independence(self):
        check_features_batch_independent_and_no_label_columns(toy(),cfg())
    def test_altitude_gate(self):
        check_altitude_feature_is_opt_in(toy(),cfg())
    def test_atmosphere(self):
        check_sea_level_atmosphere_reference(toy(),cfg())
    def test_ids(self):
        check_submission_preserves_non_contiguous_ids()
    def test_join(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as root:
            check_ids_join_not_row_position(toy(),Path(root))
    def test_joint_models(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        for architecture in ['direct','target_torque','linear']:
            with self.subTest(architecture=architecture), TemporaryDirectory() as root:
                check_joint_fit_predict_and_artifact_roundtrip(toy(),cfg(),architecture,Path(root))


if __name__=='__main__':
    unittest.main()
