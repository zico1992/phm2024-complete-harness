from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAW = ['trq_measured', 'oat', 'mgt', 'pa', 'ias', 'np', 'ng']
ENV = ['oat', 'pa', 'ias']


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False,
                                   default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                          encoding='utf-8')


def load_x(path):
    d = pd.read_csv(path)
    if set(d.columns) != set(['id'] + RAW):
        raise ValueError(f'{path}: expected id plus {RAW}; got {list(d.columns)}')
    if d.id.isna().any() or d.id.duplicated().any():
        raise ValueError(f'{path}: ids must be unique and present')
    if not pd.api.types.is_integer_dtype(d.id) or (d.id < 0).any():
        raise ValueError(f'{path}: ids must be nonnegative integers')
    d[RAW] = d[RAW].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(d[RAW].to_numpy()).all():
        raise ValueError(f'{path}: missing/nonfinite features; define an explicit imputation policy first')
    return d


def load_data(root):
    root = Path(root)
    x = load_x(root / 'X_train.csv')
    y = pd.read_csv(root / 'y_train.csv')
    if set(y.columns) != {'id', 'faulty', 'trq_margin'} or y.id.duplicated().any():
        raise ValueError('Invalid target schema or duplicate ids')
    if set(x.id) != set(y.id):
        raise ValueError('Feature and target ids differ')
    d = x.merge(y, on='id', how='left', validate='one_to_one', sort=False)
    if not set(d.faulty.unique()) <= {0, 1} or not np.isfinite(d.trq_margin).all():
        raise ValueError('Invalid labels')
    if (d.trq_margin <= -100).any() or (d.trq_measured <= 0).any():
        raise ValueError('Cannot reconstruct a positive design torque')
    # Exactly repeated observations share a group, including contradictory labels.
    d['duplicate_group'] = pd.util.hash_pandas_object(d[RAW], index=False).to_numpy()
    return d, {n: load_x(root / f'X_{n}.csv') for n in ['test', 'validation']}


def split_groups(d, seed):
    def split(ix, fraction, offset):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction,
                                     random_state=seed + offset).split(ix, groups=d.iloc[ix].duplicate_group))
        return ix[a], ix[b]
    all_ix = np.arange(len(d))
    dev, final = split(all_ix, .15, 0)
    build, calibration = split(dev, .10 / .85, 1)
    train, selection = split(build, .15 / .75, 2)
    parts = dict(train=train, selection=selection, calibration=calibration, final=final)
    for name, ix in parts.items():
        if d.iloc[ix].faulty.nunique() != 2:
            raise ValueError(f'{name}: both classes required; increase sample size')
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            assert set(d.iloc[parts[a]].duplicate_group).isdisjoint(d.iloc[parts[b]].duplicate_group)
    return parts


def manifests(root):
    result = {}
    for p in sorted(Path(root).glob('*.csv')):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        result[p.name] = {'sha256': h.hexdigest(), 'bytes': p.stat().st_size}
    return result
