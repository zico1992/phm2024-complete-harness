"""JSON-only runtime preflight. Never deserialize a model to discover its version."""
import json
import platform
import re
import warnings
from pathlib import Path
from importlib.metadata import version
import joblib
from sklearn.exceptions import InconsistentVersionWarning

PACKAGES={'sklearn':'scikit-learn','numpy':'numpy','pandas':'pandas','scipy':'scipy','joblib':'joblib','threadpoolctl':'threadpoolctl'}
REQUIRED=['sklearn','numpy','pandas']


def current_versions():
    return {'python':platform.python_version(),**{key:version(package) for key,package in PACKAGES.items()}}


def python_minor(value):
    match=re.search(r'(\d+)\.(\d+)',str(value))
    if not match: raise ValueError(f'Invalid Python version metadata: {value}')
    return match.group(0)


def check_versions(expected,source='baseline metadata'):
    actual=current_versions()
    missing=[key for key in REQUIRED if not expected.get(key)]
    if missing: raise ValueError(f'{source}: missing runtime versions {missing}; recover the original environment metadata')
    mismatches=[]
    for key in PACKAGES:
        if expected.get(key) and expected[key]!=actual[key]:
            mismatches.append(f'{key}: expected {expected[key]}, installed {actual[key]}')
    if expected.get('python') and python_minor(expected['python'])!=python_minor(actual['python']):
        mismatches.append(f"python: expected {python_minor(expected['python'])}, installed {python_minor(actual['python'])}")
    if mismatches:
        raise ValueError('Runtime preflight failed BEFORE model loading ('+source+'): '+ '; '.join(mismatches)+'. Activate/recreate the original training environment; do not upgrade the frozen artifact.')
    return {'status':'passed','source':source,'expected':expected,'installed':actual,
            'checked_packages':[key for key in PACKAGES if expected.get(key)],
            'unrecorded_packages':[key for key in PACKAGES if not expected.get(key)],
            'python_policy':'major.minor must match when recorded; patch differences permitted',
            'model_loaded':False}


def check_model_environment(model_path,environment=None):
    path=Path(environment) if environment else Path(model_path).parent/'environment.json'
    if not path.exists():
        raise ValueError(f'Runtime metadata missing: {path}. Supply --environment pointing to the original training environment.json; never infer versions by unpickling.')
    return check_versions(json.loads(path.read_text()),str(path))


def check_baseline_environment(root):
    path=Path(root)/'manifest.json'
    manifest=json.loads(path.read_text())
    return check_versions(manifest['versions'],str(path))


def load_trusted(path):
    # Legacy metadata can omit versions; estimator warnings must still stop loading.
    with warnings.catch_warnings():
        warnings.simplefilter('error',InconsistentVersionWarning)
        try: return joblib.load(path)
        except InconsistentVersionWarning as exc:
            raise ValueError('Estimator version disagrees with metadata. Recover the original environment; do not suppress version warnings.') from exc
