"""Typed executable settings alongside immutable historical prose protocols."""
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'configs/runtime_settings_v1.json'

def load_settings(path=PATH):
    value=json.loads(Path(path).read_text())
    types={'schema_version':str,'pca_components':int,'pca_seed':int,'pca_solver':str,
        'logistic_solver':str,'logistic_seed':int,'logistic_tolerance':float,
        'model_c':float,'model_iterations':int,'platt_c':float,'fe_c':float,'fe_iterations':int}
    if set(value)!=set(types) or any(type(value[k]) is not t for k,t in types.items()):
        raise ValueError('runtime settings have missing/unknown fields or incorrect types')
    if value['schema_version']!='runtime_settings_v1' or value['pca_solver']!='auto' or value['logistic_solver']!='lbfgs':
        raise ValueError('unsupported runtime settings')
    import math
    for k in ('pca_components','logistic_tolerance','model_c','model_iterations','platt_c','fe_c','fe_iterations'):
        if not math.isfinite(value[k]) or value[k]<=0:raise ValueError('invalid setting '+k)
    return value

def validate_protocol(p):
    if not isinstance(p,dict) or p.get('schema_version') not in {f'predictor_protocol_v{i}' for i in range(1,6)}:
        raise ValueError('unsupported predictor protocol')
    for section in ('inputs','metrics','controls'):
        if not isinstance(p.get(section),dict):raise ValueError('missing mapping '+section)
    for k in ('splits','scheme'):
        if not isinstance(p['inputs'].get(k),str) or not p['inputs'][k]:raise ValueError('invalid input '+k)
    for k in ('shuffle_permutations','leakage_permutations'):
        if k in p['controls'] and (type(p['controls'][k]) is not int or p['controls'][k]<0):
            raise ValueError('invalid control '+k)
    if not isinstance(p['metrics'].get('bootstrap'),str) or not re.search(r'seed \d+',p['metrics']['bootstrap']):
        raise ValueError('bootstrap requires an explicit integer seed')
    if 'diagnostic_models' in p:
        d=p['diagnostic_models']
        if not isinstance(d,dict) or not isinstance(d.get('models'),list):raise ValueError('invalid diagnostic models')
    return p

SETTINGS=load_settings()

def logistic_options(c,iterations,intercept=True):
    return dict(C=c,max_iter=iterations,fit_intercept=intercept,solver=SETTINGS['logistic_solver'],
                random_state=SETTINGS['logistic_seed'],tol=SETTINGS['logistic_tolerance'])
