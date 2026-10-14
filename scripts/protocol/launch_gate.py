"""Fail closed until an actual RTX 4080 reproduces the T4 Stage 10 set."""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/localization'))
from integrity import read_csv,validate_outcomes,key,sha256,atomic_json
from geo_utils import geodesic_m

def check(directory=None):
    folder=Path(directory) if directory else ROOT/'results/stage10/rtx4080_bench'
    record={'checked_utc':datetime.now(timezone.utc).isoformat(),'pass':False,'required_attempts':180}
    try:
        t4=ROOT/'results/stage10/kaggle_bench'
        env=json.loads((folder/'env.json').read_text())
        reference_env=json.loads((t4/'env.json').read_text())
        if '4080' not in str(env.get('gpu')) or env.get('cuda_available') is not True:
            raise ValueError('actual RTX 4080 CUDA runtime evidence required')
        if 'T4' not in str(reference_env.get('gpu')):
            raise ValueError('reference is not a recorded T4 run')
        for field in ('python','torch'):
            if not env.get(field):raise ValueError('missing runtime '+field)
        if env.get('tf32_matmul') is not False or env.get('tf32_cudnn') is not False:
            raise ValueError('4080 TF32 must be off')
        a=read_csv(folder/'outcomes.csv');b=read_csv(t4/'outcomes.csv')
        ka=validate_outcomes(a);kb=validate_outcomes(b)
        if ka!=kb or len(ka)!=180:raise ValueError('exact 180-attempt key set required')
        refs={key(r):r for r in b};diffs=[];dist=[]
        for r in a:
            ref=refs[key(r)]
            if any(r[f]!=ref[f] for f in ('config_sha256','prereg_sha256','accept_rule')):
                raise ValueError('matcher/protocol identity mismatch')
            if r['outcome_primary']!=ref['outcome_primary']:diffs.append(key(r))
            if r['accepted']==ref['accepted']=='1':
                dist.append(geodesic_m(float(r['pred_lon']),float(r['pred_lat']),float(ref['pred_lon']),float(ref['pred_lat'])))
        record.update(outcome_disagreements=diffs,both_accepted=len(dist),
            positions_within_5m=sum(d<=5 for d in dist),runtime=env,
            input_sha256={str(p):sha256(p) for p in (folder/'outcomes.csv',folder/'env.json',t4/'outcomes.csv',t4/'env.json')})
        record['pass']=not diffs and bool(dist) and sum(d<=5 for d in dist)>=.95*len(dist)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        record['blocker']=str(exc)
    atomic_json(ROOT/'reports/launch_gate_2026-10-14.json',record)
    return record

if __name__=='__main__':
    rec=check(sys.argv[1] if len(sys.argv)>1 else None)
    import torch
    atomic_json(ROOT/'reports/runtime_inventory_2026-10-14.json',{
        'local':{'python':platform.python_version(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
                 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'kaggle':json.loads((ROOT/'results/stage10/kaggle_bench/env.json').read_text()),
        'rtx4080':rec.get('runtime'),'rtx4080_status':'measured' if rec.get('runtime') else 'not available; no version inferred'})
    print(json.dumps(rec,indent=2));raise SystemExit(0 if rec['pass'] else 1)
