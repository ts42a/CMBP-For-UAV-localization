"""Freeze synthetic-only drift calibration; no dataset or labels are read."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SEED=20261014

def calibrate():
    phi=.95
    # 100 m between opportunities, ten increments over a straight 1 km route.
    # AR(1) starts at zero. Sum of innovations has weights (1-phi^k)/(1-phi).
    weights=(1-phi**np.arange(1,11))/(1-phi)
    rng=np.random.default_rng(SEED)
    records={}
    innovations=rng.normal(size=(100000,10,2))
    for name,rate in [('L0',0),('L1',.02),('L2',.05),('L3',.10)]:
        target=rate*1000/2
        sd_w=target/np.sqrt(2*np.sum(weights**2))
        error=(innovations*weights[None,:,None]*sd_w).sum(axis=1)
        measured=float(np.sqrt(np.mean(np.sum(error**2,axis=1))))
        records[name]={'rate':rate,'sd_s':rate/2,'sd_b_rad':rate/2,
            'sd_h_rad':0 if rate==0 else float(np.deg2rad(.5)), 'sd_w_m':float(sd_w),
            'target_ar1_rms_1km_m':target,'simulated_ar1_rms_1km_m':measured}
    return {'schema_version':'drift_calibration_v1','seed':SEED,'simulation_routes':100000,
        'synthetic_step_m':100,'synthetic_steps':10,'phi':phi,
        'definition':'AR(1) radial RMS endpoint contribution over ten 100 m steps; not total drift or a measured sensor model',
        'route_seeds':[91001,91002,91003,91004,91005], 'fidelity_seed':92001,
        'bootstrap_seed':93001,'stress_seed':94001,'levels':records}

if __name__=='__main__':
    target=ROOT/'configs/drift_calibration.json'
    content=json.dumps(calibrate(),indent=2)+'\n'
    if target.exists() and target.read_text()!=content:
        raise SystemExit('Frozen calibration differs; do not overwrite')
    target.write_text(content,encoding='utf-8')
    print(target)
