"""Matched source-region real/synthetic fits evaluated only on real held-out labels."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/predictor'),str(ROOT/'scripts/localization')]
from evaluate_predictors import Data,metrics
from fitted_predictor import FittedPredictor
from paired_contrasts import contrast
from integrity import atomic_json,sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--real',required=True);ap.add_argument('--synthetic',required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    real,syn=Data(Path(args.real)),Data(Path(args.synthetic))
    keys=['region','filename','radius_m','offset_k']
    if not real.df[keys].equals(syn.df[keys]):raise ValueError('real/synthetic sample budgets or ordered keys differ')
    if set(real.df.variant)!= {'prior'} or set(syn.df.variant)!= {'syn_prior'}:raise ValueError('wrong query-source variants')
    scheme=json.loads((ROOT/'configs/splits_v1.json').read_text())['schemes']['primary_7group']
    pilot=pd.read_csv(ROOT/'results/stage5/pilot/outcomes.csv',dtype={'region':str})
    excluded=set(zip(pilot.region,pilot.filename))
    headline=np.array([(r,f) not in excluded for r,f in zip(real.df.region,real.df.filename)])
    rows=[];predictions=[]
    for fold in scheme['outer_folds']:
        g=fold['fold_id'];tr=np.flatnonzero(real.group!=g);te=np.flatnonzero((real.group==g)&headline)
        if not len(te):raise ValueError('no headline real test rows in '+g)
        for name,data in [('M2_real_train',real),('M2_synthetic_train',syn)]:
            model=FittedPredictor('M2_dino',data,tr,g)
            probability=model.data(real,te)
            rows.append(dict(fold=g,model=name,variant='raw',source_rows=len(tr),**metrics(real.y[te],probability)))
            for idx,p in zip(te,probability):predictions.append(dict(fold=g,model=name,**real.df.iloc[idx][keys].to_dict(),label=int(real.y[idx]),probability=float(p)))
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics_per_fold.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    paired=contrast(frame,('M2_real_train','raw'),('M2_synthetic_train','raw'),'auroc')
    atomic_json(out/'summary.json',{'scope':'sample diagnostic; no confirmatory H1 claim',
        'real_inputs':real.input_hashes,'synthetic_inputs':syn.input_hashes,
        'hypotheses_sha256':sha256(ROOT/'configs/hypothesis_decisions_v1.yaml'),
        'contrast':paired,'excluding_G2':contrast(frame[frame.fold!='G2'],('M2_real_train','raw'),('M2_synthetic_train','raw'),'auroc')})
    print(json.dumps(paired,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
