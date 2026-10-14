"""Paired group bootstrap supplementary report; never modifies historical summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def contrast(df,left,right,metric,seed=20261014,reps=10000):
    a=df[(df.model==left[0])&(df.variant==left[1])].set_index('fold')[metric]
    b=df[(df.model==right[0])&(df.variant==right[1])].set_index('fold')[metric]
    if a.index.has_duplicates or b.index.has_duplicates:raise ValueError('duplicate fold/model metrics')
    paired=pd.concat([a,b],axis=1,keys=['a','b']).dropna()
    delta=(paired.a-paired.b).to_numpy()
    if not len(delta):return {'eligible_groups':0,'inconclusive':True}
    if not np.isfinite(delta).all():raise ValueError('nonfinite fold metrics')
    rng=np.random.default_rng(seed)
    samples=delta[rng.integers(0,len(delta),(reps,len(delta)))].mean(axis=1)
    return {'left':list(left),'right':list(right),'metric':metric,'eligible_groups':len(delta),
        'groups':paired.index.tolist(),'inconclusive':len(delta)<5,'difference':float(delta.mean()),
        'ci95':np.quantile(samples,[.025,.975]).tolist(),'seed':seed,'replicates':reps}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('metrics');ap.add_argument('--out',required=True);args=ap.parse_args()
    df=pd.read_csv(args.metrics)
    specs=[(('M3_dino_distinct','raw'),('M1_texture','raw'),'auroc'),
           (('M2_dino','cal_v2_fe'),('M2_dino','raw'),'brier'),
           (('M2_dino','cal_v2_fe'),('M2_dino','raw'),'log_loss')]
    # Model names must be present; misspellings cannot silently produce empty results.
    available=set(zip(df.model,df.variant))
    for a,b,_ in specs:
        if a not in available or b not in available:raise ValueError(f'missing contrast model: {a}, {b}')
    result={'source':str(Path(args.metrics).resolve()),'scope':'supplementary paired group contrasts; pilot input cannot establish full-scale hypotheses',
        'contrasts':[contrast(df,a,b,m) for a,b,m in specs]}
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
