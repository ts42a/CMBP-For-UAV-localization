"""Report numbers from a freshly reproduced PC pilot batch, with explicit scope."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/localization'))
from integrity import sha256

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('verification_directory');args=ap.parse_args()
    folder=Path(args.verification_directory)
    data=folder/'v3/metrics_per_fold.csv'
    df=pd.read_csv(data)
    if data.read_bytes()!=(ROOT/'results/stage8/pilot_v3/metrics_per_fold.csv').read_bytes():
        raise ValueError('fresh baseline differs from reference; investigate before reporting')
    rows=[]
    for model in ('M0_constant','M1_texture','M2_dino','M3_dino_distinct'):
        d=df[(df.model==model)&(df.variant=='raw')]
        rows.append({'model':model,'macro_auroc':float(d.auroc.mean()),'macro_brier':float(d.brier.mean()),
            'eligible_auroc_groups':int(d.auroc.notna().sum())})
    rec={'date':'2026-10-14','scope':'PC pilot function check: 900 attempts, 100 frames; not confirmatory full-scale evidence',
        'source':str(data.resolve()),'source_sha256':sha256(data),'models':rows}
    (ROOT/'reports/baseline_pc_2026-10-14.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    text=['# PC baseline sanity check — 2026-10-14','',rec['scope']+'.',
        'Fresh protocol-v3 CPU run; fold metrics reproduced the historical reference byte for byte.',
        'Equal weight per eligible geographic group; G2 development group included. Undefined AUROC groups omitted.',
        '', '| Model | Macro AUROC | Macro Brier | AUROC groups |','|---|---:|---:|---:|']
    text += [f"| {r['model']} | {r['macro_auroc']:.4f} | {r['macro_brier']:.4f} | {r['eligible_auroc_groups']} |" for r in rows]
    text += ['',f"Source: `{data.as_posix()}`; SHA-256 `{sha256(data)}`.",
        'These pilot numbers do not pass or fail the newly registered H1–H3 full-scale tests.']
    (ROOT/'reports/baseline_pc_2026-10-14.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(rec,indent=2))
