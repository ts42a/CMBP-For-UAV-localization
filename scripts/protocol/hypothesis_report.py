"""Apply the committed numeric rules; local samples are always exploratory."""
import argparse
import json
import sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/localization'))
from integrity import atomic_json,sha256

def decide(contrasts,kind,full=False):
    if any(c.get('eligible_groups',0)<5 for c in contrasts):return {'verdict':'inconclusive','criteria_met':None}
    if kind in ('H1','H2'):
        c=contrasts[0];passed=c['difference']>=.02 and c['ci95'][0]>0
    elif kind=='H3':
        b,l=contrasts;passed=b['difference']<=-.005 and l['difference']<=-.01 and b['ci95'][1]<0 and l['ci95'][1]<0
    else:raise ValueError('unknown hypothesis')
    return {'verdict':('pass' if passed else 'fail') if full else 'exploratory_only','criteria_met':bool(passed)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run_dir');args=ap.parse_args();run=Path(args.run_dir)
    protocol=ROOT/'configs/hypothesis_decisions_v1.yaml';p=yaml.safe_load(protocol.read_text())
    expected='difference >= 0.02 and paired 95 percent CI lower bound > 0'
    if p['H1']['pass']!=expected or p['H2']['pass']!=expected or p['H3']['pass']!='Brier difference <= -0.005 and log-loss difference <= -0.01 and both paired 95 percent CI upper bounds < 0':
        raise ValueError('decision implementation requires the registered numeric rules')
    h1=json.loads((run/'h1/summary.json').read_text())['contrast']
    paired=json.loads((run/'reports/paired.json').read_text())['contrasts']
    h2=next(c for c in paired if c['metric']=='auroc')
    b=next(c for c in paired if c['metric']=='brier');l=next(c for c in paired if c['metric']=='log_loss')
    plan=json.loads((run/'run_plan.json').read_text());frames=json.loads((run/'frames.json').read_text())
    full=plan['identity']['mode']=='full' and frames['selected_frames']==frames['eligible_frames']
    report={'scope':'full-data registered decisions' if full else 'local development sample; not confirmatory',
        'protocol_sha256':sha256(protocol),'chronology':p['chronology'],
        'H1':decide([h1],'H1',full),'H2':decide([h2],'H2',full),'H3':decide([b,l],'H3',full),
        'inputs':{str(path.relative_to(run)):sha256(path) for path in (run/'h1/summary.json',run/'reports/paired.json')}}
    atomic_json(run/'reports/hypotheses.json',report)
