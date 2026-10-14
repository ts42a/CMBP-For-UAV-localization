"""Seeded comparison of 2,000 distinct recorded live replay attempts with label lookup.

Consumes genuine live outcomes; never synthesizes a pass from fixture rows.
An insufficient attempt pool is INCOMPLETE, not a passed fidelity check.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from pyproj import CRS,Transformer
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/localization'))
from integrity import read_csv,validate_outcomes,sha256,digest,atomic_json
from route_replay import fidelity

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replay',required=True);ap.add_argument('--frames',required=True)
    ap.add_argument('--labels',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    selected=json.loads(Path(args.frames).read_text())['frames']
    manifest={(r['region_folder'],r['filename']):r for r in read_csv(ROOT/'datasets/manifests/manifest.csv')}
    transforms={}
    for region in sorted({f['region'] for f in selected}):
        first=min([f for f in selected if f['region']==region],key=lambda f:f['num'])
        r=manifest[(region,first['filename'])]
        transforms[region]=Transformer.from_crs('EPSG:4326',CRS.from_proj4(
            f"+proj=aeqd +lat_0={r['lat']} +lon_0={r['lon']} +datum=WGS84 +units=m"),always_xy=True)
    labels=read_csv(Path(args.labels));validate_outcomes(labels)
    lookup={}
    for r in labels:
        if r['variant']!='prior':raise ValueError('fidelity lookup requires real prior labels')
        lookup.setdefault((r['region'],r['filename'],int(r['radius_m'])),[]).append(r)
    pool={};sources={};identities={}
    for path in sorted(Path(args.replay).rglob('*.json')):
        # Include only ordinary policy records with region+level/budget metadata.
        # Ignore memoized fixes, source reference streams, stress injections and summaries.
        if 'live_cache' in path.parts or 'source_streams' in path.parts or 'stress' in path.stem:continue
        value=json.loads(path.read_text())
        if not all(k in value for k in ('region','level','budget','policy','frames')):continue
        if value['policy']=='P6_oracle':continue
        region=value['region'];transform=transforms[region]
        sources[str(path)]=sha256(path)
        for f in value['frames']:
            if not f['attempt']:continue
            cache_root=Path(args.replay)/'live_cache'/region
            if region not in identities:
                identities[region]=json.loads((cache_root/'identity.json').read_text())
                if digest(identities[region]['identity'])!=identities[region]['sha256']:raise ValueError('corrupt live-cache identity')
            query={'frame':str(f['frame']),'estimate':[float(x) for x in f['estimate_before']],'radius':int(f['radius'])}
            cache_key=digest({'identity':identities[region]['sha256'],'query':query})
            cached=json.loads((cache_root/(cache_key+'.json')).read_text())
            if (cached['query']!=query or digest(cached['fix'])!=cached['sha256'] or
                    cached['fix']['accepted']!=f['accepted'] or cached['fix']['position']!=f.get('fix_position')):
                raise ValueError('replay record differs from immutable live result')
            token=digest([region,f['frame'],f['radius'],f['estimate_before']])
            if token in pool:continue
            candidates=lookup.get((region,f['frame'],f['radius']))
            if not candidates:raise ValueError('missing lookup grid for recorded replay frame')
            centre=np.asarray(f['estimate_before'])
            row=min(candidates,key=lambda r:np.linalg.norm(np.asarray(transform.transform(float(r['center_lon']),float(r['center_lat'])))-centre))
            live_outcome='accepted_correct' if f['correct'] else ('accepted_incorrect' if f['accepted'] else 'rejected')
            rec={'attempt_id':token,'region':region,'level':value['level'],'budget':value['budget'],
                 'live_outcome':live_outcome,'lookup_outcome':row['outcome_primary']}
            if f['accepted'] and row['accepted']=='1':
                if f.get('fix_position') is None:raise ValueError('old replay has no accepted-fix position; regenerate')
                rec['position_difference_m']=float(np.linalg.norm(np.asarray(f['fix_position'])-
                    np.asarray(transform.transform(float(row['pred_lon']),float(row['pred_lat'])))))
            pool[token]=rec
    evidence={'labels_sha256':sha256(Path(args.labels)),'selection_sha256':sha256(Path(args.frames)),
              'source_sha256':sources,'unique_attempts_available':len(pool),'seed':92001}
    covered_levels={r['level'] for r in pool.values()};covered_budgets={r['budget'] for r in pool.values()}
    covered_regions={r['region'] for r in pool.values()}
    if (len(pool)<2000 or covered_levels!={'L0','L1','L2','L3'} or covered_budgets!={.05,.1,.2,.4}
            or covered_regions!={f['region'] for f in selected}):
        evidence.update({'status':'INCOMPLETE','pass':False,'reason':'requires 2000 distinct attempts and all declared levels/budgets','selected_source':'A_live'})
    else:
        rng=np.random.default_rng(92001);keys=sorted(pool)
        pairs=[pool[keys[i]] for i in rng.choice(len(keys),2000,replace=False)]
        evidence.update(fidelity(pairs));evidence['status']='PASS' if evidence['pass'] else 'FAIL'
        evidence['sample']=pairs
    atomic_json(Path(args.out),evidence);print(evidence['status'],len(pool),'distinct live attempts available')
    return 0  # execution succeeded; scientific acceptance is the explicit status above

if __name__=='__main__':raise SystemExit(main())
