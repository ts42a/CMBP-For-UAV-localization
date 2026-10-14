"""Strict Stage 11 gate; no model inference and no parameter fitting."""
import json
import csv
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/localization'))
from integrity import OUTCOME_FIELDS, read_csv, validate_outcomes, sha256, key, atomic_json

LEGACY_STAGE11 = '25b60606259d4937632407d0629b613ae6388996'
STAGE11_TAG = '88631e377a5ab8b006d9ad372ca516c1b77f8bdf'


def canonical_commit(value):
    return STAGE11_TAG if value == LEGACY_STAGE11 else value


def assess(rows, frames, pilot, prereg, expected_identity, env, expected_commit):
    """Pure acceptance logic, also exercised with adversarial synthetic fixtures."""
    radii = prereg['search_window']['radii_m']
    offsets = prereg['prior_offset']['offsets_per_frame_per_radius']
    region = expected_identity['region']
    expected = {(region, f, 'prior', str(r), str(k)) for f in frames for r in radii for k in range(offsets)}
    got = validate_outcomes(rows, expected_identity, prereg['tolerances_m']['primary'])
    if not frames or got != expected:
        raise ValueError(f'exact attempt grid mismatch: missing={len(expected-got)}, unexpected={len(got-expected)}')
    refs = {key(p):p for p in pilot if p['region'] == region}
    if not refs or not set(refs).issubset(expected):
        raise ValueError('no pilot overlap or expected pilot keys absent from full frame set')
    overlap = [(r, refs[key(r)]) for r in rows if key(r) in refs]
    if len(overlap) != len(refs):
        raise ValueError('not every expected pilot attempt is present')
    same = sum(r['outcome_primary'] == p['outcome_primary'] for r,p in overlap)
    accepted = [(r,p) for r,p in overlap if r['accepted'] == p['accepted'] == '1']
    def distance(r,p):
        return math.hypot((float(r['pred_lat'])-float(p['pred_lat']))*110574,
            (float(r['pred_lon'])-float(p['pred_lon']))*111320*math.cos(math.radians(float(p['pred_lat']))))
    within = sum(distance(r,p) <= 5 for r,p in accepted)
    if same < .97*len(overlap) or within < .95*len(accepted):
        raise ValueError('pilot outcome/position agreement failed')
    if env.get('cuda_available') is not True:
        raise ValueError('CUDA execution was not recorded')
    if env.get('tf32_matmul') is not False or env.get('tf32_cudnn') is not False:
        raise ValueError('TF32-off execution was not recorded')
    if canonical_commit(env.get('bundle_commit')) != canonical_commit(expected_commit):
        raise ValueError('unexpected bundle commit (history rewrite alias applied)')
    return {'pass':True, 'complete':True, 'rows':len(rows), 'expected_rows':len(expected),
        'frames':len(frames), 'expected_frames':len(frames), 'exact_grid':True,
        'pilot_overlap_attempts':len(overlap), 'pilot_overlap_identical_outcome':same,
        'pilot_overlap_both_accepted':len(accepted), 'pilot_overlap_within_5m':within,
        'bundle_commit':env['bundle_commit'], 'canonical_commit':canonical_commit(env['bundle_commit'])}


def check_region(region, expected_commit=STAGE11_TAG, directory=None, write=True):
    from datetime import datetime, timezone
    folder = Path(directory) if directory else ROOT/'results/full/labels_v1'/region
    rec = {'region':region, 'checked_utc':datetime.now(timezone.utc).isoformat(), 'gate_version':2}
    try:
        prereg = yaml.safe_load((ROOT/'configs/preregistration_v1.yaml').read_text())
        rows = read_csv(folder/'outcomes.csv', OUTCOME_FIELDS)
        frames = {r['filename'] for r in read_csv(ROOT/'datasets/manifests/manifest.csv') if r['region_folder']==region}
        identity = {'region':region, 'variant':'prior', 'config_sha256':sha256(ROOT/prereg['inputs']['frozen_matcher']),
                    'prereg_sha256':sha256(ROOT/'configs/preregistration_v1.yaml'), 'accept_rule':prereg['accept_rule']['selected']}
        manifest = json.loads((folder/'run_manifest.json').read_text())
        for field in ('config_sha256','prereg_sha256','accept_rule'):
            if manifest.get(field) != identity[field]:
                raise ValueError(f'run manifest {field} differs from frozen protocol')
        if manifest.get('splits_sha256') != sha256(ROOT/'configs/splits_v1.json'):
            raise ValueError('run manifest split identity differs')
        env = json.loads((folder/'env.json').read_text())
        rec.update(assess(rows, frames, read_csv(ROOT/'results/stage5/pilot/outcomes.csv'), prereg, identity, env, expected_commit))
        rec['input_sha256'] = {name:sha256(folder/name) for name in ('outcomes.csv','env.json','run_manifest.json')}
    except (ValueError, KeyError, OSError, TypeError, csv.Error) as e:
        rec.update({'pass':False, 'error':str(e)})
    if write:
        atomic_json(folder/'integrity_check.json',rec)
    return rec


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--regions',default='05,10')
    ap.add_argument('--expected-commit',default=STAGE11_TAG)
    args=ap.parse_args()
    results=[check_region(r,args.expected_commit) for r in args.regions.split(',')]
    print(json.dumps(results,indent=2))
    raise SystemExit(0 if all(x['pass'] for x in results) else 1)
