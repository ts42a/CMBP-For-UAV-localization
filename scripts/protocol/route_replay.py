"""Deterministic route replay core. Coordinates are local metric EN, not degrees.

Predictor callback(frame_id, estimate_EN, radius) -> texture/raw/cal probabilities.
Localizer callback(frame_id, estimate_EN, radius) -> Fix. Neither receives truth.
The simulator alone owns route truth, corrupted odometry and scoring.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Fix:
    accepted: bool
    position: tuple[float, float] | None


def valid_fix(fix):
    if not isinstance(fix, Fix) or type(fix.accepted) is not bool:
        raise ValueError('invalid localizer response')
    if fix.accepted and (fix.position is None or len(fix.position)!=2 or not np.isfinite(fix.position).all()):
        raise ValueError('accepted fix requires finite metric position')
    return fix


def odometry(truth, params, seed, phi=.95):
    truth=np.asarray(truth,dtype=float)
    if truth.ndim!=2 or truth.shape[1]!=2 or len(truth)<2 or not np.isfinite(truth).all():
        raise ValueError('route requires at least two finite EN positions')
    rng=np.random.default_rng(seed)
    scale=rng.normal(0,params['sd_s']);bias=rng.normal(0,params['sd_b_rad'])
    noise=np.zeros(2);steps=[np.zeros(2)]
    for delta in np.diff(truth,axis=0):
        angle=bias+rng.normal(0,params['sd_h_rad'])
        c,s=math.cos(angle),math.sin(angle)
        noise=phi*noise+rng.normal(0,params['sd_w_m'],2)
        steps.append((1+scale)*np.array([[c,-s],[s,c]])@delta+noise)
    return np.asarray(steps)


def replay(frame_ids, truth, params, seed, policy, predictor, localizer,
           budget=.1, threshold=.5, initial_fix=None, exact_attempts=None, stress_frames=()):
    """Run one policy. Comparators may be rerun with exact_attempts=realized count.

    Initialization is an externally supplied known fix; default is the simulated
    starting GNSS fix. Only odometry and accepted measurements update the state.
    P6 oracle is intentionally separate from this deployable policy interface.
    """
    if policy not in ('P0_equal','P1_random','P2_texture','P3_uncalibrated','P4_calibrated','P5_untuned_product'):
        raise ValueError('unknown deployable policy')
    if not 0<=budget<=1 or not 0<=threshold<=1:
        raise ValueError('budget and threshold must be in [0,1]')
    truth=np.asarray(truth,dtype=float);steps=odometry(truth,params,seed)
    n=len(truth)
    if len(frame_ids)!=n or len(set(frame_ids))!=n:
        raise ValueError('frame identities must be unique and aligned')
    target=round(budget*n) if exact_attempts is None else exact_attempts
    if not isinstance(target,int) or not 0<=target<=n:
        raise ValueError('invalid attempt budget')
    if exact_attempts is not None and policy not in ('P0_equal','P1_random'):
        raise ValueError('exact budget reruns apply to comparator policies')
    rng=np.random.default_rng(np.random.SeedSequence([seed,1]))
    schedule=set(np.linspace(0,n-1,target).round().astype(int)) if policy=='P0_equal' else set(rng.choice(n,target,replace=False))
    estimate=np.array(truth[0] if initial_fix is None else initial_fix,dtype=float)
    distance=0.;product=1.;records=[];stress=set(stress_frames)
    for t,frame in enumerate(frame_ids):
        estimate+=steps[t];distance+=float(np.linalg.norm(steps[t]))
        sigma=30+params['rate']*distance
        radius=next((r for r in (100,300,1000) if r>=2*sigma),1000)
        probabilities=predictor(frame,tuple(estimate),radius)
        if set(probabilities)!= {'texture','raw','cal'} or any(not np.isfinite(p) or not 0<=p<=1 for p in probabilities.values()):
            raise ValueError('predictor must return finite texture/raw/cal probabilities')
        if policy in ('P0_equal','P1_random'):
            attempt=t in schedule
        elif policy=='P5_untuned_product':
            product*=1-probabilities['cal'];attempt=1-product>=.9
        else:
            name={'P2_texture':'texture','P3_uncalibrated':'raw','P4_calibrated':'cal'}[policy]
            attempt=probabilities[name]>=threshold
        used=False;gated=False;accepted=False;correct=False
        before=tuple(estimate)
        if attempt:
            fix=valid_fix(localizer(frame,before,radius))
            if t in stress:fix=Fix(False,None)
            accepted=fix.accepted
            if accepted:
                correct=bool(np.linalg.norm(np.asarray(fix.position)-truth[t])<=100)
                gated=bool(np.linalg.norm(np.asarray(fix.position)-estimate)>max(3*sigma,radius+300))
                if not gated:
                    estimate=np.asarray(fix.position,dtype=float).copy();distance=0.;product=1.;used=True
        error=float(np.linalg.norm(estimate-truth[t]))
        records.append({'frame':frame,'estimate_before':before,'estimate_after':tuple(estimate),
            'sigma':sigma,'radius':radius,'attempt':bool(attempt),'accepted':accepted,
            'correct':correct,'used':used,'gated':gated,'error_m':error,'probabilities':dict(probabilities)})
    errors=np.array([r['error_m'] for r in records]);episodes=0;streak=0
    for error in errors:
        streak=streak+1 if error>200 else 0
        if streak==10:episodes+=1
    metrics={'fraction_within_100m':float(np.mean(errors<=100)),'max_error_m':float(errors.max()),
        'terminal_error_m':float(errors[-1]),'loss_episodes':episodes,
        'attempts':sum(r['attempt'] for r in records),'gated_fixes':sum(r['gated'] for r in records),
        'accepted_correct':sum(r['accepted'] and r['correct'] for r in records),
        'accepted_incorrect':sum(r['accepted'] and not r['correct'] for r in records)}
    return {'policy':policy,'seed':seed,'metrics':metrics,'frames':records}


def fidelity(pairs):
    """Require exactly 2,000 unique planned live/lookup comparisons; zero accepted is inconclusive."""
    if len(pairs)!=2000 or len({p['attempt_id'] for p in pairs})!=2000:
        raise ValueError('fidelity check requires 2000 distinct attempts')
    allowed={'rejected','accepted_correct','accepted_incorrect'}
    same=0;both=0;within=0
    for p in pairs:
        if p['live_outcome'] not in allowed or p['lookup_outcome'] not in allowed:
            raise ValueError('unknown fidelity outcome')
        same+=p['live_outcome']==p['lookup_outcome']
        if p['live_outcome'].startswith('accepted_') and p['lookup_outcome'].startswith('accepted_'):
            both+=1
            distance=float(p['position_difference_m'])
            if not np.isfinite(distance) or distance<0:raise ValueError('invalid position comparison')
            within+=distance<=25
    return {'attempts':2000,'same_outcome':same,'both_accepted':both,'within_25m':within,
        'pass':same>=1800 and both>0 and within>=.9*both,
        'selected_source':'C_hybrid' if same>=1800 and both>0 and within>=.9*both else 'A_live'}
