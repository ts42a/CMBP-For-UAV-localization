"""Local diagnostic route replay with real online features and live matcher outputs.

Thresholds use live P0 source-route streams defined in replay_execution_v1.json.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/predictor'),str(ROOT/'scripts/localization'),str(ROOT/'scripts')]
from integrity import read_csv,sha256,source_identity,atomic_json
from route_replay import Fix,replay
from replay_io import CachedLocalizer,paired_route_interval,paired_h4_interval
from evaluate_predictors import Data,TEX_COLS,EXTRA_COLS
from fitted_predictor import fit_fold
from map_features import MapFeatureService
from region_map import load_region
from tile_index import TileSpec
from dino_embedder import load_embedding_cache,DinoGlobalEmbedder
from windowed_matcher import RegionIndex,QueryContext,localize_windowed
from patch_verify import PatchVerifier
from label_windowed import is_accepted


class LiveMatcher:
    def __init__(self,index,frames,transform,embedder,frozen,rule):
        self.index=index;self.paths=frames;self.transform=transform;self.embedder=embedder
        self.frozen=frozen;self.rule=rule;self.verifier=PatchVerifier(embedder.model,frozen['fine_match'])
        self.current=None;self.context=None

    def __call__(self,frame,estimate,radius):
        import cv2
        if frame!=self.current:
            bgr=cv2.imread(str(self.paths[frame]))
            if bgr is None:raise ValueError('unreadable UAV image '+str(self.paths[frame]))
            self.context=QueryContext(bgr,self.embedder,self.frozen,self.verifier)
            self.current=frame;self.index.clear_cache()
        lon,lat=self.transform.transform(*estimate,direction='INVERSE')
        r=localize_windowed(self.context,self.index,lat,lon,radius,self.frozen,self.verifier)
        accepted=bool(is_accepted(self.rule,r))
        position=self.transform.transform(float(r['pred_lon']),float(r['pred_lat'])) if accepted else None
        return Fix(accepted,tuple(position) if position else None)


def source_streams(rows,params,seed,budget,out,identity,embedder,frozen,rule,cache_dir,extent,exclusion):
    """Reference route states are independent of fitted predictors; compute once per cell."""
    from pyproj import CRS,Transformer
    streams={}
    for region in sorted({r['region_folder'] for r in rows}):
        route=sorted([r for r in rows if r['region_folder']==region],key=lambda r:int(r['num']))
        transform=Transformer.from_crs('EPSG:4326',CRS.from_proj4(
            f"+proj=aeqd +lat_0={route[0]['lat']} +lon_0={route[0]['lon']} +datum=WGS84 +units=m"),always_xy=True)
        truth=np.array([transform.transform(float(r['lon']),float(r['lat'])) for r in route])
        rasters,tiles=load_region(region,TileSpec(1024,512))
        embs,_,_=load_embedding_cache(ROOT/f'datasets/cache/embeddings/region{region}_dinov2_vits14_224.npz',tiles=tiles,rasters=rasters)
        index=RegionIndex(tiles,rasters,embs);service=MapFeatureService(index,extent,exclusion)
        paths={r['filename']:ROOT/'datasets/raw/UAV_VisLoc_dataset'/region/'drone'/r['filename'] for r in route}
        live=CachedLocalizer(cache_dir/region,dict(identity,region=region),LiveMatcher(index,paths,transform,embedder,frozen,rule))
        values=[];near_values=[];window_values=[]
        def record(frame,estimate,radius):
            lon,lat=transform.transform(*estimate,direction='INVERSE')
            features,near,window=service.extract(lat,lon,radius)
            values.append(features);near_values.append(near);window_values.append(window)
            return dict(texture=.5,raw=.5,cal=.5)  # unused by equal-spacing scheduler
        result=replay([r['filename'] for r in route],truth,params,seed,'P0_equal',record,live,budget=budget)
        arrays={'texture':np.array([[r[k] for k in TEX_COLS] for r in values]),
            'extra':np.array([[r[k] for k in EXTRA_COLS] for r in values]),
            'near':np.asarray(near_values,dtype=float),'window':np.asarray(window_values,dtype=float)}
        streams[region]=arrays
        (out/'source_streams').mkdir(exist_ok=True)
        np.savez_compressed(out/'source_streams'/f'{region}.npz',**arrays)
        atomic_json(out/'source_streams'/f'{region}.json',result)
        print(f'source reference stream {region}: {len(route)} opportunities, {result["metrics"]["attempts"]} live attempts',flush=True)
    return streams


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frames',required=True);ap.add_argument('--features',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=91001)
    ap.add_argument('--level',choices=['L0','L1','L2','L3'],default='L2');ap.add_argument('--budget',type=float,default=.1)
    ap.add_argument('--stress',action='store_true',help='Also run paired 5/10/20-frame failure blocks')
    ap.add_argument('--all-cells',action='store_true',help='Run all registered levels, budgets and seeds; thresholds remain explicitly diagnostic')
    ap.add_argument('--cache-dir',help='Share exact-centre live results across cells')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.all_cells:
        import subprocess
        records=[]
        for level in ('L0','L1','L2','L3'):
            for budget in (.05,.1,.2,.4):
                for seed in (91001,91002,91003,91004,91005):
                    dest=out/f'{level}_b{budget}_s{seed}'
                    command=[sys.executable,__file__,'--frames',args.frames,'--features',args.features,'--out',str(dest),
                        '--level',level,'--budget',str(budget),'--seed',str(seed),'--cache-dir',str(out/'live_cache')]
                    if args.stress:command.append('--stress')
                    subprocess.check_call(command)
                    records.append({'level':level,'budget':budget,'seed':seed,'summary':str(dest/'summary.json')})
        left={};right={}
        for seed in (91001,91002,91003,91004,91005):
            cell=out/f'L2_b0.1_s{seed}'
            for p in sorted(cell.glob('[0-9][0-9]_P4_calibrated.json')):
                r=json.loads(p.read_text());base=json.loads((cell/f"{r['region']}_P4_calibrated_matched_equal.json").read_text())
                if [x['frame'] for x in r['frames']]!=[x['frame'] for x in base['frames']]:raise ValueError('H4 frame mismatch')
                left.setdefault(r['region'],[]).append([x['error_m'] for x in r['frames']])
                right.setdefault(r['region'],[]).append([x['error_m'] for x in base['frames']])
        secondary={}
        for other in ('P2_texture','P3_uncalibrated'):
            l={};r={}
            for seed in (91001,91002,91003,91004,91005):
                for path in sorted((out/f'L2_b0.1_s{seed}').glob(f'[0-9][0-9]_P4_vs_{other}.json')):
                    record=json.loads(path.read_text());region=record['region']
                    l.setdefault(region,[]).append([x['error_m'] for x in record['left']['frames']])
                    r.setdefault(region,[]).append([x['error_m'] for x in record['right']['frames']])
            secondary['P4_vs_'+other]=paired_h4_interval(l,r)
            secondary['P4_vs_'+other]['scope']='Matched actual-attempt P4 vs '+other+'; five paired seeds'
        atomic_json(out/'summary.json',{'scope':'all 80 cells on supplied routes; live source, source-route thresholds',
            'cells':records,'h4_primary':paired_h4_interval(left,right),'h4_secondary':secondary})
        return 0
    from pyproj import CRS,Transformer
    selection=json.loads(Path(args.frames).read_text());selected={(r['region'],r['filename']) for r in selection['frames']}
    rows=[r for r in read_csv(ROOT/'datasets/manifests/manifest.csv') if (r['region_folder'],r['filename']) in selected]
    data=Data(Path(args.features));scheme=json.loads((ROOT/'configs/splits_v1.json').read_text())['schemes']['primary_7group']
    groups={r:g for g,v in scheme['groups'].items() for r in v['regions']}
    folds={f['fold_id']:f for f in scheme['outer_folds']};fits={}
    prereg=yaml.safe_load((ROOT/'configs/preregistration_v1.yaml').read_text())
    frozen=yaml.safe_load((ROOT/prereg['inputs']['frozen_matcher']).read_text())
    fp=yaml.safe_load((ROOT/'configs/predictor_protocol_v1.yaml').read_text())['distinctiveness']
    calibration=json.loads((ROOT/'configs/drift_calibration.json').read_text());params=calibration['levels'][args.level]
    identity={'source':source_identity(ROOT),'features':data.input_hashes,'selection':sha256(Path(args.frames)),
        'matcher':sha256(ROOT/prereg['inputs']['frozen_matcher']),'model':sha256(ROOT/'configs/dinov2_artifacts_v1.json')}
    identity['replay_execution_sha256']=sha256(ROOT/'configs/replay_execution_v1.json')
    cache_dir=Path(args.cache_dir) if args.cache_dir else out/'live_cache'
    embedder=DinoGlobalEmbedder(image_size=224);all_results=[];paired_left={};paired_right={}
    streams=source_streams(rows,params,args.seed,args.budget,out,identity,embedder,frozen,prereg['accept_rule']['selected'],
                           cache_dir,float(fp['search_extent_m'].split('+')[1]),float(fp['exclusion_m']))
    for region in sorted({r['region_folder'] for r in rows}):
        route=sorted([r for r in rows if r['region_folder']==region],key=lambda r:int(r['num']))
        if len(route)<2:raise ValueError('replay needs at least two frames per region')
        group=groups[region]
        if group not in fits:fits[group]=fit_fold(data,folds[group])
        models,calibrate,fit_manifest,_=fits[group]
        source_prob={name:[] for name in ('texture','raw','cal')}
        for source_region,stream in streams.items():
            if groups[source_region]==group:continue
            raw=models['M2_dino'].arrays(stream['texture'],stream['near'],stream['window'],stream['extra'])
            source_prob['raw'].extend(raw);source_prob['cal'].extend(calibrate(raw))
            source_prob['texture'].extend(models['M1_texture'].arrays(stream['texture'],stream['near'],stream['window'],stream['extra']))
        if not source_prob['raw']:raise ValueError('no source-route threshold opportunities')
        transform=Transformer.from_crs('EPSG:4326',CRS.from_proj4(
            f"+proj=aeqd +lat_0={route[0]['lat']} +lon_0={route[0]['lon']} +datum=WGS84 +units=m"),always_xy=True)
        truth=np.array([transform.transform(float(r['lon']),float(r['lat'])) for r in route]);ids=[r['filename'] for r in route]
        truth_by_id=dict(zip(ids,truth))
        rasters,tiles=load_region(region,TileSpec(1024,512))
        embs,_,_=load_embedding_cache(ROOT/f'datasets/cache/embeddings/region{region}_dinov2_vits14_224.npz',tiles=tiles,rasters=rasters)
        index=RegionIndex(tiles,rasters,embs)
        service=MapFeatureService(index,float(fp['search_extent_m'].split('+')[1]),float(fp['exclusion_m']))
        def predict(frame,estimate,radius):
            lon,lat=transform.transform(*estimate,direction='INVERSE')
            features,near,window=service.extract(lat,lon,radius)
            raw=models['M2_dino'].query(features,near,window)
            return {'texture':models['M1_texture'].query(features,near,window),'raw':raw,'cal':float(calibrate(np.array([raw]))[0])}
        if embedder is None:embedder=DinoGlobalEmbedder(image_size=224)
        paths={r['filename']:ROOT/'datasets/raw/UAV_VisLoc_dataset'/region/'drone'/r['filename'] for r in route}
        live=LiveMatcher(index,paths,transform,embedder,frozen,prereg['accept_rule']['selected'])
        cached=CachedLocalizer(cache_dir/region,dict(identity,region=region),live)
        def oracle(frame,estimate,radius):
            fix=cached(frame,estimate,radius)
            return fix.accepted and np.linalg.norm(np.asarray(fix.position)-truth_by_id[frame])<=100
        region_results=[]
        policies=['P0_equal','P1_random','P2_texture','P3_uncalibrated','P4_calibrated','P5_untuned_product','P6_oracle']
        for policy in policies:
            field={'P2_texture':'texture','P3_uncalibrated':'raw','P4_calibrated':'cal'}.get(policy)
            threshold=float(np.quantile(source_prob[field],1-args.budget)) if field else .5
            result=replay(ids,truth,params,args.seed,policy,predict,cached,budget=args.budget,threshold=threshold,oracle=oracle)
            result.update(region=region,level=args.level,budget=args.budget,threshold=threshold,fit=fit_manifest,
                threshold_scope='source-route P0 reference stream at same level/seed/budget, excludes held-out group',
                threshold_source_fraction=float(np.mean(np.asarray(source_prob[field])>=threshold)) if field else None)
            attempts=[f for f in result['frames'] if f['attempt']]
            streak=0;completed=[]
            for attempt in attempts:
                if attempt['correct']:completed.append(streak);streak=0
                else:streak+=1
            pmean=float(np.mean([f['probabilities']['cal'] for f in attempts])) if attempts else None
            result['failure_runs']={'completed_lengths':completed,'right_censored_trailing_length':streak,
                'mean_calibrated_probability':pmean,
                'geometric_expected_probability':{str(k):pmean*(1-pmean)**k for k in range(max(completed,default=0)+2)} if pmean is not None else {},
                'note':'Independent Bernoulli reference only; no route-level guarantee'}
            atomic_json(out/f'{region}_{policy}.json',result);region_results.append(result)
            print(f"replay {region} {policy}: {result['metrics']}",flush=True)
        for result in region_results:
            if result['policy'] in ('P0_equal','P1_random'):continue
            count=result['metrics']['attempts']
            comparator=replay(ids,truth,params,args.seed,'P0_equal',predict,cached,exact_attempts=count)
            atomic_json(out/f"{region}_{result['policy']}_matched_equal.json",comparator)
            result['matched_equal_metrics']=comparator['metrics']
            if result['policy']=='P4_calibrated':
                paired_left[region]=[f['error_m'] for f in result['frames']]
                paired_right[region]=[f['error_m'] for f in comparator['frames']]
        by_policy={r['policy']:r for r in region_results}
        for other,field in (('P2_texture','texture'),('P3_uncalibrated','raw')):
            cap=min(by_policy['P4_calibrated']['metrics']['attempts'],by_policy[other]['metrics']['attempts'])
            a=replay(ids,truth,params,args.seed,'P4_calibrated',predict,cached,budget=args.budget,
                threshold=float(np.quantile(source_prob['cal'],1-args.budget)),attempt_cap=cap)
            b=replay(ids,truth,params,args.seed,other,predict,cached,budget=args.budget,
                threshold=float(np.quantile(source_prob[field],1-args.budget)),attempt_cap=cap)
            if a['metrics']['attempts']!=cap or b['metrics']['attempts']!=cap:raise ValueError('matched predictor budget failed')
            atomic_json(out/f'{region}_P4_vs_{other}.json',{'region':region,'attempts_per_policy':cap,'left':a,'right':b})
        if args.stress:
            for length in (5,10,20):
                rng=np.random.default_rng(np.random.SeedSequence([calibration['stress_seed'],int(region),length]))
                start=int(rng.integers(0,max(1,len(ids)-length+1)));blocked=range(start,min(len(ids),start+length))
                for policy in policies[:-1]:
                    field={'P2_texture':'texture','P3_uncalibrated':'raw','P4_calibrated':'cal'}.get(policy)
                    threshold=float(np.quantile(source_prob[field],1-args.budget)) if field else .5
                    result=replay(ids,truth,params,args.seed,policy,predict,cached,budget=args.budget,threshold=threshold,stress_frames=blocked)
                    comparator=replay(ids,truth,params,args.seed,'P0_equal',predict,cached,
                                      exact_attempts=result['metrics']['attempts'],stress_frames=blocked)
                    result['matched_equal_metrics']=comparator['metrics']
                    atomic_json(out/f'{region}_{policy}_stress{length}.json',result)
        all_results.extend([{k:v for k,v in r.items() if k!='frames'} for r in region_results])
        atomic_json(out/'progress.json',{'finished_regions':sorted({r['region'] for r in all_results}),'results':all_results})
    report={'scope':'Local sample, one seed/level/budget; live source A; exploratory, no H4 decision',
        'identity':identity,'results':all_results,'paired_P4_minus_matched_equal':paired_route_interval(paired_left,paired_right)}
    atomic_json(out/'summary.json',report)
    return 0

if __name__=='__main__':raise SystemExit(main())
