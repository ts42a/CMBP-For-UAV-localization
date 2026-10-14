#!/usr/bin/env python3
"""CMBL command-prompt launcher: Local sample or gated Full run.

  python test.py
  python test.py --mode local --percent 8 --threads 4
  python test.py --mode local --percent 8 --plan-only
  python test.py --mode local --percent 8 --run-dir results/local/run_8pct

Rerun the identical command to resume. Logs and status.json are in the run folder.
Full mode never rents hardware or submits Kaggle jobs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
REGIONS=('01','02','03','04','05','06','08','09','10','11')

def acquire_lock(folder):
    """OS-held lock releases automatically if the runner crashes; prevents duplicate writers."""
    import atexit
    handle=(Path(folder)/'runner.lock').open('a+b')
    if handle.tell()==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close();raise RuntimeError('This run folder already has an active runner')
    atexit.register(handle.close)
    return handle

def select_frames(rows,percent,seed):
    """Proportional allocation with deterministic contiguous blocks for valid route replay."""
    if not math.isfinite(percent) or not 0<percent<=100:raise ValueError('Data percentage must be >0 and <=100')
    groups={r:sorted([x for x in rows if x['region_folder']==r],key=lambda x:int(x['num'])) for r in REGIONS}
    if any(not v for v in groups.values()):raise ValueError('All ten eligible regions are required')
    total=sum(map(len,groups.values()));target=round(total*percent/100)
    if target<2*len(groups):raise ValueError('Choose enough data for at least two frames per region')
    raw={r:len(v)*target/total for r,v in groups.items()}
    counts={r:max(2,math.floor(q)) for r,q in raw.items()}
    while sum(counts.values())>target:
        eligible=[r for r in groups if counts[r]>2]
        if not eligible:raise ValueError('Sample too small')
        r=max(eligible,key=lambda r:counts[r]-raw[r]);counts[r]-=1
    while sum(counts.values())<target:
        r=max((r for r in groups if counts[r]<len(groups[r])),key=lambda r:raw[r]-counts[r]);counts[r]+=1
    frames=[]
    for r,data in groups.items():
        available=len(data)-counts[r]+1
        start=int.from_bytes(hashlib.sha256(f'{seed}|{r}|local-block'.encode()).digest()[:8],'little')%available
        frames.extend({'region':r,'filename':x['filename'],'num':int(x['num'])} for x in data[start:start+counts[r]])
    return {'schema_version':'local_frame_selection_v1','seed':seed,'requested_percent':percent,
        'actual_percent':100*len(frames)/total,'eligible_frames':total,'selected_frames':len(frames),
        'selection':'proportional contiguous blocks per region; local diagnostic, not a random-frame geographic estimate',
        'per_region':counts,'frames':frames}


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode',choices=['local','full'])
    ap.add_argument('--percent',type=float)
    ap.add_argument('--seed',type=int,default=20261014)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--run-dir')
    ap.add_argument('--plan-only',action='store_true',help='Show sizes and stages without inference')
    ap.add_argument('--skip-replay',action='store_true',help='Explicitly omit live route diagnostic; recorded as skipped')
    ap.add_argument('--status',action='store_true',help='Print saved run status without starting work')
    ap.add_argument('--stop',action='store_true',help='Request a safe stop at the next heartbeat; identical command resumes')
    args=ap.parse_args()
    if (args.status or args.stop) and args.mode is None:args.mode='local'
    if args.mode is None:
        if not sys.stdin.isatty():ap.error('Use --mode local or --mode full for noninteractive execution')
        print('\nCMBL workflow\n  1. Local — configurable data sample, CPU diagnostics\n  2. Full  — full dataset; requires cloud 4080 acceptance\n')
        choice=input('Choose 1 or 2 [1]: ').strip() or '1'
        if choice not in ('1','2'):ap.error('Choose 1 or 2')
        args.mode='local' if choice=='1' else 'full'
        if args.mode=='local' and args.percent is None:
            try:args.percent=float(input('Data percentage [8]: ').strip() or '8')
            except ValueError:ap.error('Enter a numeric percentage')
        try:args.threads=int(input(f'CPU threads [{args.threads}]: ').strip() or str(args.threads))
        except ValueError:ap.error('Enter an integer thread count')
    percent=100. if args.mode=='full' else (8. if args.percent is None else args.percent)
    if args.mode=='full' and args.percent not in (None,100):ap.error('Full mode uses 100%; use Local for a sample')
    if not 1<=args.threads<=max(1,os.cpu_count() or 1):ap.error('Thread count exceeds this computer or is below 1')
    sys.path[:0]=[str(ROOT/'scripts/localization'),str(ROOT/'scripts/protocol')]
    from integrity import read_csv,sha256,atomic_json,source_identity,digest,semantic_manifest
    rows=read_csv(ROOT/'datasets/manifests/manifest.csv')
    try:selection=select_frames(rows,percent,args.seed)
    except ValueError as exc:ap.error(str(exc))
    default=f"results/{args.mode}/run_{percent:g}pct_seed{args.seed}"
    out=(ROOT/(args.run_dir or default)).resolve()
    if not out.is_relative_to(ROOT/'results'/args.mode):
        ap.error(f'Run directory must be under results/{args.mode}/ to protect historical outputs')
    if args.status or args.stop:
        if not (out/'status.json').is_file():ap.error('No saved run status at '+str(out))
        if args.stop:
            (out/'STOP').write_text(datetime.now(timezone.utc).isoformat());print('Stop requested; allow up to 30 seconds.')
        print((out/'status.json').read_text());return 0
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]=str(args.threads)
    env.update(CUDA_VISIBLE_DEVICES='' if args.mode=='local' else env.get('CUDA_VISIBLE_DEVICES',''),PYTHONUNBUFFERED='1')
    env['TEMP']=env['TMP']=str(ROOT/'build/test-tmp');Path(env['TEMP']).mkdir(parents=True,exist_ok=True)
    import platform
    from importlib.metadata import version
    identity={'mode':args.mode,'selection_sha256':digest(selection),'threads':args.threads,
        'skip_replay':args.skip_replay,'source_sha256':source_identity(ROOT),'launcher_sha256':sha256(Path(__file__)),
        'python':platform.python_version(),'packages':{n:version(n) for n in ('torch','numpy','scikit-learn','opencv-python')},
        'config_sha256':{p.name:sha256(p) for p in sorted((ROOT/'configs').glob('*')) if p.is_file()},
        'dataset_semantic_sha256':semantic_manifest(ROOT/'datasets/manifests/manifest.csv'),
        'embedding_files_sha256':{p.name:sha256(p) for p in sorted((ROOT/'datasets/cache/embeddings').glob('region*_dinov2_vits14_224.*'))}}
    n=selection['selected_frames']
    print(f"\nMode: {args.mode}; {n}/{selection['eligible_frames']} frames ({selection['actual_percent']:.3f}%)")
    print(f"Real labels: {n*9:,} attempts; synthetic labels: {n*9:,} attempts; CPU threads: {args.threads}")
    print('Regions:',selection['per_region']);print('Output:',out)
    print('Stages: preflight -> real/synthetic labels -> features -> v5 predictors -> H1 -> noise bound -> paired contrasts -> live replay -> report')
    print('Local replay: one L2 / 10% budget / seed 91001 diagnostic, P0-P6 plus matched equal spacing and failure-block stress.')
    if args.mode=='full':
        from launch_gate import check
        gate=check()
        if not gate['pass']:
            print('FULL RUN BLOCKED: cloud RTX 4080 comparison is pending. Use Local meanwhile.');return 2
        print('Full data processing is enabled. Replay uses live matching and recorded source-route thresholds; the hybrid fidelity check remains a separate gate.')
    if args.plan_only:return 0
    out.mkdir(parents=True,exist_ok=True)
    lock_handle=acquire_lock(out)
    (out/'STOP').unlink(missing_ok=True)
    print('Hashing selected UAV images for resume identity...',flush=True)
    by_frame={(r['region_folder'],r['filename']):r for r in rows}
    images={}
    for frame in selection['frames']:
        row=by_frame[(frame['region'],frame['filename'])]
        path=Path(row.get('image_path',''))
        if not path.is_file():path=ROOT/'datasets/raw/UAV_VisLoc_dataset'/frame['region']/'drone'/frame['filename']
        images[frame['region']+'/'+frame['filename']]=sha256(path)
    identity['query_images_sha256']=images
    plan=out/'run_plan.json'
    if plan.exists():
        if json.loads(plan.read_text())['identity']!=identity:raise ValueError('Run inputs/code/settings changed. Keep this run and choose a new --run-dir.')
    else:atomic_json(plan,{'identity':identity,'created_utc':datetime.now(timezone.utc).isoformat(),'scope':args.mode+' data workflow; replay is diagnostic, not confirmatory H4 evidence'})
    atomic_json(out/'frames.json',selection)
    (out/'logs').mkdir(exist_ok=True);(out/'reports').mkdir(exist_ok=True)
    status_path=out/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {'stages':{},'identity_sha256':digest(identity)}
    py=sys.executable
    def stage(name,command,outputs):
        if (out/'STOP').exists():
            status['state']='interrupted';atomic_json(status_path,status);raise KeyboardInterrupt
        outputs=[Path(p) for p in outputs];old=status['stages'].get(name,{})
        if old.get('state')=='complete' and old.get('outputs')=={str(p):sha256(p) for p in outputs if p.exists()} and all(p.is_file() for p in outputs):
            print(name+': already complete; output hashes verified',flush=True);return
        log=out/'logs'/f'{name}.log'
        status.update(state='running',current_stage=name,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid())
        status['stages'][name]={'state':'running','command':command,'log':str(log)};atomic_json(status_path,status)
        print(f'\n[{name}] running — {log}',flush=True)
        with log.open('a',encoding='utf-8') as stream:
            stream.write('\nCOMMAND '+json.dumps(command)+'\n');stream.flush()
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            status['child_pid']=process.pid;atomic_json(status_path,status)
            try:
                while True:
                    try:code=process.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        status['updated_utc']=datetime.now(timezone.utc).isoformat();atomic_json(status_path,status)
                        if (out/'STOP').exists():raise KeyboardInterrupt
                        progress=''
                        if name.startswith('labels_') and (out/name/'outcomes.csv').exists():
                            import csv
                            with (out/name/'outcomes.csv').open(newline='') as f:done=max(0,sum(1 for _ in csv.reader(f))-1)//9
                            status['progress']={'frames_complete':done,'frames_total':n};atomic_json(status_path,status)
                            progress=f' {done}/{n} frames;'
                        print(f'[{name}]{progress} still running; see {log.name}',flush=True)
            except KeyboardInterrupt:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                else:process.terminate()
                process.wait()
                status['state']='interrupted';status['stages'][name]['state']='interrupted';atomic_json(status_path,status)
                raise
        if code or not all(p.is_file() for p in outputs):
            status['state']='failed';status['stages'][name].update(state='failed',exit_code=code);atomic_json(status_path,status)
            raise RuntimeError(f'{name} failed; inspect {log}. Rerun the identical command after investigating.')
        status['stages'][name].update(state='complete',outputs={str(p):sha256(p) for p in outputs})
        atomic_json(status_path,status);print(f'[{name}] complete',flush=True)
    stage('preflight',[py,'scripts/protocol/verify.py','--only','env,frozen,splits,contracts'],[ROOT/'build/verify/last_verify.json'])
    for source in ('real','synthetic'):
        dest=out/f'labels_{source}'
        stage('labels_'+source,[py,'scripts/label_windowed.py','--set','full','--frames-json',str(out/'frames.json'),
            '--query-source',source,'--out',str(dest),'--resume'],[dest/'outcomes.csv',dest/'run_manifest.json'])
        from integrity import validate_outcomes,key
        labelled=read_csv(dest/'outcomes.csv');variant='prior' if source=='real' else 'syn_prior'
        expected={(r['region'],r['filename'],variant,str(rad),str(k)) for r in selection['frames'] for rad in (100,300,1000) for k in range(3)}
        if validate_outcomes(labelled)!=expected:raise ValueError('sample label grid differs from requested selection')
        features=out/f'features_{source}'
        stage('features_'+source,[py,'-m','cProfile','-o',str(out/f'features_{source}.prof'),'scripts/predictor/map_features.py',
            '--outcomes',str(dest/'outcomes.csv'),'--out',str(features),'--fig-name',str(out/'figures'/source)],
            [features/'features.csv',features/'embeddings.npz',features/'run_manifest.json'])
    pred=out/'predictors'
    stage('predictors',[py,'scripts/predictor/evaluate_predictors.py','--protocol','configs/predictor_protocol_v5.yaml',
        '--features',str(out/'features_real'),'--out',str(pred),'--name','local_v5','--reports-dir',str(out/'reports'),
        '--figures-dir',str(out/'figures')],[pred/'predictions.csv',pred/'metrics_per_fold.csv',out/'reports/local_v5_summary.json'])
    stage('h1',[py,'scripts/h1/evaluate_h1.py','--real',str(out/'features_real'),'--synthetic',str(out/'features_synthetic'),
        '--out',str(out/'h1')],[out/'h1/summary.json',out/'h1/metrics_per_fold.csv'])
    stage('noise_bound',[py,'scripts/predictor/label_noise_bound.py','--features',str(out/'features_real'),'--outcomes',str(out/'labels_real/outcomes.csv'),
        '--out',str(out/'noise_bound'),'--name','local','--reports-dir',str(out/'reports')],[out/'reports/local_label_noise_bound.json'])
    stage('paired',[py,'scripts/predictor/paired_contrasts.py',str(pred/'metrics_per_fold.csv'),'--out',str(out/'reports/paired.json')],
        [out/'reports/paired.json'])
    stage('hypotheses',[py,'scripts/protocol/hypothesis_report.py',str(out)],[out/'reports/hypotheses.json'])
    if not args.skip_replay:
        stage('replay',[py,'scripts/protocol/run_replay.py','--frames',str(out/'frames.json'),'--features',str(out/'features_real'),
            '--out',str(out/'replay'),'--stress',*(['--all-cells'] if args.mode=='full' else [])],[out/'replay/summary.json'])
        stage('fidelity',[py,'scripts/protocol/replay_fidelity.py','--replay',str(out/'replay'),'--frames',str(out/'frames.json'),
            '--labels',str(out/'labels_real/outcomes.csv'),'--out',str(out/'reports/fidelity.json')],[out/'reports/fidelity.json'])
    else:status['stages']['replay']={'state':'skipped','reason':'explicit --skip-replay'}
    status.update(state='complete',current_stage=None,updated_utc=datetime.now(timezone.utc).isoformat())
    if (out/'reports/fidelity.json').is_file():
        status['fidelity_acceptance']=json.loads((out/'reports/fidelity.json').read_text())['status']
    atomic_json(status_path,status)
    report=['# Local workflow result','',f'{n} frames; {n*9} real and {n*9} synthetic attempts.',
        'This is a development sample, not the registered full-scale result.','']
    report += [f"- {name}: {record['state']}" for name,record in status['stages'].items()]
    if 'fidelity_acceptance' in status:report += ['',f"Scientific fidelity acceptance: **{status['fidelity_acceptance']}**. A completed stage is not a claim that its acceptance criteria passed."]
    report+=['','Cloud 4080 comparison, production launch and the 2,000-attempt hybrid fidelity study remain separate gates.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\nLocal workflow complete:',out/'REPORT.md');return 0


if __name__=='__main__':
    venv=ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if venv.exists() and Path(sys.executable).resolve()!=venv.resolve():
        raise SystemExit(subprocess.call([str(venv),str(Path(__file__).resolve()),*sys.argv[1:]]))
    try:raise SystemExit(main())
    except KeyboardInterrupt:print('\nStopped. Rerun the identical command to resume.');raise SystemExit(130)
