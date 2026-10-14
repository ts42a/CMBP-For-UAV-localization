"""Crash-safe memoization of expensive live localization at exact query centres."""
import json
import sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'localization'))
from integrity import atomic_json,digest
from route_replay import Fix,valid_fix

class CachedLocalizer:
    def __init__(self,folder,identity,live):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.identity=digest(identity);self.live=live;self.hits=0;self.misses=0
        manifest=self.folder/'identity.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!={'identity':identity,'sha256':self.identity}:
                raise ValueError('live-cache identity changed; use a new run directory')
        else:atomic_json(manifest,{'identity':identity,'sha256':self.identity})

    def __call__(self,frame,estimate,radius):
        query={'frame':str(frame),'estimate':[float(x) for x in estimate],'radius':int(radius)}
        token=digest({'identity':self.identity,'query':query});path=self.folder/(token+'.json')
        if path.exists():
            rec=json.loads(path.read_text())
            if rec.get('query')!=query or rec.get('sha256')!=digest(rec['fix']):raise ValueError('corrupt live-cache entry')
            self.hits+=1;f=rec['fix'];return valid_fix(Fix(f['accepted'],tuple(f['position']) if f['position'] is not None else None))
        fix=valid_fix(self.live(frame,estimate,radius));value=asdict(fix)
        atomic_json(path,{'query':query,'fix':value,'sha256':digest(value)})
        self.misses+=1;return fix

def paired_route_interval(left,right,reps=2000,seed=93001,block_size=50):
    """Paired moving-block resampling, routes first then common within-route blocks."""
    import numpy as np
    if set(left)!=set(right) or not left:raise ValueError('unmatched routes')
    arrays=[]
    for route in sorted(left):
        a,b=left[route],right[route]
        if len(a)!=len(b) or len(a)==0:raise ValueError('unaligned route frames')
        arrays.append((np.asarray(a)<=100).astype(float)-(np.asarray(b)<=100).astype(float))
    return block_difference_interval(arrays,reps,seed,block_size)

def block_difference_interval(arrays,reps=2000,seed=93001,block_size=50):
    import numpy as np
    rng=np.random.default_rng(seed);means=[]
    for _ in range(reps):
        values=[]
        for i in rng.integers(0,len(arrays),len(arrays)):
            d=arrays[i];size=min(block_size,len(d));starts=rng.integers(0,len(d)-size+1,int(np.ceil(len(d)/size)))
            values.append(np.concatenate([d[s:s+size] for s in starts])[:len(d)].mean())
        means.append(float(np.mean(values)))
    return {'paired_difference':float(np.mean([a.mean() for a in arrays])),
        'ci95':np.quantile(means,[.025,.975]).tolist(),'routes':len(arrays),'replicates':reps,'seed':seed,
        'block_frames':block_size,'scope':'sampled route diagnostic; not full H4 evidence'}

def paired_h4_interval(left,right,reps=2000,seed=93001):
    """Preserve paired seeds together; bootstrap aligned frame blocks then routes."""
    import numpy as np
    if set(left)!=set(right) or not left:raise ValueError('unmatched primary H4 routes')
    differences=[]
    for region in sorted(left):
        a,b=np.asarray(left[region]),np.asarray(right[region])
        if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=5:raise ValueError('H4 requires five aligned seeds per route')
        differences.append(((a<=100).astype(float)-(b<=100).astype(float)).mean(axis=0))
    result=block_difference_interval(differences,reps,seed)
    result.update(scope='Primary L2/B=0.10 P4 minus matched P0; five paired seeds retained together',
                  supported=result['ci95'][0]>0)
    return result
