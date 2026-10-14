"""Small real-map profile: bounded cache reuse, with exact output parity checked."""
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/predictor'),str(ROOT/'scripts/localization')]
from map_features import MapFeatureService
from region_map import load_region
from tile_index import TileSpec
from windowed_matcher import RegionIndex
from dino_embedder import load_embedding_cache

if __name__=='__main__':
    rasters,tiles=load_region('10',TileSpec(1024,512))
    a,_,_=load_embedding_cache(ROOT/'datasets/cache/embeddings/region10_dinov2_vits14_224.npz',tiles=tiles,rasters=rasters)
    service=MapFeatureService(RegionIndex(tiles,rasters,a));t=tiles[len(tiles)//2]
    start=time.perf_counter();reference=service.extract(t.center_lat,t.center_lon,1000);cold=time.perf_counter()-start
    start=time.perf_counter()
    for _ in range(30):
        value=service.extract(t.center_lat,t.center_lon,1000)
        assert value[0]==reference[0] and np.array_equal(value[1],reference[1]) and np.array_equal(value[2],reference[2])
    warm=(time.perf_counter()-start)/30
    result={'date':'2026-10-14','region':'10','queries':30,'cold_seconds':cold,'warm_mean_seconds':warm,
        'cached_tiles':len(service.texture_cache),'cache_limit':service.cache_limit,'exact_repeat_parity':True,
        'interpretation':'Single-centre microbenchmark; reuse helps repeated queries, not a whole-pipeline speedup estimate'}
    (ROOT/'reports/map_feature_profile_2026-10-14.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
