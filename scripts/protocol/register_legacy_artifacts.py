"""One-time inventory of preserved pilot artifacts, with explicit legacy limitations.

This records existing bytes and validates tile order/geometry. It does not claim
that the legacy producer recorded these hashes at creation time. Never overwrite
the registry to make changed inputs pass; regenerate or review their provenance.
"""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/localization'))
from integrity import sha256, semantic_manifest, atomic_json, read_csv, validate_outcomes
from dino_embedder import load_embedding_cache, verify_model_artifacts, PIN_PATH
from region_map import load_region
from tile_index import TileSpec


def main():
    target=ROOT/'configs/legacy_artifacts_v1.json'
    if target.exists():
        raise SystemExit('Registry already exists; never overwrite an identity baseline automatically')
    verify_model_artifacts()
    record={'schema_version':'legacy_artifacts_v1','recorded_date':'2026-10-14',
        'limitation':'Audited snapshot of preserved legacy bytes, not retroactive producer provenance. Pilot association is grandfathered only for this exact pair; new features must carry keys and hashes.',
        'embedding_caches':{}, 'feature_pairs':[]}
    for p in sorted((ROOT/'datasets/cache/embeddings').glob('*.npz')):
        r=p.name[6:8]
        rasters,tiles=load_region(r,TileSpec(1024,512))
        load_embedding_cache(p,tiles=tiles)
        record['embedding_caches'][p.name]={'data_sha256':sha256(p),
            'metadata_sha256':sha256(p.with_suffix('.meta.json')),
            'model_pin_sha256':sha256(PIN_PATH),
            'rasters_sha256':{r.filename:sha256(r.path) for r in rasters}}
        print('registered region',r,flush=True)
    folder=ROOT/'results/stage6_7/pilot'
    record['feature_pairs'].append({'features_csv_sha256':sha256(folder/'features.csv'),
                                    'embeddings_npz_sha256':sha256(folder/'embeddings.npz')})
    manifest=ROOT/'datasets/manifests/manifest.csv'
    record['dataset_manifest']={'original_sha256':sha256(manifest),'semantic_sha256':semantic_manifest(manifest)}
    atomic_json(target,record)


if __name__=='__main__':
    main()
