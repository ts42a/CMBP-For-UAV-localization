"""Repository root and standard paths (no hard-coded drive letters)."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("CMBL_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
DATASET_ROOT = ROOT / "datasets" / "raw" / "UAV_VisLoc_dataset"
MANIFEST = ROOT / "datasets" / "manifests" / "manifest.csv"
SAT_META = ROOT / "datasets" / "manifests" / "satellite_metadata.csv"
MATCHER_PROTOCOL = ROOT / "configs" / "matcher_protocol.yaml"
# Superseded Stage 3A config; renamed 2026-09-29 from matcher_dinov3_frozen.yaml (it holds DINOv2), content unchanged.
DINO_CONFIG = ROOT / "configs" / "matcher_dinov2_stage3a_frozen.yaml"
CACHE_EMB = ROOT / "datasets" / "cache" / "embeddings"
RESULTS_PILOT = ROOT / "results" / "pilot"
