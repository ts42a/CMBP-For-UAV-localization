"""Fail-closed, dependency-light contracts for research artifacts.

Historical files are never silently repaired or blessed. Validation errors identify
the source and require a new output or explicit, evidence-backed migration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

KEY = ("region", "filename", "variant", "radius_m", "offset_k")
OUTCOME_FIELDS = [
    "region", "group_7", "filename", "num", "variant", "radius_m", "offset_k",
    "offset_east_m", "offset_north_m", "offset_dist_m", "center_lat", "center_lon",
    "truth_dist_to_center_m", "truth_within_radius", "n_window_tiles", "n_candidates",
    "pred_lat", "pred_lon", "predicted_tile_id", "predicted_raster_id", "top_score",
    "score_margin", "fine_inliers", "used_homography", "geo_verify_status",
    "geodesic_error_m", "accept_rule", "accepted", "outcome_primary",
    "patch_n_mutual", "patch_query_rot_deg", "patch_sim_scale", "patch_sim_angle_deg", "patch_context_scale",
    "runtime_s", "frame_setup_s", "frame_cpu_s", "matcher_id", "config_path", "config_sha256", "prereg_sha256",
]

OUTCOMES = {"accepted_correct", "accepted_incorrect", "rejected"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_csv(path, fields=None):
    """Reject short/long records, duplicate headers and empty artifacts."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        header = reader.fieldnames
        if not header or len(header) != len(set(header)):
            raise ValueError(f"{path}: missing or duplicate CSV header")
        if fields is not None and header != list(fields):
            raise ValueError(f"{path}: unexpected CSV schema")
        rows = []
        for i, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}:{i}: incomplete or oversized record; preserve file and recover explicitly")
            rows.append(row)
    return rows


def key(row):
    try:
        result = tuple(str(row[k]) for k in KEY)
        if any(not v or v == "None" for v in result):
            raise ValueError("blank key")
        if int(result[3]) <= 0 or int(result[4]) < 0:
            raise ValueError("invalid radius/offset")
        if result[3] != str(int(result[3])) or result[4] != str(int(result[4])):
            raise ValueError("noncanonical numeric key")
        return result
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"invalid attempt key: {e}") from e


def number(row, name, lo=None, hi=None):
    try:
        v = float(row[name])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"missing/invalid {name}") from e
    if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ValueError(f"out-of-domain {name}: {v}")
    return v


def validate_outcomes(rows, expected=None, tolerance=100, allow_unlabelled=False):
    if not rows:
        raise ValueError("empty outcome table")
    seen = set()
    identities = set()
    for i, row in enumerate(rows, 2):
        k = key(row)
        if k in seen:
            raise ValueError(f"duplicate attempt key {k}")
        seen.add(k)
        for field, value in (expected or {}).items():
            if str(row.get(field, "")) != str(value):
                raise ValueError(f"row {i}: {field} differs from expected identity")
        for field in ("config_sha256", "prereg_sha256"):
            v = row.get(field, "")
            if len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
                raise ValueError(f"row {i}: invalid {field}")
        identities.add(tuple(row.get(n, "") for n in ("config_sha256", "prereg_sha256", "accept_rule")))
        number(row, "center_lat", -90, 90)
        number(row, "center_lon", -180, 180)
        for n in ("offset_east_m", "offset_north_m"):
            number(row, n)
        for n in ("runtime_s", "frame_setup_s", "frame_cpu_s", "fine_inliers", "n_window_tiles"):
            number(row, n, 0)
        outcome, accepted = row.get("outcome_primary"), row.get("accepted")
        if allow_unlabelled and outcome == "" and accepted == "" and row.get("accept_rule") == "":
            continue
        if outcome not in OUTCOMES or accepted not in ("0", "1") or not row.get("accept_rule"):
            raise ValueError(f"row {i}: invalid or unlabelled outcome")
        if (accepted == "0") != (outcome == "rejected"):
            raise ValueError(f"row {i}: acceptance/outcome conflict")
        has_lat, has_lon = row.get("pred_lat", "") != "", row.get("pred_lon", "") != ""
        if has_lat != has_lon or (accepted == "1" and not has_lat):
            raise ValueError(f"row {i}: incomplete predicted position")
        if has_lat:
            number(row, "pred_lat", -90, 90)
            number(row, "pred_lon", -180, 180)
            err = number(row, "geodesic_error_m", 0)
            # The producer rounds error to 3 decimals after classifying at full precision.
            if outcome == "accepted_correct" and err > tolerance + .0005:
                raise ValueError(f"row {i}: correct label exceeds tolerance")
            if outcome == "accepted_incorrect" and err < tolerance - .0005:
                raise ValueError(f"row {i}: incorrect label below tolerance")
        elif row.get("geodesic_error_m", "") != "":
            raise ValueError(f"row {i}: error without prediction")
        rule = row["accept_rule"]
        if rule.startswith(("patch_", "homography_")):
            status = 'patch_similarity' if rule.startswith('patch_') else 'homography'
            should_accept = (has_lat and row.get("geo_verify_status") == status
                             and row.get("used_homography") == "1"
                             and number(row, "fine_inliers", 0) >= int(rule.rsplit('_', 1)[1]))
            if (accepted == "1") != should_accept:
                raise ValueError(f"row {i}: acceptance rule inconsistent")
        elif rule == "any_output":
            if (accepted == "1") != (has_lat and row.get("geo_verify_status") != "empty_window"):
                raise ValueError(f"row {i}: any_output inconsistent")
        else:
            raise ValueError(f"row {i}: unknown acceptance rule {rule}")
    if len(identities) != 1:
        raise ValueError("mixed experiment identities in outcome table")
    return seen


def replace_with_retry(source, target, attempts=21, delay=0.25):
    """Keep atomic replacement; tolerate brief Windows sharing/antivirus locks."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == attempts - 1:
                raise
            time.sleep(delay)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        replace_with_retry(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def append_frame_atomic(path, rows, fields):
    """Commit a complete frame by replacing one CSV on the same filesystem."""
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as dst:
            if path.exists():
                with path.open('rb') as src:
                    shutil.copyfileobj(src, dst)
                    if src.tell():
                        src.seek(-1, 2)
                        if src.read(1) not in (b'\n', b'\r'):
                            dst.write(b'\n')
        with open(temp, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not path.exists() or path.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        replace_with_retry(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def source_identity(root):
    paths = sorted(p for base in ('scripts', 'kaggle') for p in (Path(root)/base).rglob('*.py'))
    return digest({p.relative_to(root).as_posix(): sha256(p) for p in paths})


def prepare_run(out_dir, manifest):
    """Preserve the first manifest. Legacy runs require a new output directory."""
    out_dir = Path(out_dir)
    target = out_dir/'run_manifest.json'
    fingerprint = digest(manifest['experiment'])
    manifest['experiment_sha256'] = fingerprint
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        if not old.get('experiment_sha256'):
            raise ValueError('Legacy run has no immutable fingerprint; use a new --out and retain the original')
        if digest(old['experiment']) != old['experiment_sha256'] or old['experiment_sha256'] != fingerprint:
            raise ValueError('Resume experiment identity changed; use a new --out')
        events = out_dir/'resume_events.jsonl'
        with events.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'at':manifest['started_at'], 'experiment_sha256':fingerprint})+'\n')
    elif (out_dir/'outcomes.csv').exists():
        raise ValueError('Outcome CSV has no provenance manifest; cannot resume')
    else:
        atomic_json(target, manifest)


def semantic_manifest(path):
    """Portable identity excludes only machine-local image path, preserves every other field."""
    rows = read_csv(path)
    return digest(sorted([{k:v for k,v in r.items() if k != 'image_path'} for r in rows],
                         key=lambda r: (r.get('region_folder',''), r.get('filename',''))))


def encoded_keys(rows):
    return [json.dumps(key(r), separators=(',', ':')) for r in rows]


def validate_feature_pair(folder, df, emb, groups, legacy_registry=None):
    import numpy as np
    folder = Path(folder)
    rows = df.to_dict('records')
    keys = encoded_keys(rows)
    if not rows or len(set(keys)) != len(keys):
        raise ValueError('empty or duplicate feature attempt keys')
    for r in rows:
        if r['label'] not in (0, 1) or groups.get(str(r['region'])) != r['group_7']:
            raise ValueError('invalid feature label or region/group membership')
    for name in ('near', 'window'):
        a = emb[name]
        if a.shape != (len(rows),384) or not np.isfinite(a).all():
            raise ValueError(f'invalid {name} feature dimensions/values')
    numeric = df.select_dtypes(include='number').to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError('nonfinite feature values')
    hashes = {'features_csv_sha256':sha256(folder/'features.csv'), 'embeddings_npz_sha256':sha256(folder/'embeddings.npz')}
    manifest = json.loads((folder/'run_manifest.json').read_text())
    if 'attempt_keys' in emb:
        if list(emb['attempt_keys']) != keys:
            raise ValueError('feature/embedding ordered attempt keys differ')
        if any(manifest.get(k) != v for k,v in hashes.items()):
            raise ValueError('feature input file hash differs from producer manifest')
    else:
        # Explicit grandfathering of the unchanged pilot pair, never count-based fallback.
        pairs = (legacy_registry or {}).get('feature_pairs', [])
        if hashes not in pairs:
            raise ValueError('unkeyed legacy feature pair not in audited byte-identity registry; regenerate')
    return hashes
