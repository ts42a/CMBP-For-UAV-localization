#!/usr/bin/env python3
"""One command that re-checks the frozen pipeline. Run it before every freeze, tag or full run.

  py -3 scripts/protocol/verify.py             # everything (~5 min on the PC CPU)
  py -3 scripts/protocol/verify.py --quick     # skip the two slow checks (labels, predictor_v4)
  py -3 scripts/protocol/verify.py --only splits,frozen

Checks (each re-uses an existing script; nothing is written to reports/ or results/):
  env           installed versions equal the pins in requirements-localization.txt; one OpenCV package
  frozen        frozen files still hash to what the committed outputs were produced with
  splits        scripts/protocol/make_splits_v1.py --check
  predictor     protocol v3 on the pilot features reproduces results/stage8/pilot_v3 byte-for-byte
                (the run itself asserts no test-group row in any fit and fe_platt monotonicity;
                the leakage control must PASS)
  predictor_v4  protocol v4 with half the pilot frames excluded: predictions equal v3, the
                all-frames sensitivity equals the v3 headline, headline metrics match a
                recomputation from the flagged rows (deviation D5)
  predictor_v5  protocol v5 (proposed) on the pilot with the headline exclusion off: the registered
                models' predictions equal v3 exactly and the two _diag models are present
  labels        golden reproduction: the frozen labeller on 2 pilot frames (18 attempts) agrees
                with results/stage5/pilot/outcomes.csv (scripts/protocol/stage10_compare.py)

Scratch output goes to build/verify/<timestamp>/ (git-ignored); the result table is also written
to build/verify/last_verify.json. Exit code 0 only if every selected check passes, 1 if any fails,
2 if none fails but some were skipped.

A check SKIPs when it needs local files that are not in the public repository (UAV-VisLoc and the
per-frame metadata derived from it, which are not cleared for redistribution; see the README).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
CHECKS = ("env", "frozen", "splits", "contracts", "full_labels", "predictor", "predictor_v4", "predictor_v5", "labels")
SLOW = ("predictor_v4", "predictor_v5", "labels")
PILOT_OUT = ROOT / "results" / "stage5" / "pilot" / "outcomes.csv"
V3_DIR = ROOT / "results" / "stage8" / "pilot_v3"
BENCH_REGIONS = "01,02,03,04,05,06,08,09,10,11"


class Fail(Exception):
    pass


class Skip(Exception):
    pass


LOCAL = {  # local-only inputs (git-ignored), per check
    "frozen": ["results/stage5/pilot/outcomes.csv"],
    "splits": ["datasets/manifests/manifest.csv"],
    "predictor": ["results/stage6_7/pilot/features.csv", "results/stage6_7/pilot/embeddings.npz"],
    "predictor_v4": ["results/stage6_7/pilot/features.csv", "results/stage6_7/pilot/embeddings.npz",
                     "results/stage5/pilot/outcomes.csv"],
    "predictor_v5": ["results/stage6_7/pilot/features.csv", "results/stage6_7/pilot/embeddings.npz",
                     "datasets/manifests/frame_agl.csv"],
    "labels": ["datasets/raw/UAV_VisLoc_dataset", "datasets/manifests/manifest.csv",
               "datasets/manifests/satellite_metadata.csv", "datasets/cache/embeddings",
               "results/stage5/pilot/outcomes.csv"],
}


def needs(check: str) -> None:
    missing = [p for p in LOCAL.get(check, []) if not (ROOT / p).exists()]
    if missing:
        raise Skip("needs local files not in the public repository: " + ", ".join(missing))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if rc:
        tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
        raise Fail(f"exit {rc}: {' | '.join(tail)}  (log {log})")


def check_env(tmp: Path) -> str:
    pins = {}
    for line in (ROOT / "requirements-localization.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line.strip())
        if m:
            pins[m.group(1)] = m.group(2)
    bad = []
    for name, want in pins.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            bad.append(f"{name} missing (want {want})")
            continue
        if have.split("+")[0] != want:
            bad.append(f"{name} {have} != {want}")
    cv = [n for n in ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless")
          if _installed(n)]
    if len(cv) != 1:
        bad.append(f"{len(cv)} OpenCV packages installed ({', '.join(cv)}); they overwrite each other's cv2")
    if bad:
        raise Fail("; ".join(bad) + f"  [python {sys.executable}]")
    return f"{len(pins)} pins match, one OpenCV ({cv[0]}), python {sys.version.split()[0]}"


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def check_frozen(tmp: Path) -> str:
    import csv
    import yaml
    prereg = yaml.safe_load((ROOT / "configs/preregistration_v1.yaml").read_text(encoding="utf-8"))
    matcher = ROOT / prereg["inputs"]["frozen_matcher"]
    with PILOT_OUT.open(encoding="utf-8", newline="") as f:
        cfg = {r["config_sha256"] for r in csv.DictReader(f)}
    if cfg != {sha256(matcher)}:
        raise Fail(f"{matcher.name} sha256 {sha256(matcher)[:12]} differs from the pilot rows {sorted(c[:12] for c in cfg)}")
    v3_manifest = json.loads((V3_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    v3 = ROOT / v3_manifest["protocol_path"]
    if sha256(v3) != v3_manifest["protocol_sha256"]:
        raise Fail(f"{v3.name} changed since the v3 pilot run")
    v4 = yaml.safe_load((ROOT / "configs/predictor_protocol_v4.yaml").read_text(encoding="utf-8"))
    feat = ROOT / v4["inputs"]["features_protocol"]
    if sha256(feat) != v4["inputs"]["features_protocol_sha256"]:
        raise Fail(f"{feat.name} no longer matches features_protocol_sha256 in predictor_protocol_v4.yaml")
    return f"matcher config, predictor v3 and features protocol hashes unchanged"


def check_splits(tmp: Path) -> str:
    log = tmp / "splits.log"
    run([PY, "scripts/protocol/make_splits_v1.py", "--check"], log)
    return log.read_text(encoding="utf-8").strip().splitlines()[-1]


def evaluate(protocol: str, tmp: Path, name: str) -> tuple[Path, dict]:
    out = tmp / name
    run([PY, "scripts/predictor/evaluate_predictors.py", "--protocol", protocol, "--features", "results/stage6_7/pilot",
         "--out", str(out), "--name", name, "--reports-dir", str(tmp), "--figures-dir", str(tmp / "figures")],
        tmp / f"{name}.log")
    return out, json.loads((tmp / f"{name}_summary.json").read_text(encoding="utf-8"))


def check_predictor(tmp: Path) -> str:
    out, summary = evaluate("configs/predictor_protocol_v3.yaml", tmp, "v3")
    for f in ("predictions.csv", "metrics_per_fold.csv"):
        if (out / f).read_bytes() != (V3_DIR / f).read_bytes():
            raise Fail(f"{f} differs from {V3_DIR.relative_to(ROOT)}")
    ref = json.loads((ROOT / "reports/stage6_8_pilot_v3_summary.json").read_text(encoding="utf-8"))
    if summary != ref:
        raise Fail("summary JSON differs from reports/stage6_8_pilot_v3_summary.json")
    lk = summary["leakage_control_auroc"]
    if not lk["pass"]:
        raise Fail(f"leakage control FAIL (macro {lk['macro_mean']:.3f})")
    return f"v3 reproduced byte-for-byte; leakage control {lk['macro_mean']:.3f} PASS"


def check_predictor_v4(tmp: Path) -> str:
    import numpy as np
    import pandas as pd
    import yaml
    from sklearn.metrics import brier_score_loss, roc_auc_score

    frames = pd.read_csv(PILOT_OUT, dtype=str)[["region", "filename"]].drop_duplicates()
    half = frames[frames.groupby("region").cumcount() % 2 == 0]
    half.to_csv(tmp / "half_frames.csv", index=False)
    proto = yaml.safe_load((ROOT / "configs/predictor_protocol_v4.yaml").read_text(encoding="utf-8"))
    proto["evaluation"]["headline_exclude_frames"] = str(tmp / "half_frames.csv")
    (tmp / "protocol_v4_half.yaml").write_text(yaml.safe_dump(proto, sort_keys=False), encoding="utf-8")
    out, s4 = evaluate(str(tmp / "protocol_v4_half.yaml"), tmp, "v4_half")

    p4 = pd.read_csv(out / "predictions.csv", dtype={"region": str})
    p3 = pd.read_csv(V3_DIR / "predictions.csv", dtype={"region": str})
    cols = [c for c in p3.columns if c != "protocol_sha256"]
    if not p4[cols].equals(p3[cols]):
        raise Fail("v4 changed training: predictions differ from v3")
    s3 = json.loads((ROOT / "reports/stage6_8_pilot_v3_summary.json").read_text(encoding="utf-8"))
    for key, v in s3["models"].items():
        for k in ("auroc", "brier", "log_loss", "ece"):
            a, b = s4["sensitivity_all_frames"][key][k], v["macro"][k]
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
                raise Fail(f"all-frames sensitivity {key} {k} {a} != v3 headline {b}")
    excl = set(map(tuple, half.values))
    listed = [(r, f) in excl for r, f in zip(p4.region, p4.filename)]
    if list(~p4.in_headline) != listed:
        raise Fail("in_headline does not match the exclusion list")
    mf = pd.read_csv(out / "metrics_per_fold.csv")
    for (fold, m), g in p4[p4.in_headline].groupby(["fold", "model"]):
        r = mf[(mf.fold == fold) & (mf.model == m) & (mf.variant == "raw")].iloc[0]
        y, p = g.label.to_numpy(), g.p_raw.to_numpy()
        if r.n != len(g) or abs(r.brier - brier_score_loss(y, p)) > 1e-12:
            raise Fail(f"headline metrics for {fold}/{m} do not match the flagged rows")
        if len(np.unique(y)) == 2 and abs(r.auroc - roc_auc_score(y, p)) > 1e-12:
            raise Fail(f"headline AUROC for {fold}/{m} does not match the flagged rows")
    return f"headline {s4['headline_rows']}/{s4['n_rows']} rows; training unchanged; sensitivity == v3; metrics recomputed"


def check_predictor_v5(tmp: Path) -> str:
    import pandas as pd
    import yaml

    proto = yaml.safe_load((ROOT / "configs/predictor_protocol_v5.yaml").read_text(encoding="utf-8"))
    del proto["evaluation"]["headline_exclude_frames"]  # every pilot frame would be excluded
    (tmp / "protocol_v5_pilot.yaml").write_text(yaml.safe_dump(proto, sort_keys=False), encoding="utf-8")
    out, s5 = evaluate(str(tmp / "protocol_v5_pilot.yaml"), tmp, "v5_pilot")
    p5 = pd.read_csv(out / "predictions.csv", dtype={"region": str})
    p3 = pd.read_csv(V3_DIR / "predictions.csv", dtype={"region": str})
    cols = [c for c in p3.columns if c != "protocol_sha256"]
    reg = p5[p5.model.isin(p3.model.unique())].reset_index(drop=True)
    if not reg[cols].equals(p3[cols]):
        raise Fail("v5 changed the registered models' predictions")
    diag = sorted(set(p5.model) - set(p3.model))
    if diag != sorted(proto["diagnostic_models"]["models"]):
        raise Fail(f"diagnostic models {diag} != protocol {proto['diagnostic_models']['models']}")
    return f"registered predictions == v3; diagnostics present: {', '.join(diag)}"


def check_labels(tmp: Path) -> str:
    out = tmp / "labels"
    run([PY, "scripts/label_windowed.py", "--set", "pilot", "--per-region", "2", "--regions", BENCH_REGIONS,
         "--max-frames", "2", "--out", str(out), "--resume"], tmp / "labels.log")
    run([PY, "scripts/protocol/stage10_compare.py", str(out), "--expect-frames", "2", "--name", "labels_compare",
         "--reports-dir", str(tmp)], tmp / "labels_compare.log")
    s = json.loads((tmp / "labels_compare.json").read_text(encoding="utf-8"))
    a = s["checks"]
    if not a["agreement_pass"] or a["outcome_disagreements"] != 0:
        raise Fail(f"golden labels disagree: {a['outcome_disagreements']} outcome(s), agreement_pass={a['agreement_pass']}")
    return (f"{a['n_attempts']} attempts identical outcomes, max position diff {a['position_max_diff_m']} m, "
            f"config hash {'matches' if a['config_sha256_matches_reference'] else 'DIFFERS'}")


def check_contracts(tmp: Path) -> str:
    import os
    scratch = tmp / 'test-tmp'
    scratch.mkdir()
    old = {k: os.environ.get(k) for k in ('TEMP', 'TMP')}
    try:
        os.environ.update(TEMP=str(scratch.resolve()), TMP=str(scratch.resolve()))
        run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], tmp/'contracts.log')
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return 'integrity and replay regression tests passed'


def check_full_labels(tmp: Path) -> str:
    from stage11_acceptance import check_region
    regions = sorted(p.name for p in (ROOT/'results/full/labels_v1').glob('[0-9][0-9]') if p.is_dir())
    if not regions:
        raise Skip('no fetched full-label regions')
    records = [check_region(r) for r in regions]
    sys.path.insert(0,str(ROOT/'scripts/localization'))
    from dino_embedder import load_embedding_cache
    from region_map import load_region
    from tile_index import TileSpec
    for record in records:
        if record['pass']:
            try:
                rasters,tiles=load_region(record['region'],TileSpec(1024,512))
                load_embedding_cache(ROOT/'datasets/cache/embeddings'/f"region{record['region']}_dinov2_vits14_224.npz",tiles=tiles,rasters=rasters)
                record['source_cache_identity_pass']=True
            except (OSError,ValueError,KeyError) as exc:
                record.update({'pass':False,'source_cache_identity_pass':False,'error':str(exc)})
    (tmp/'full_labels.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    failures = [r for r in records if not r['pass']]
    if failures:
        raise Fail(str(failures))
    return '; '.join(f"{r['region']}: {r['rows']} attempts, exact grid and pilot agreement PASS" for r in records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help=f"comma list of {','.join(CHECKS)}")
    ap.add_argument("--skip", default="", help="comma list to skip")
    ap.add_argument("--quick", action="store_true", help=f"skip {', '.join(SLOW)}")
    args = ap.parse_args()
    sel = args.only.split(",") if args.only else list(CHECKS)
    skip = set(filter(None, args.skip.split(","))) | (set(SLOW) if args.quick else set())
    sel = [c for c in sel if c not in skip]
    unknown = [c for c in sel if c not in CHECKS]
    if unknown:
        raise SystemExit(f"unknown check(s): {unknown}")

    tmp = ROOT / "build" / "verify" / datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp.mkdir(parents=True, exist_ok=True)
    fns = {c: globals()[f"check_{c}"] for c in CHECKS}
    results = []
    for c in sel:
        t0 = time.perf_counter()
        print(f"[{c}] ...", flush=True)
        try:
            needs(c)
            detail, status = fns[c](tmp), "PASS"
        except Skip as e:
            detail, status = str(e), "SKIP"
        except Fail as e:
            detail, status = str(e), "FAIL"
        results.append({"check": c, "pass": status == "PASS", "status": status,
                        "seconds": round(time.perf_counter() - t0, 1), "detail": detail})
        print(f"[{c}] {status} ({results[-1]['seconds']} s): {detail}", flush=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "scripts", "configs"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    rec = {"commit": commit, "dirty_scripts_or_configs": dirty, "python": sys.executable, "scratch": str(tmp),
           "all_pass": all(r["pass"] for r in results), "results": results}
    (ROOT / "build" / "verify" / "last_verify.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    failed = any(r["status"] == "FAIL" for r in results)
    skipped = sum(r["status"] == "SKIP" for r in results)
    verdict = "FAILED" if failed else ("ALL PASS" if not skipped else f"INCOMPLETE ({skipped} skipped: local files missing)")
    print(f"\n{verdict} — {sum(r['pass'] for r in results)}/{len(results)} checks passed "
          f"at {commit}{' (uncommitted changes)' if dirty else ''}; scratch {tmp.relative_to(ROOT)}")
    return 1 if failed else (2 if skipped else 0)


if __name__ == "__main__":
    raise SystemExit(main())
