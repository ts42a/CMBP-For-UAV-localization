#!/usr/bin/env python3
"""Label-free frame-quality flags (proposed predictor protocol v5, frame-quality sensitivity).

Uses only the manifest (CSV position, time, height) and the AGL audit; no localizer outcome is
read, so excluding flagged frames cannot select on the outcome. Flags:

  track_jump    on a straight segment (turn < 20 deg), the CSV position departs from the
                time-interpolated line between its two neighbours by more than
                max(30 m, 5 x the flight's median departure). The straight-segment condition was
                added after a first version flagged 167 frames that were all ~90 deg turns.
  speed_outlier the implied speed to either neighbour exceeds 3 x the flight's median speed
  low_agl       height above ground < 150 m (take-off/landing: the camera sees a few tens of
                metres, far below the verifier's 600-900 m context crops)
  turn          heading change between the incoming and outgoing segments >= 20 deg (banked
                turns give oblique views); informational only, not part of `flagged`
  track_edge    first or last frame of a flight (no two-sided check possible); informational only

Thresholds are fixed here, before any sensitivity result exists.

  py -3 scripts/audit/frame_quality.py
Writes datasets/manifests/frame_quality.csv and reports/frame_quality_audit.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JUMP_MIN_M, JUMP_MAD_X, SPEED_X, LOW_AGL_M, TURN_DEG = 30.0, 5.0, 3.0, 150.0, 20.0
M_LAT, M_LON_EQ = 110_574.0, 111_320.0


def main() -> int:
    m = pd.read_csv(ROOT / "datasets/manifests/manifest.csv", dtype={"region_folder": str})
    m = m[m.metadata_valid_coords == True].rename(columns={"region_folder": "region"})  # noqa: E712
    agl = pd.read_csv(ROOT / "datasets/manifests/frame_agl.csv", dtype={"region": str})
    m = m.merge(agl[["region", "filename", "agl_m"]], on=["region", "filename"], validate="one_to_one")
    out = []
    for reg, g in m.groupby("region"):
        g = g.sort_values("num").reset_index(drop=True)
        t = pd.to_datetime(g.date, errors="coerce", format="ISO8601").astype("int64").to_numpy() / 1e9
        t[t < 0] = np.nan  # unparseable dates (region 07, limitation 6): no speed/jump check
        lat0 = g.lat.mean()
        y = (g.lat.to_numpy() - lat0) * M_LAT
        x = (g.lon.to_numpy() - g.lon.mean()) * M_LON_EQ * np.cos(np.radians(lat0))
        n = len(g)
        dep = np.full(n, np.nan)
        spd_prev, spd_next = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n):
            if i > 0 and t[i] > t[i - 1]:
                spd_prev[i] = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) / (t[i] - t[i - 1])
            if i < n - 1 and t[i + 1] > t[i]:
                spd_next[i] = np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) / (t[i + 1] - t[i])
            if 0 < i < n - 1 and t[i + 1] > t[i - 1]:
                w = (t[i] - t[i - 1]) / (t[i + 1] - t[i - 1])
                xi, yi = x[i - 1] + w * (x[i + 1] - x[i - 1]), y[i - 1] + w * (y[i + 1] - y[i - 1])
                dep[i] = np.hypot(x[i] - xi, y[i] - yi)
        bearing = np.degrees(np.arctan2(np.diff(x), np.diff(y)))
        turn = np.r_[np.nan, np.abs((np.diff(bearing) + 180) % 360 - 180), np.nan]
        med_dep = np.nanmedian(dep) if np.isfinite(dep).any() else 0.0
        both = np.concatenate([spd_prev, spd_next])
        med_spd = np.nanmedian(both) if np.isfinite(both).any() else np.inf
        jump_thr = max(JUMP_MIN_M, JUMP_MAD_X * med_dep)
        for i in range(n):
            out.append({
                "region": reg, "filename": g.filename[i], "num": int(g.num[i]),
                "departure_m": round(float(dep[i]), 2) if np.isfinite(dep[i]) else "",
                "speed_mps_max": round(float(np.nanmax([spd_prev[i], spd_next[i]])), 2)
                if np.isfinite([spd_prev[i], spd_next[i]]).any() else "",
                "turn_deg": round(float(turn[i]), 1) if np.isfinite(turn[i]) else "",
                "track_jump": int(np.isfinite(dep[i]) and dep[i] > jump_thr and np.isfinite(turn[i]) and turn[i] < TURN_DEG),
                "speed_outlier": int(np.nanmax([spd_prev[i], spd_next[i], 0.0]) > SPEED_X * med_spd),
                "low_agl": int(g.agl_m[i] < LOW_AGL_M),
                "turn": int(np.isfinite(turn[i]) and turn[i] >= TURN_DEG),
                "track_edge": int(i in (0, n - 1)),
            })
    q = pd.DataFrame(out)
    q["flagged"] = ((q.track_jump + q.speed_outlier + q.low_agl) > 0).astype(int)
    q.to_csv(ROOT / "datasets/manifests/frame_quality.csv", index=False)

    s = q.groupby("region")[["track_jump", "speed_outlier", "low_agl", "flagged", "turn"]].sum()
    s["frames"] = q.groupby("region").size()
    md = ["# Frame-quality audit — label-free", "",
          "`scripts/audit/frame_quality.py`. Manifest positions, times and AGL only; no localizer outcome is read. "
          f"Rules: track_jump = on a straight segment (turn < {TURN_DEG:.0f}°), departure from the time-interpolated neighbour line > max({JUMP_MIN_M:.0f} m, "
          f"{JUMP_MAD_X:.0f} × flight median); speed_outlier = implied speed > {SPEED_X:.0f} × flight median; "
          f"low_agl = AGL < {LOW_AGL_M:.0f} m; turn (informational, not flagged) = heading change ≥ {TURN_DEG:.0f}°. "
          "A first version without the straight-segment condition flagged 167 frames, all ~90° turns. Per-frame flags: `datasets/manifests/frame_quality.csv`.", "",
          "| region | frames | track_jump | speed_outlier | low_agl | flagged (any) | turn (info) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r, x in s.iterrows():
        md.append(f"| {r} | {x.frames} | {x.track_jump} | {x.speed_outlier} | {x.low_agl} | {x.flagged} | {x.turn} |")
    md += ["", f"Total flagged: {int(q.flagged.sum())} of {len(q)} frames ({q.flagged.mean():.1%}).", "",
           "Region 07's CSV dates are malformed (limitation 6), so its speed and jump checks cannot run.", "",
           "These flags catch position glitches and near-ground frames. They do **not** catch a smooth systematic "
           "offset of a whole track segment (e.g. region 06, `reports/gt_track_offset_audit.md`); that residual "
           "ground-truth noise is bounded separately by the label-noise sensitivity (protocol v5)."]
    (ROOT / "reports/frame_quality_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
