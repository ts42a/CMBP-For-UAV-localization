# Frame-quality audit — label-free

`scripts/audit/frame_quality.py`. Manifest positions, times and AGL only; no localizer outcome is read. Rules: track_jump = on a straight segment (turn < 20°), departure from the time-interpolated neighbour line > max(30 m, 5 × flight median); speed_outlier = implied speed > 3 × flight median; low_agl = AGL < 150 m; turn (informational, not flagged) = heading change ≥ 20°. A first version without the straight-segment condition flagged 167 frames, all ~90° turns. Per-frame flags: `datasets/manifests/frame_quality.csv`.

| region | frames | track_jump | speed_outlier | low_agl | flagged (any) | turn (info) |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 817 | 0 | 0 | 0 | 0 | 18 |
| 02 | 1071 | 0 | 0 | 0 | 0 | 18 |
| 03 | 768 | 0 | 0 | 0 | 0 | 14 |
| 04 | 738 | 0 | 0 | 0 | 0 | 14 |
| 05 | 473 | 0 | 0 | 0 | 0 | 22 |
| 06 | 344 | 0 | 0 | 6 | 6 | 19 |
| 07 | 30 | 0 | 0 | 0 | 0 | 0 |
| 08 | 1033 | 0 | 0 | 0 | 0 | 20 |
| 09 | 766 | 0 | 0 | 0 | 0 | 10 |
| 10 | 144 | 0 | 0 | 0 | 0 | 14 |
| 11 | 590 | 0 | 0 | 0 | 0 | 24 |

Total flagged: 6 of 6774 frames (0.1%).

Region 07's CSV dates are malformed (limitation 6), so its speed and jump checks cannot run.

These flags catch position glitches and near-ground frames. They do **not** catch a smooth systematic offset of a whole track segment (e.g. region 06, `reports/gt_track_offset_audit.md`); that residual ground-truth noise is bounded separately by the label-noise sensitivity (protocol v5).
