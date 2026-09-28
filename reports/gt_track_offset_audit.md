# Ground-truth track-offset audit (read-only)

Source: `results/stage5/pilot/outcomes.csv`. One row per frame: its best patch fit with >= 16 inliers (accepted or not). Offset = CSV position -> predicted position, split into along-track (+ = ahead along the flight direction) and cross-track (+ = right of track) components. 'Far on-axis' = offset > 100 m within ±30° of the track axis; chance share is 1/3.

Labels are **not** corrected from this: using matcher output to move ground truth would be circular. Treat any signal as a ground-truth limitation to report.

| region | frames | median along-track m (95% CI) | median abs cross-track m | far (>100 m) | far on-axis |
|---|---:|---|---:|---:|---:|
| 01 | 7 | -7.1 (-46.4, 20.0) | 34.7 | 2 | 0/2 |
| 02 | 5 | 7.3 (3.8, 23.8) | 17.1 | 0 | 0/0 |
| 03 | 10 | 14.2 (0.2, 22.5) | 11.2 | 0 | 0/0 |
| 04 | 10 | 16.5 (-1.0, 32.8) | 22.9 | 0 | 0/0 |
| 05 | 10 | 7.2 (-196.5, 64.3) | 65.6 | 5 | 1/5 |
| 06 | 9 | 68.4 (-7.4, 591.4) | 26.7 | 5 | 4/5 |
| 08 | 9 | -0.2 (-15.5, 17.2) | 21.7 | 1 | 0/1 |
| 09 | 9 | 14.9 (-1.4, 27.0) | 8.6 | 1 | 0/1 |
| 10 | 4 | 14.4 (-213.9, 378.2) | 94.0 | 2 | 1/2 |
| 11 | 10 | -8.1 (-18.2, 18.6) | 37.0 | 0 | 0/0 |

All regions: 6/16 far offsets on the track axis (chance 1/3).

Far offsets:

| region | frame | inliers | accepted | offset m | offset bearing | track bearing | axis diff |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | 01_0245.JPG | 18 | 0 | 535.4 | 300.1 | 0.4 | 60.3 |
| 01 | 01_0327.JPG | 17 | 0 | 427.1 | 89.5 | 180.4 | 89.1 |
| 05 | 05_0142.JPG | 19 | 0 | 725.4 | 135.4 | 270.3 | 45.1 |
| 05 | 05_0236.JPG | 20 | 1 | 803.7 | 117.6 | 159.0 | 41.4 |
| 05 | 05_0331.JPG | 17 | 0 | 400.9 | 286.0 | 89.7 | 16.3 |
| 05 | 05_0378.JPG | 19 | 0 | 201.2 | 212.9 | 270.6 | 57.8 |
| 05 | 05_0473.JPG | 17 | 0 | 397.5 | 20.2 | 270.6 | 70.4 |
| 06 | 06_0103.JPG | 17 | 0 | 592.1 | 341.2 | 343.8 | 2.6 |
| 06 | 06_0138.JPG | 19 | 0 | 146.9 | 281.5 | 343.8 | 62.2 |
| 06 | 06_0172.JPG | 20 | 1 | 345.9 | 164.4 | 164.3 | 0.1 |
| 06 | 06_0275.JPG | 16 | 0 | 1508.0 | 334.6 | 344.3 | 9.7 |
| 06 | 06_0310.JPG | 18 | 0 | 180.5 | 139.6 | 164.8 | 25.3 |
| 08 | 08_0107.JPG | 19 | 0 | 1567.5 | 15.9 | 100.4 | 84.5 |
| 09 | 09_0766.JPG | 16 | 0 | 1762.7 | 261.3 | 129.9 | 48.6 |
| 10 | 10_0043.JPG | 17 | 0 | 505.5 | 42.9 | 157.9 | 65.0 |
| 10 | 10_0101.JPG | 17 | 0 | 420.7 | 3.2 | 337.2 | 26.0 |

Per-frame rows: `reports/gt_track_offset_audit.csv`.
