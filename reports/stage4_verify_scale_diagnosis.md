# Stage 4 diagnosis: verification vs satellite crop width (region 03 tune frames)

Truth-centred crops resampled to 1024 px; frozen AKAZE/RANSAC settings; min_inliers = 12.

| Crop width | frames | median inliers | frames >= 12 inliers |
|---:|---:|---:|---:|
| 60 m | 24 | 0 | 0 |
| 150 m | 24 | 0 | 0 |
| 300 m | 24 | 0 | 2 |
| 600 m | 24 | 0 | 4 |
| 900 m | 24 | 0 | 2 |
| 1200 m | 24 | 0 | 2 |
| 1800 m | 20 | 0 | 0 |

300 m is the current tile footprint. Diagnostic only; truth-centred crops are not a localizer.
