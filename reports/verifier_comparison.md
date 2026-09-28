# Verifier comparison — region 03 tune frames only

Protocol: `reports/verifier_switch_protocol.md` (committed before any run). Tolerance 100 m.
A = 216 pre-registered windowed attempts (24 frames x radii 100/300/1000 m x 3 offsets). B = zero-offset, R = 100 m, 24 frames.

| Verifier | Rule | A accepted | A acc. correct | A acc. incorrect | A precision | A frames w/ correct | A median err (accepted) | A p90 err | B acc. correct /24 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| akaze | homography_12 | 7 | 7 | 0 | 1.0 | 1 | 12.2 | 12.2 | 1 |
| patch_s1 | patch_10 | 212 | 167 | 45 | 0.7877 | 24 | 21.6 | 430.4 | 24 |
| patch_s1 | patch_15 | 172 | 153 | 19 | 0.8895 | 22 | 15.9 | 238.5 | 22 |
| patch_s1 | patch_20 | 95 | 93 | 2 | 0.9789 | 14 | 15.6 | 35.2 | 12 |
| patch_s1 | patch_30 | 15 | 15 | 0 | 1.0 | 2 | 35.7 | 35.7 | 2 |
| patch_s2 | patch_10 | 215 | 195 | 20 | 0.907 | 24 | 23.6 | 46.5 | 24 |
| patch_s2 | patch_15 | 205 | 192 | 13 | 0.9366 | 24 | 22.0 | 42.7 | 24 |
| patch_s2 | patch_20 | 185 | 185 | 0 | 1.0 | 24 | 20.1 | 36.6 | 24 |
| patch_s2 | patch_30 | 144 | 144 | 0 | 1.0 | 20 | 19.9 | 36.6 | 18 |
| patch_s3 | patch_10 | 215 | 185 | 30 | 0.8605 | 24 | 22.6 | 352.7 | 24 |
| patch_s3 | patch_15 | 191 | 166 | 25 | 0.8691 | 21 | 21.9 | 352.7 | 19 |
| patch_s3 | patch_20 | 123 | 123 | 0 | 1.0 | 17 | 21.2 | 31.1 | 15 |
| patch_s3 | patch_30 | 34 | 34 | 0 | 1.0 | 5 | 17.2 | 21.5 | 4 |
| patch_s23 | patch_10 | 215 | 196 | 19 | 0.9116 | 24 | 22.4 | 46.5 | 24 |
| patch_s23 | patch_15 | 210 | 195 | 15 | 0.9286 | 24 | 22.2 | 42.7 | 24 |
| patch_s23 | patch_20 | 188 | 188 | 0 | 1.0 | 24 | 20.1 | 36.6 | 24 |
| patch_s23 | patch_30 | 144 | 144 | 0 | 1.0 | 20 | 19.9 | 36.6 | 18 |

**Selected by the pre-registered criterion:** `patch_s23` with `patch_20`

| Criterion | Value | Pass |
|---|---|---|
| 1_accepted_correct_ge_2x_akaze | 188 (need 14) | yes |
| 2_precision_ge_0.90 | 1.0 | yes |
| 3_frames_with_correct_ge_8 | 24 | yes |
| 4_zero_offset_ge_12_of_24 | 24 | yes |
| 5_visual_ge_0.90 | {'panels': 46, 'scene_match_yes': 46, 'fraction': 1.0} | yes |

**Decision: IMPROVES**
