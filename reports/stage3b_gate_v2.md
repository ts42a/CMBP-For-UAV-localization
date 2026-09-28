# Stage 3B validation gate — rerun with re-frozen matcher

**Gate: PASS**

Matcher `configs/matcher_stage4_frozen.yaml`, accept rule `patch_20`, region-03 val (30 frames, untouched), full-map search. Criteria from `reports/verifier_switch_protocol.md`.

| Check | Value | Pass |
|---|---|---|
| n_val_30 | 30 | yes |
| accepted_correct_ge_5 | 17 | yes |
| precision_ge_0.80 | 0.944 | yes |
| mix_present | {'rejected': 12, 'accepted_incorrect': 1} | yes |
| search_used_gt_position_0 | 0 | yes |
| exact_csv_copies_0 | 0 | yes |
| visual_ge_0.80 | {'panels': 17, 'yes': 17, 'fraction': 1.0} | yes |

Accepted-correct errors (m): [6.992, 7.81, 14.529, 16.741, 16.876, 17.857, 18.065, 20.759, 22.72, 23.669, 28.915, 29.945, 30.201, 30.869, 37.985, 39.611, 40.482]

For reference, the original Stage 3B AKAZE gate reported 9/30 within 100 m, but only 2 of those were homography-verified; the rest were unverified tile-centre fallbacks.
