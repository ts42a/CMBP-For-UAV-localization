# PC baseline sanity check — 2026-10-14

PC pilot function check: 900 attempts, 100 frames; not confirmatory full-scale evidence.
Fresh protocol-v3 CPU run; fold metrics reproduced the historical reference byte for byte.
Equal weight per eligible geographic group; G2 development group included. Undefined AUROC groups omitted.

| Model | Macro AUROC | Macro Brier | AUROC groups |
|---|---:|---:|---:|
| M0_constant | 0.5000 | 0.2846 | 6 |
| M1_texture | 0.5524 | 0.3203 | 6 |
| M2_dino | 0.6511 | 0.2873 | 6 |
| M3_dino_distinct | 0.6390 | 0.2930 | 6 |

Source: `build/verify/20261014_230356/v3/metrics_per_fold.csv`; SHA-256 `04c081cc0274b3ea6d02df8c504d46c9046232b030eb78b878402994120c9c00`.
These pilot numbers do not pass or fail the newly registered H1–H3 full-scale tests.
