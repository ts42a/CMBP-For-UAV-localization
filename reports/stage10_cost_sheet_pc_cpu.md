# Stage 10 cost sheet — pc_cpu

Timing from `results/stage5/pilot/outcomes.csv` (990 attempts): attempt 5.82 s, frame setup 4.67 s, frame total 57.1 s (wall).

| item | workload | hours |
|---|---|---:|
| full_labels | 6744 frames x 9 attempts | 106.9 |
| h1_synthetic | 6744 synthetic queries x 9 attempts | 106.9 |
| replay | 379,350 attempts = sum(budgets 0.05,0.1,0.2,0.4) x 6744 frames x 3 drift x 5 seeds x 5 policies | 622.4 |
| session overhead | 93 sessions (assumed 180 s (model load + data access) per session) | 4.7 |
| **total x 1.5 safety** | | **1261.3** |

Labels + H1 only (x 1.5): 320.8 h.

Weekly quota assumed 30 h (**confirm on your account**) → 42.04 quota-weeks. Rule: projected GPU hours <= 3 x the weekly GPU quota measured on the user's own account → **exceeds 3 quota-weeks**.

Replay counts are assumptions until Stage 9 fixes routes, drift levels, seeds, policies and budgets; they assume query features are cached per frame and reused across policies and seeds.
