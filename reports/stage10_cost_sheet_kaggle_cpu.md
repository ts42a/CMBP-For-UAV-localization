# Stage 10 cost sheet — kaggle_cpu

Timing from `results/stage10/kaggle_cpu_bench/outcomes.csv` (180 attempts): attempt 10.38 s, frame setup 8.13 s, frame total 101.5 s (wall).

| item | workload | hours |
|---|---|---:|
| full_labels | 6744 frames x 9 attempts | 190.2 |
| h1_synthetic | 6744 synthetic queries x 9 attempts | 190.2 |
| replay | 379,350 attempts = sum(budgets 0.05,0.1,0.2,0.4) x 6744 frames x 3 drift x 5 seeds x 5 policies | 1108.5 |
| session overhead | 166 sessions (measured import + model load 35 s per session (data access not included)) | 1.6 |
| **total x 1.5 safety** | | **2235.7** |

Labels + H1 only (x 1.5): 570.5 h.

Weekly quota assumed 30 h (**confirm on your account**) → 74.52 quota-weeks. Rule: projected GPU hours <= 3 x the weekly GPU quota measured on the user's own account → **exceeds 3 quota-weeks**.

Replay counts are assumptions until Stage 9 fixes routes, drift levels, seeds, policies and budgets; they assume query features are cached per frame and reused across policies and seeds.
