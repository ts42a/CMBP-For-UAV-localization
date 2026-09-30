# Stage 10 cost sheet — kaggle_gpu

Timing from `F:\CMBL\results\stage10\kaggle_bench\outcomes.csv` (180 attempts): attempt 1.95 s, frame setup 1.22 s, frame total 18.8 s (wall).

| item | workload | hours |
|---|---|---:|
| full_labels | 6744 frames x 9 attempts | 35.2 |
| h1_synthetic | 6744 synthetic queries x 9 attempts | 35.2 |
| replay | 379,350 attempts = sum(budgets 0.05,0.1,0.2,0.4) x 6744 frames x 3 drift x 5 seeds x 5 policies | 207.7 |
| session overhead | 31 sessions (measured import + model load 36 s per session (data access not included)) | 0.3 |
| **total x 1.5 safety** | | **417.5** |

Labels + H1 only (x 1.5): 105.5 h.

Weekly quota assumed 30 h (**confirm on your account**) → 13.92 quota-weeks. Rule: projected GPU hours <= 3 x the weekly GPU quota measured on the user's own account → **exceeds 3 quota-weeks**.

Replay counts are assumptions until Stage 9 fixes routes, drift levels, seeds, policies and budgets; they assume query features are cached per frame and reused across policies and seeds.
