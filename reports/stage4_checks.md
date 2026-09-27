# Stage 4 checks (windowed dinov2_c2f)

**All pass: False**

## zero_offset
```json
{
  "n": 5,
  "accepted_correct": 0,
  "outcomes": {
    "accepted_correct": 0,
    "rejected": 5,
    "accepted_incorrect": 0
  },
  "per_frame": [
    [
      "03_0006.JPG",
      "rejected",
      "91.581",
      "0"
    ],
    [
      "03_0292.JPG",
      "rejected",
      "36.713",
      "5"
    ],
    [
      "03_0501.JPG",
      "rejected",
      "56.321",
      "0"
    ],
    [
      "03_0562.JPG",
      "rejected",
      "49.162",
      "0"
    ],
    [
      "03_0757.JPG",
      "rejected",
      "87.884",
      "0"
    ]
  ],
  "pass": false
}
```
## determinism
```json
{
  "identical_excluding_timing": true,
  "pass": true
}
```
## negative_control
```json
{
  "n": 5,
  "truth_within_radius": 0,
  "outcomes": {
    "accepted_correct": 0,
    "rejected": 5,
    "accepted_incorrect": 0
  },
  "pass": true
}
```
## kill_resume
```json
{
  "rows_when_killed": 4,
  "process_finished_before_kill": false,
  "rows_after_resume": 20,
  "rows_reference": 20,
  "duplicate_keys": 0,
  "identical_to_uninterrupted_excluding_timing": true,
  "pass": true
}
```
## cost
```json
{
  "mean_cpu_s_per_frame": 19.47,
  "mean_wall_s_per_attempt": 1.069,
  "mean_wall_s_per_attempt_by_radius": {
    "100": 0.292,
    "300": 2.208
  },
  "mean_frame_setup_wall_s": 1.592,
  "note": "frame_cpu_s sums CPU over all threads; embedding extraction may have run concurrently"
}
```
