# Stage 4 checks (windowed dinov2_c2f)

**All pass: True**

## zero_offset
```json
{
  "n": 5,
  "accepted_correct": 5,
  "outcomes": {
    "accepted_correct": 5,
    "rejected": 0,
    "accepted_incorrect": 0
  },
  "per_frame": [
    [
      "03_0006.JPG",
      "accepted_correct",
      "19.904",
      "31"
    ],
    [
      "03_0292.JPG",
      "accepted_correct",
      "10.146",
      "33"
    ],
    [
      "03_0501.JPG",
      "accepted_correct",
      "27.018",
      "43"
    ],
    [
      "03_0562.JPG",
      "accepted_correct",
      "10.838",
      "36"
    ],
    [
      "03_0757.JPG",
      "accepted_correct",
      "33.261",
      "25"
    ]
  ],
  "pass": true
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
  "mean_cpu_s_per_frame": 252.89,
  "mean_wall_s_per_attempt": 6.619,
  "mean_wall_s_per_attempt_by_radius": {
    "100": 1.718,
    "300": 13.442
  },
  "mean_frame_setup_wall_s": 5.41,
  "note": "frame_cpu_s sums CPU over all threads; embedding extraction may have run concurrently"
}
```
