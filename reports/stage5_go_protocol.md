# Stage 5 go-decision protocol (declared before the aggregate is read)

**Date:** 2026-09-28, written while `results/stage5/pilot` is still running (regions 01-02
of 11 complete at time of writing; process still active). Declared before the pooled
Stage 5 numbers are inspected, per the pre-registration principle: fix decision rules
before seeing results.

## Disclosure (added 2026-09-28, same day)

This document was **not** written blind to all pilot data. Before it was written,
partial pilot rows for regions 01-02 (117 rows, 13 frames) had already been
tabulated during a codebase review: 38.5% accepted-correct, 61.5% rejected,
**0 accepted-incorrect**. The removal of the unregistered `all_three_outcomes_present`
gate (Section 1) was therefore made after seeing partial data that would have failed
it. The justification is that the gate was never in the frozen yaml, so the code is
being brought back to the pre-registered rule rather than loosened beyond it, but
readers should weigh that timing. No other regions (03-11) had been inspected when
this document was written.

## 1. Gating criteria — unchanged from `configs/preregistration_v1.yaml` `stage5_go`

Only two checks decide go/no-go:

- `prevalence_band`: pooled accepted-correct rate at the primary tolerance in [0.15, 0.85]
- `monotonic_check`: accepted-correct rate must not increase across the four
  prior-offset/radius bins (`0-0.5R`, `0.5-1R`, `1-1.5R`, `1.5R+`)

`scripts/analysis/summarize_windowed.py` previously also required all three outcome
types (`accepted_correct`, `rejected`, `accepted_incorrect`) to be present before
declaring `stage5_go`. That criterion is not in the frozen yaml. **Fixed 2026-09-28:**
it is now reported under `informational_not_gating` and no longer gates the decision.
It was never pre-registered and could have failed the study for an untracked reason
(e.g. zero accepted-incorrect rows given the D1 verifier's near-1.0 precision on
region 03 is not itself evidence against proceeding).

The plan prose ("success falls as radius and offset grow") is broader than the frozen
yaml, which only operationalises the *offset* trend. The radius trend is now also
reported (`accepted_correct_nonincreasing_with_radius`) but stays informational:
adding it as a new gate now, after part of the pilot (regions 01-02) has already run,
would be moving the goalposts mid-run. If it is to become a gate, that requires a
dated deviation entry written before the aggregate is read — not this document, since
this document is itself written before the aggregate is read but after two regions
have completed.

## 2. Region 03 (localizer-dev region)

The frozen `prevalence_definition` is "accepted_correct rate over all attempts... pooled",
with no region excluded, so it is used as-is for the gate. `splits_v1.json` separately
flags region 03 as `localizer_dev_contaminated` and asks for dev-contaminated folds to
be reported separately in the final analysis — not excluded from Stage 5 pooling.
`summarize_windowed.py` now additionally reports `pooled_excluding_localizer_dev_region`
for transparency; this number is informational only and never substitutes for the gate.

## 3. Is a further re-freeze available if Stage 5 fails?

Two plan passages each describe "one allowed change":

- Stage 4: "change the acceptance rule or the matcher — but never to one trained on
  UAV-VisLoc" → already exercised as deviation **D1** (verifier switch, commit `57b9df2`).
- Stage 5: "change the frozen localizer or tolerance once, document it, and re-freeze —
  never tuning on test regions."

**Decision, declared now:** D1 was triggered by, and evaluated only on, the region-03
tune split, before any multi-region Stage 5 attempt was made. It counts against the
Stage 4 lever, not the Stage 5 one. The Stage 5 lever therefore remains available,
**if and only if** the pre-registered `stage5_go` gate (Section 1) fails on this pilot.
If invoked, the allowed change is: adjust the primary tolerance and/or the accept-rule
threshold, selected on region-03 tune data only — not swap the underlying matcher
again, since UAV-VisLoc-trained alternatives are disallowed and the DINOv2 patch
verifier is the only vetted non-trained candidate. Any such change is logged as
deviation **D2**, followed by the same re-gate sequence D1 used (Stage 3B val gate
rerun + Stage 4 checks rerun) before any new Stage 5 pilot is run.

If the gate passes, no further change happens and this section is moot.

## 4. Known scope caveats about "radius" in this pilot (recorded before results are read)

- The prior offset's per-axis sigma is `0.5 x radius` (`preregistration_v1.yaml
  prior_offset`), so the true position falls inside the nominal window ≈86.5% of the
  time at every radius by construction. The three radii do not test "how far off the
  prior can be" as cleanly as the label alone suggests.
- The DINOv2 patch verifier's context crops are 2x-3x the 1024 px retrieval tile
  (`matcher_stage4_frozen.yaml` `fine_match.context_scales`), i.e. roughly 600-900 m
  wide, independent of the nominal search radius. A `radius_m: 100` attempt is
  retrieval-windowed at 100 m but fine-verified against ~600-900 m of context.

These are pre-existing properties of the frozen D1 matcher and the pre-registration,
not new bugs, and are **not** being changed now — doing so would itself be an
undeclared mid-pilot deviation. They are recorded here so the radius-stratified tables
in `reports/stage5_pilot_summary.md` are read correctly: flat or non-monotonic
behaviour across the three nominal radii is expected on these grounds alone and does
not by itself indicate a problem with the localizer.

## Status at time of writing

`results/stage5/pilot` in progress (regions 01-02 of 11 complete, process still
running). This document is committed before the aggregate summary is generated.
