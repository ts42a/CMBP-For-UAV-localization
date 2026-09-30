# Technical review — CMBL

Reviewed 30 September 2026. Baseline: commit `2bba012`, plus the existing working-tree edits. This was a review, not a remediation: application code, frozen protocols and existing research results were not changed.

## Overall assessment

**Suitable for supervised research pilots; not yet ready for unattended production or operational UAV navigation.** The project has unusually explicit research protocols, geographic splits, deviation records, pilot evidence and reproduction checks. Its most important weaknesses are silent data-integrity failures at stage boundaries. Passing the existing checks does not establish that arbitrary new, resumed or downloaded runs are valid.

This is a Python research pipeline, not a full-stack service. Database optimization, HTTP authorization and web-session security are not applicable to the inspected architecture. Security attention should focus on downloaded code, models, archives, credentials and artifact provenance. This review does not certify compliance with a security standard or establish absence of dependency vulnerabilities.

No Critical issue was established. High findings below concern scientific-data correctness and expensive-run integrity, not demonstrated remote exploitation.

## Scope and verification

Inspected the README, frozen protocols, split generation, windowed labeller, retrieval and patch verifier, map feature extraction, predictor/calibration evaluator, Kaggle packaging/runner/acceptance checks, archive handling and verification scripts. Parsed all 65 Python files under `scripts/` and `kaggle/` successfully. This was not a line-by-line audit of every historical audit script or external dependency.

Executed `verify.py --quick`: **4/4 PASS** — environment (16 package pins, Python 3.14.7, one OpenCV distribution), frozen-file checks, split regeneration and byte-for-byte predictor-v3 pilot reproduction. The leakage-control macro AUROC was 0.506 and passed its registered criterion.

Executed the remaining checks separately: **3/3 PASS**. Predictor v4 preserved training predictions and reproduced the exclusion/sensitivity metrics; v5 preserved the registered predictions and added both diagnostic models. Golden-label reproduction produced 18 attempts with identical outcomes, matching configuration hashes and a maximum position difference of 0.0 m. **All seven existing checks passed across the two invocations.** Logs are in `build/verify/20260930_121902/` and `build/verify/20260930_122035/` (git-ignored).

Two isolated reproductions executed the actual functions extracted from the source, using temporary inputs and mocked file providers, without loading a model or changing real research data:

- `done_keys` accepted a truncated CSV row containing only the five key fields as a completed attempt.
- `check_labels` returned `pass=True` for nine unique attempts with an invalid variant and offset grid, and zero pilot overlap.

These demonstrate validation defects, not evidence that the existing committed pilot outputs are corrupt. No cloud jobs were submitted or changed, and the current remote Stage 11 status was not independently verified.

## Findings

### R1 — High: full-scale acceptance permits the wrong experiment

**Location:** [kaggle_stage10.py:352](F:/CMBL/scripts/protocol/kaggle_stage10.py:352), especially the `complete` predicate and `pilot_ok` at line 386.

**Problem:** Acceptance checks the total row count, filename set and key uniqueness, but not equality with the expected `(region, filename, variant, radius, offset)` grid. Nine arbitrary unique keys can replace the registered three radii times three offsets. Pilot agreement explicitly succeeds when there is no overlap. It also does not validate the row-level preregistration hash, acceptance rule or outcome domain.

**Evidence:** A synthetic one-frame input with nine `wrong_variant` attempts, a single radius and offsets 0–8 passed. `pilot_overlap_attempts` was zero. The committed protocol requires pilot reproduction and the registered grid.

**Recommendation:** Derive the complete expected key set from the frozen protocol and manifest, require exact equality, require every expected pilot key, validate region/config/preregistration/acceptance-rule fields, and validate outcome/coordinate consistency. Treat missing overlap as failure.

**Expected impact:** Prevents incorrect or incomplete experiments from receiving the Stage 11 release gate. Add a regression fixture for the demonstrated bypass before accepting additional full-scale artifacts.

### R2 — High: resume can mix experiments and replace their provenance

**Location:** [label_windowed.py:188](F:/CMBL/scripts/label_windowed.py:188), manifest write at line 195 and skip logic in the frame loop.

**Problem:** `--resume` trusts existing keys without comparing the previous configuration, preregistration, acceptance rule, synthetic protocol, split identity or code identity. It then overwrites `run_manifest.json`. Changing the acceptance rule or a protocol and resuming skips old keys, appends new rows under new semantics and presents the latest invocation as the run manifest. A fully completed run can have its manifest replaced without producing a single new row.

**Recommendation:** Validate an immutable experiment fingerprint before opening outputs. Preserve the initial manifest and append resume-event records. Require a new output directory for incompatible settings, and validate existing rows against the original manifest.

**Expected impact:** Prevents silent mixed-label datasets and preserves an auditable history of interrupted runs.

### R3 — High: interrupted writes can permanently skip incomplete attempts

**Location:** [label_windowed.py:138](F:/CMBL/scripts/label_windowed.py:138), CSV append at line 299.

**Problem:** `csv.DictReader` fills missing trailing columns with `None`; `done_keys` only inspects the leading key fields. A terminated write can therefore leave a row with a complete key but no outcome that resume treats as finished. Appending without repairing an unterminated final record can also join the next record to it. Flushing each row does not make the record atomic.

**Evidence:** A CSV ending after `01,a.jpg,prior,100,0` was accepted as one completed attempt by the actual function.

**Recommendation:** Validate schema, field count and required values before marking a row complete. Detect and quarantine/truncate an incomplete tail safely. Prefer atomic per-frame files or a transactional attempt store with a unique key; export CSV as an interchange format.

**Expected impact:** Makes recovery safe after interruption. Test mid-record termination as well as the existing after-frame kill/resume scenario.

### R4 — High: unknown outcomes silently become negative training labels

**Location:** [map_features.py:117](F:/CMBL/scripts/predictor/map_features.py:117).

**Problem:** The label is computed as equality to `accepted_correct`. Blank, misspelled or unsupported outcomes all become zero. The labeller intentionally permits blank outcomes with `--accept-rule none`, so this is reachable through a supported command, even without file corruption.

**Recommendation:** Accept only the three registered outcome values; reject missing labels before feature generation. Check the acceptance rule, tolerance, unique attempt keys and outcome-run fingerprint. Keep explicitly unlabelled attempts outside supervised feature outputs.

**Expected impact:** Prevents silently converting unlabelled or invalid data into failures and biasing prevalence, calibration and model evaluation.

### R5 — High: embedding caches are validated by count rather than identity

**Location:** [label_windowed.py:233](F:/CMBL/scripts/label_windowed.py:233), [map_features.py:97](F:/CMBL/scripts/predictor/map_features.py:97), [extract_all_reference_embeddings.py:63](F:/CMBL/scripts/extract_all_reference_embeddings.py:63).

**Problem:** Consumers discard the returned tile IDs and metadata and check only the number of embeddings. Cache reuse in extraction uses the same count-based criterion. A reordered cache or a different raster, grid or model producing the same count is accepted, associating descriptors with the wrong coordinates. Rich tile metadata already exists but is not enforced.

**Recommendation:** Validate ordered tile identity, geometry, raster-content identity, preprocessing, model revision and weight checksum. Include these in a cache fingerprint and refuse incompatible reuse; record the consumed cache fingerprint in every downstream run.

**Expected impact:** Prevents spatially incorrect retrieval and feature generation that can otherwise look like valid scientific results.

### R6 — High: feature rows and embedding rows have no verified association

**Location:** [map_features.py:149](F:/CMBL/scripts/predictor/map_features.py:149), [evaluate_predictors.py:68](F:/CMBL/scripts/predictor/evaluate_predictors.py:68), evaluator manifest at line 424.

**Problem:** The CSV and NPZ are separate positional arrays. Their association is documented as “same as features.csv,” but the evaluator does not verify keys or both file hashes. Sorting/editing the CSV independently, or copying an NPZ from another run of the same size, silently pairs incorrect labels and descriptors. Group assignments are also trusted from the CSV rather than checked against region membership in the chosen split scheme. The evaluator records the CSV hash but not the NPZ hash.

**Recommendation:** Store ordered attempt keys in the NPZ or use one keyed artifact. Validate equal dimensions, exact ordered key identity, uniqueness, finite values, label domain and region-to-group membership. Hash all inputs, including embeddings, splits, quality flags and AGL metadata.

**Expected impact:** Prevents misalignment and incorrectly assigned hold-outs; makes predictor results reproducible from their manifests.

### R7 — Medium: the frozen model loads code from an unpinned upstream revision

**Location:** [dino_embedder.py:15](F:/CMBL/scripts/localization/dino_embedder.py:15), `torch.hub.load` at line 36.

**Problem:** `facebookresearch/dinov2` has no revision pin and is loaded with `trust_repo=True`. A fresh cache can obtain different upstream model code while the project still claims the same frozen localizer. Trusting the repository author does not identify the precise code that ran. Kaggle packaging records a weight checksum, but the local model loader does not enforce it.

**Recommendation:** Pin a reviewed upstream revision; verify a recorded weights digest and repo revision before use, including offline bundles. Record both in run/cache manifests. Add dependency vulnerability scanning separately from numerical reproduction checks.

**Expected impact:** Improves reproducibility and constrains the model supply-chain trust boundary. PyTorch documents that Hub loads repository code and that the trust setting governs execution of trusted repositories: [official Hub documentation](https://docs.pytorch.org/docs/2.14/hub.html).

### R8 — Medium: archive import is not isolated from the workspace

**Location:** [stage10_kernel.py:70](F:/CMBL/kaggle/stage10_kernel.py:70), [kaggle_stage10.py:393](F:/CMBL/scripts/protocol/kaggle_stage10.py:393).

**Problem:** Kaggle results are extracted directly into the repository root after deleting the old region output. Even a normal ZIP member such as `scripts/label_windowed.py` can target an unrelated workspace file without needing traversal. A corrupt archive can leave the old result removed and an incomplete replacement. TAR extraction uses the interpreter's default filter; the recorded Kaggle environment is Python 3.12.13, whereas the local environment is 3.14.7.

**Recommendation:** Extract into a new staging directory; allow only the expected result prefix and regular files; reject unexpected links and oversized content; validate checksums and schema before replacing an output. Retain the previous result until replacement succeeds. Explicitly use an appropriate TAR data filter across supported Python versions.

**Expected impact:** Limits accidental or malicious overwrite and makes downloads recoverable. This is a trust-boundary weakness, not evidence of a malicious bundle. Python documents the safer default beginning in 3.14 and recommends explicit filtering for earlier versions: [tarfile documentation](https://docs.python.org/3/library/tarfile.html#extraction-filters).

### R9 — Medium: verification lacks portable negative-path regression coverage

**Location:** [verify.py:44](F:/CMBL/scripts/protocol/verify.py:44), [stage4_checks.py:84](F:/CMBL/scripts/protocol/stage4_checks.py:84); repository test/CI layout.

**Problem:** Useful integration and golden checks exist, so this is not an untested project. However, most require local-only data/artifacts; a public clone cannot exercise them without reconstruction. No tracked dedicated unit-test suite or CI workflow was found. The existing interruption check kills after complete rows and does not cover R3. Baseline reproduction does not challenge malformed inputs, stale caches, mixed provenance or the acceptance bypass.

**Recommendation:** Add a small distributable synthetic fixture suite for R1–R6, empty/single-class folds, missing files and multi-raster coordinate handling. Run it in CI alongside syntax/static checks. Keep data-dependent golden checks as a separate tier. Measure coverage of critical contracts rather than claiming a percentage from file inspection.

**Expected impact:** Finds the demonstrated failures cheaply, allows external contributors to validate changes and protects experimental invariants.

### R10 — Medium: executable behavior is duplicated across code and prose configuration

**Location:** [evaluate_predictors.py:85](F:/CMBL/scripts/predictor/evaluate_predictors.py:85), bootstrap parsing at lines 209 and 352, protocol dispatch at line 190.

**Problem:** PCA dimensions, logistic-regression parameters, clipping and bin counts are hardcoded while YAML describes them separately. Bootstrap parameters are parsed from English strings. Unknown protocol versions fall back to v1-style behavior instead of being rejected. The large evaluator `main` combines protocol interpretation, fitting, controls, statistics and report writing; widespread path injection makes isolated imports harder.

**Recommendation:** Introduce validated, versioned configuration schemas and reject unknown versions. Either consume typed parameters or explicitly validate constants against the frozen protocol. Separate data validation, fold fitting, evaluation and reporting behind stable interfaces; package common code so CLIs remain thin. Preserve registered numerical behavior and log any scientific changes as deviations.

**Expected impact:** Reduces configuration drift and makes tests/refactoring safer. The current functional design does not need an elaborate class hierarchy to improve responsibility separation or dependency injection.

### R11 — Medium: repeated feature computation limits scale

**Location:** [label_windowed.py:239](F:/CMBL/scripts/label_windowed.py:239), [patch_verify.py:148](F:/CMBL/scripts/localization/patch_verify.py:148), [evaluate_predictors.py:99](F:/CMBL/scripts/predictor/evaluate_predictors.py:99).

**Problem:** The per-frame cache clear discards satellite patch features that are independent of the query, so neighboring frames can repeat model inference on the same map crops. Predictor fitting recomputes the same scaler/PCA transforms for M2, M3 and M5 with identical training rows. Region-window lookup also scans all tile centers, which is reasonable at current map sizes but scales linearly per attempt.

**Recommendation:** Profile representative full runs. Separate bounded satellite-feature caching from query-specific match caching; retain the latter's frame lifetime. Reuse transformations only for exactly identical training row sets, with fold-specific keys. Consider a spatial index only if measured window-search cost warrants it.

**Expected impact:** Potentially reduces repeated inference and preprocessing without changing the experiment. No speedup is claimed without profiling and numerical-agreement checks. Keep caches bounded and avoid cross-fold fitting leakage.

### R12 — Medium: clean-machine reconstruction conflicts with frozen raw-file hashes

**Location:** [README.md:89](F:/CMBL/README.md:89), [make_splits_v1.py:106](F:/CMBL/scripts/protocol/make_splits_v1.py:106), [package_kaggle.py:66](F:/CMBL/scripts/protocol/package_kaggle.py:66).

**Problem:** The README correctly says a rebuilt manifest contains machine-specific absolute paths and therefore a different SHA256. Split verification nevertheless compares the whole regenerated payload including that SHA256; Kaggle packaging rejects the changed manifest. Following the fresh-clone reconstruction instructions therefore cannot reproduce the original frozen checks without the original local-only manifest.

**Recommendation:** Separate portable dataset/assignment identity from machine-local paths. Hash canonical, sorted semantic fields using relative image identifiers, and maintain a separate mapping to local paths. Provide an explicit migration/provenance record rather than silently replacing frozen hashes. Document one tested clean-machine reconstruction path and the selected protocol for a full run.

**Expected impact:** Enables independent reproduction while preserving evidence that the actual scientific records and assignments are unchanged.

### R13 — Medium: RQ3 has a proposed protocol but no end-to-end replay implementation

**Location:** [README.md:9](F:/CMBL/README.md:9), [stage9_protocol_v1.yaml:2](F:/CMBL/configs/stage9_protocol_v1.yaml:2), `scripts/analysis/` and `scripts/h1/`.

**Problem:** Route-aware localization maintenance is a stated research objective. The repository contains reach analysis and a proposed replay protocol, but no completed policy/estimator/replay engine was found. The protocol also retains an unresolved outcome-source decision. Synthetic-query generation exists, but a generator alone is not evidence for H1. The README discloses these stages as pending; this is an acknowledged requirement gap, not a misleading completion claim.

**Recommendation:** Finalize and approve replay choices before generating comparative outcomes. Implement recorded-route replay, matched actual-attempt budgets, paired seeds, false-fix propagation, truth-access boundaries and lookup-vs-live fidelity checks. Add a tiny deterministic end-to-end route fixture. Validate whether fitted model persistence/reusable inference interfaces are needed before connecting scheduling.

**Expected impact:** Completes RQ3 and prevents pilot label prediction from being presented as evidence of navigation or scheduling benefit.

### R14 — Medium: headline comparative conclusions lack corresponding uncertainty estimates

**Location:** [evaluate_predictors.py:350](F:/CMBL/scripts/predictor/evaluate_predictors.py:350), H3 comparison at lines 419 and 510.

**Problem:** Bootstrap intervals are computed for pooled absolute metrics, while headline conclusions use macro metrics and whether calibrated Brier/log loss is numerically lower than raw. There is no paired interval for the macro improvement underlying that conclusion. Independent bootstrap draws for each model/variant also do not directly estimate a paired difference. Existing reports can describe observed changes, but cannot use these outputs alone to establish their robustness.

**Recommendation:** Add paired region-group resampling for the prespecified macro contrasts, preserving the same sampled groups across compared methods. Report groupwise changes, fold counts and uncertainty; keep undefined single-class ranking metrics explicit. With only seven groups, state the limited precision and avoid treating repeated attempts as independent geographic replications. Register any new inferential decision rule before outcome inspection; retain the original descriptive comparisons.

**Expected impact:** Aligns uncertainty reporting with the actual research claim without converting exploratory improvements into unsupported hypothesis conclusions.

### R15 — Low: full-run reports retain pilot-only scope text

**Location:** [evaluate_predictors.py:386](F:/CMBL/scripts/predictor/evaluate_predictors.py:386), Markdown text at line 474.

**Problem:** Both the JSON scope and Markdown introduction always say the results are a 10-frame-per-region Stage 5 pilot, even when passed full-scale feature files. Version comments also still describe v5 as proposed despite its frozen status in YAML.

**Recommendation:** Require explicit run scope or derive it from validated input provenance; render actual counts, protocol status and exclusions. Update stale comments and provide full-run commands separately from the pilot recipe.

**Expected impact:** Prevents mislabelling future full-scale results and improves onboarding and report interpretation.

## Assessment across the requested dimensions

| Dimension | Assessment |
|---|---|
| Code quality and architecture | Sensible modules for geometry, retrieval, verification and evaluation. Shared data contracts and CLI separation need improvement; R10 is the main maintainability issue. Avoid unnecessary framework rewrites. |
| Functionality and requirements | The real-label and predictor/calibration pilot paths exist and reproduce. RQ1/RQ2 are not established full-scale findings yet. RQ3 and H1 completion remain explicit gaps; R13. |
| Performance and scalability | Good use of frozen global embeddings, batched embedding extraction, memory-mapped rasters and per-frame reuse. Remaining repeated computation merits profiling; R11. No database exists to optimize. |
| Security and best practices | YAML uses safe loading, most subprocess calls use argument arrays, and credentials are excluded from version control. Unpinned executable model code and workspace-wide archive import remain important boundaries; R7–R8. No web authentication layer is required by the current offline tool design. |
| Testing and reliability | Reproduction and geographic-split checks are valuable. R1–R6 expose untested invalid-input and interruption paths. Golden-output agreement alone is insufficient to validate new artifacts. |
| Documentation and maintainability | Strong stage/status documentation, preregistration and deviation ledger. Portable reconstruction and run-scope reporting need repair; R12/R15. |

## Prioritized recommendations

1. **Before accepting more full-scale results:** fix R1–R6 and add regression fixtures. Revalidate existing full-scale files without modifying labels or retuning the localizer. A failed new integrity check should trigger diagnosis and provenance review, not automatic scientific parameter changes.
2. **Before treating the pipeline as reproducible infrastructure:** pin model artifacts, isolate archive imports, add portable CI tests and resolve manifest portability (R7–R9, R12).
3. **Before drawing full-study conclusions:** complete the approved replay/H1 work, align uncertainty reporting with headline comparisons and correct report scope (R13–R15).
4. **Then improve developer efficiency and throughput:** introduce validated configuration and fold-safe reusable transformations; benchmark bounded map-feature caching (R10–R11).

Production readiness depends on the target. For offline supervised research, the project is a credible foundation once the data-integrity blockers are repaired. For unattended full-scale processing it currently needs stronger contracts, recovery and artifact validation. For use aboard a UAV, this repository provides no demonstrated real-time integration, operational fail-safe behavior or flight validation; research benchmark agreement is not an operational readiness test.

