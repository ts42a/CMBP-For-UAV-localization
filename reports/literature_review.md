# Literature Review — Stage 1

**Access date:** 2026-09-22  
**Scope:** Named priors for map-based UAV localization-success prediction; one focused search for localizability/matchability predictors that could invalidate the proposed gap.  
**Bound:** Findings are limited to retrieved and reviewed material below. This is not a proof that no relevant paper exists outside the search.

## Document versions and access

| Paper | Local file(s) | Version / access notes |
|---|---|---|
| RSPNet (Li et al.) | `sources/RSPNet_Li2023_rs15174353_webfetch.txt` (+ doi webfetch) | MDPI full HTML via WebFetch; direct PDF/HTML download blocked (403/Akamai). Identity verified: *Remote Sens.* 2023, 15(17), 4353, DOI 10.3390/rs15174353 |
| Jiang et al. 2024 | `sources/Jiang2024_rs16010161_webfetch.txt` | Same access path; DOI 10.3390/rs16010161 |
| Robust Belief Roadmap | `sources/RobustBeliefRoadmap_1304.7256.pdf` / `.html` | arXiv:1304.7256 |
| SafeVPR | `sources/SafeVPR_2605.28048.pdf` / `.html` | arXiv:2605.28048 |
| SASGeo | `sources/SASGeo_2607.07737.pdf` / `.html` | arXiv:2607.07737 |
| MLSNet | `sources/MLSNet_SciOpen.html`, `MLSNet_full_text_api.json`, `MLSNet_openalex.json` | **Full text unavailable.** SciOpen PDF 502; OpenAlex `has_fulltext=false`, `pdf_url=null`, `any_repository_has_fulltext=false`. Abstract + metadata only |
| Search candidates | `sources/search_*.txt` | VPR uncertainty; VPR integrity+navigation; LiDAR localizability |

Details: `sources/ACCESS_LOG_2026-09-22.txt`.

---

## 1. RSPNet — Li et al., 2023

**Identity match:** Confirmed (title, authors Xin Li / Guo Zhang / Hao Cui / Jinhao Ma / Wei Wang, MDPI Remote Sensing 15(17):4353).

### Labels and outcomes
- Matchability quantified as empirical matching probability \(L_T = R_s / K\) over \(K\) simulated distorted queries (§3.1, Eq. 1, Fig. 2).
- Perturbations: colour/greyscale, Gaussian / salt-and-pepper / speckle noise (ratio 0.01), projection (0.1), brightness/colour/contrast (0.1) with random coefficients (§3.1).
- Matchers for labels: **SIFT-FLANN** and **AKAZE-BFM**. Correct if distance between correct and estimated position \(< E\); **numeric \(E\) not stated** in §3.1.
- Authors state hang-flight experiments are more precise/objective but costly; **labels come from simulation**, not hang-flight (§3.1).
- Classical baseline comparison uses matching error **< 3 pixels** (§4.2) — do not equate that to the unstated \(E\) without further evidence.

### Split and geography
- WHU-SCENE-MATCHING: 40,000 optical 600×600 images at 2 m; covers parts of India, western Tibet, Eastern Hubei (Fig. 7, §4.1).
- 80% train / 20% test; **training and test divided into two separate regions** to reduce correlation (§4.1).
- **Interpretation caution:** geographically distinct coverage boxes and an author-stated region split are not independent proof that folders/patches are statistically independent flight regions.

### Metrics
- Regression: MAE, MSE, RMSE, \(R^2\) (Table 2; RSPNet MAE **0.6581**, \(R^2\) **0.9154**).
- Binary: matchability grade threshold **5**; accuracy **0.96**, F1, AUC (Table 3 / conclusions).
- Classical: NMA / PMA (Table 1).

### Decisions / calibration / code
- Route-planning application suggested in conclusions; **no evaluated scheduling policy** in reviewed sections.
- Probability **calibration** (ECE, Brier, reliability diagrams) **not reported** in reviewed §§3–6 / tables.
- **Code:** not reported in reviewed end matter (no Data Availability Statement found after Acknowledgments/Conflicts).
- **WHU-SCENE-MATCHING:** described in paper; **not found** on `https://gpcv.whu.edu.cn/data/` (2026-09-22). Status: **public download path not found** (distinct from “repository URL exists but empty”).
- **Name collision:** `PeihaoChen/RSPNet` is an unrelated AAAI 2021 video paper.

---

## 2. Jiang, Luo, Liu — 2024

**Identity match:** Confirmed (Supeng Jiang, Haibo Luo, Yunpeng Liu; *Remote Sens.* 16(1):161).

### Trace of abstract “top three … 0.8549, 0.7993, 0.7803”
- Abstract claims average matching accuracy of top three areas under multiple matching algorithms.
- Methods: select top-3 non-overlapping \(50\times50\) windows by score \(C\) on each of 500 references; evaluate with NCC, DDIS-C, DDIS-D, CoTM (§4.3).
- **Table 4** reports AUC by algorithm and an **Average** column: No.1 **0.8549**, No.2 **0.7993**, No.3 **0.7803**.
- Separately, top-10% / top-15% vs hierarchical screening: matching accuracy **0.7895** / **0.7664** (22.2% higher than HS for top 15%) under NCC (§4.3) — different experiment from the 0.8549 triple.

### Real vs simulated
- Reference set: **500 satellite images** 400×600, built by authors (§4.1).
- Queries for quantitative matching: **simulated** real-time images (noise, blur, deformation; Fig. 4). Authors explicitly note differences from actual captured images (§4.1).
- Aerial and near-infrared examples in **Fig. 11** are qualitative versatility checks, **not** the primary quantitative protocol.
- **Not** real UAV localization outcomes as labels for a learned predictor (MSAS is handcrafted scoring).

### Success criterion and split
- Overlap \(ol = |R_{est}\cap R_{truth}| / |R_{est}\cup R_{truth}|\); success if \(ol > 0.9\) (§4.2).
- Scene-category analysis (Table 3); **geographic held-out-region protocol not reported** in reviewed §4.
- Data Availability: **“Data sharing is not applicable to this article.”** Code not reported.

### Calibration / decisions
- Reports ROC/AUC and matching probability of ranked areas — **not** probability calibration of a predictor.
- Planning/yaw reduction is motivation; **no evaluated numeric-probability scheduling**.

---

## 3. MLSNet — Yang et al., CJA 2026 (full text unavailable)

**Identity match (metadata):** Chao Yang, Yuanxin Ye, Renyuan Liu, Chengjia Fan, Liang Zhou, Jiwei Deng; *Chinese Journal of Aeronautics* 39(2):103719; DOI 10.1016/j.cja.2025.103719; OpenAlex W4412713825.

**Reviewed material only:** SciOpen abstract + landing HTML + OpenAlex record + CRediT notes from SciOpen API. **Methods, experiments, tables, and supplementary were not readable.**

### What can and cannot be said
- Abstract: MLSNet predicts **matchability segmentation maps** from reference imagery; labels “quantify matchability levels”; tested on remote sensing images; reports superiority in accuracy/efficiency vs traditional and DL methods.
- **Do not claim** that probability calibration, held-out-region protocols, real UAV outcomes, or scheduling evaluations are absent — only that they are **not established from reviewed material**.
- Proposal’s optical-vs-SAR / 9:1 / F1 claims are **not verified** here against full text.
- Unrelated `minghaidea/MLSNet` (TFBS genomics) must not be cited as this paper’s code.

**Residual risk:** MLSNet internals remain a verification gap for Stage 1 completeness.

---

## 4. Robust Belief Roadmap

**Identity match:** Shaunak D. Bopardikar, Brendan J. Englot, Alberto Speranzon; arXiv:1304.7256 (ICRA 2014 version titled with “Uncertain and Intermittent Sensing”).

### What is predicted / represented
- Analytical bound on EKF belief under **stochastic sensor misdetection** with known detection probability \(p_j\) (Assumptions 1–2).
- Path planning selects routes minimizing **goal-state uncertainty** trading accuracy and robustness (§I–IV).

### Decisions and probabilities
- **Evaluated decisions:** computational path selection under intermittent sensing (§V).
- Uses **numeric \(p_j\)**, but these are **assumed sensing probabilities**, not learned map-based localization-success probabilities.
- Evaluation is **simulation**, not real UAV imagery.

### Geographic generalization
- Not an image-region holdout study.

---

## 5. SafeVPR (proposal “SAFEVPR”)

**Identity match:** Ha Sier et al.; arXiv:2605.28048; code URL https://github.com/Hasar12139/SafeVPR.

### What is predicted / decisions
- After sequence VPR retrieval, decide **accept vs reject** the top-1 candidate.
- Score: frozen DINOv2 patch **MNN + Lowe-ratio**; calibration: **Mondrian conformal LTT** for FDR control.
- Under condition shift, reports **empirical FDR validity** (23/23 at \(\alpha=0.10\)), mean accepted FDR 0.014, mean TPR 0.75 (Abstract).

### Relation to proposed gap
- **Has** measured risk calibration (conformal FDR) and **evaluated decisions**.
- **Does not** use map-only features to predict UAV localization success before observation.
- Datasets: Oxford RobotCar, NCLT, St Lucia — **ground vehicles**, not UAV-to-satellite.
- Information setting differs: **post-observation verification**, not pre-flight map matchability scheduling.

---

## 6. SASGeo (not in proposal Sources)

**Identity match:** Natalia Trukhina, Vadim Vashkelis; arXiv:2607.07737.

### Relevance assessment
- **Topic-adjacent:** GNSS-denied UAV absolute fixes, integrity-aware reject, distinctiveness/stability language.
- **Not core matchability prior** for this study’s comparison table of RSPNet/MLSNet-style predictors:
  - Synthetic semantic PoC only; **explicitly not real-flight validation** (Abstract, §I, §IX).
  - Risk–coverage uses **uncalibrated** candidate-score margin (Fig. 5 caption; §V).
  - Authors list geographic holdouts and real UAV sequences as **future** work (§VIII).
- **Placement:** adjacent literature / integrity framing; cite when discussing reject decisions and UAV motivation, not as a completed four-criterion competitor.
- Code URL stated: https://gitlab.com/emilab-group/sasgeo (HTTP 200 this session; contents not cloned/audited).

---

## 7. Focused search — localizability prediction (VPR / LiDAR)

**Date:** 2026-09-22  
**Log file:** `literature/search_log_localizability_2026-09-22.md`

### Strong candidates read (extracts saved)
| Candidate | Why considered | Inclusion vs proposed gap |
|---|---|---|
| Laskar et al. arXiv:2404.00546 (VPR matching uncertainty / SUE) | Direct VPR uncertainty estimation | Uncertainty of retrieval match ≠ map-only UAV success probability + geographic calibration + scheduling |
| arXiv:2407.08162 (VPR integrity → robot navigation) | Real navigation decisions from predicted in-/out-of-tolerance matches | Strongest **adjacent** overlap on decisions; ground robot; post-VPR features; not UAV map holdouts or ECE of map probabilities |
| Nubert et al. IROS 2022 (LiDAR localizability) | Predict localizability before registration | LiDAR modality; not optical UAV–satellite map success |
| arXiv:2509.18954 / 2501.02558 (ICP covariance) | Predict registration uncertainty | LiDAR map matching covariances; different task |
| Gronat et al. CVPR 2013 (per-location VPR calibration) | Calibrates classifier scores across places | Calibration of place classifiers ≠ map-feature success probs for UAV intermittent correction |

**None of the search hits, as reviewed, combine all four gap criteria** (see `reports/gap_decision.md`).

---

## Cross-cutting distinctions enforced
1. **Real images ≠ real UAV localization outcomes.** Jiang/RSPNet use satellite/optical references with **simulated** queries for quantitative labels/evaluation.
2. **Suggested planning ≠ evaluated policy.** RSPNet/Jiang conclusions mention navigation/planning without scheduling experiments.
3. **Geographic coverage boxes ≠ proven independent regions.** RSPNet’s region split is author-stated; Jiang has category splits.
4. **Confidence / AUC / FDR ≠ interchangeable.** SafeVPR’s conformal FDR is calibration-like for accept/reject risk, but not ECE of map-only success probabilities on UAV holdouts.
5. **MLSNet:** unread full text → no “did not evaluate X” claims.

Machine-readable summary: `literature/comparison.csv`.
