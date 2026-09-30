# MLSNet — Yang et al. CJA (access limitation note)

- **DOI:** 10.1016/j.cja.2025.103719
- **Access attempts 2026-09-22:** SciOpen PDF 502; ScienceDirect fail; OpenAlex `has_fulltext=false` / `pdf_url=null` / `any_repository_has_fulltext=false`; SciOpen API `contentList` empty
- **Local:** `MLSNet_SciOpen.html`, `MLSNet_full_text_api.json`, `MLSNet_openalex.json`
- **Reviewed:** abstract + metadata only

## Rule
Do **not** claim calibration / held-out-region / scheduling evaluations are absent. Mark as **not established from reviewed material**.

## Abstract-level task
Matchability level segmentation (MLSNet) on remote sensing reference imagery; accuracy/efficiency vs traditional and DL methods.

## Code collision
`minghaidea/MLSNet` is an unrelated genomics model.

## Retry 2026-09-28
- OpenAlex now reports `is_oa: true` (hybrid, licence CC BY-NC-ND 4.0) but still `pdf_url: null` and `has_fulltext: false`.
- ScienceDirect returns HTTP 403 to automated requests (bot blocking), so the PDF cannot be fetched from the pipeline.
- **Action (manual):** open https://doi.org/10.1016/j.cja.2025.103719 in a browser, save the PDF to `sources/MLSNet_Yang2026.pdf`, then check its split protocol, calibration and labels against the four gap criteria in `reports/gap_decision.md`.
