# Height above ground (AGL) audit — label-free

`scripts/audit/agl_height.py`. AGL = CSV `height` − SRTM 1″ terrain (AWS skadi tiles, EGM96 metres), bilinear at each frame's CSV position. No localizer outcome is read. Per-frame values: `datasets/manifests/frame_agl.csv`.

| region | frames | CSV height median | terrain median (p5–p95) | **AGL median** (p5–p95) | AGL min | SRTM voids |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 817 | 405.8 | 18.9 (10.0–31.6) | **386.9** (374.1–396.1) | 365.3 | 0 |
| 02 | 1071 | 405.8 | 16.5 (10.0–35.0) | **389.2** (370.5–396.4) | 350.8 | 0 |
| 03 | 768 | 466.1 | 3.8 (0.8–8.0) | **462.4** (457.9–465.4) | 452.0 | 0 |
| 04 | 738 | 543.5 | 4.3 (1.0–7.7) | **539.2** (535.5–542.8) | 531.2 | 0 |
| 05 | 473 | 2313.0 | 1937.4 (1912.5–2070.1) | **375.3** (242.6–400.9) | 190.3 | 0 |
| 06 | 344 | 839.5 | 530.5 (500.9–567.7) | **307.9** (268.7–338.6) | -2.4 | 0 |
| 07 | 30 | 688.9 | 499.0 (495.4–501.0) | **189.9** (187.9–193.5) | 187.9 | 0 |
| 08 | 1033 | 551.5 | 2.9 (-3.0–6.7) | **548.6** (544.7–554.4) | 541.0 | 0 |
| 09 | 766 | 546.5 | 4.0 (1.0–29.4) | **542.5** (517.1–545.5) | 397.9 | 0 |
| 10 | 144 | 773.1 | 480.3 (475.5–487.6) | **292.6** (285.4–297.5) | 283.2 | 0 |
| 11 | 590 | 2573.5 | 1795.2 (1732.5–1918.7) | **777.8** (654.1–840.6) | 247.1 | 0 |

## Caveats

- The CSV height datum (ellipsoidal or mean sea level) is undocumented. Geoid–ellipsoid separation over these regions is of the order of tens of metres, so each region's AGL carries that much systematic offset.
- SRTM is a surface model (canopy and buildings included), ~5–10 m vertical error; it dates from 2000.
- AGL is not a footprint: no focal length or sensor size is published (Stage 2A/2D), so footprint = k × AGL needs one scale constant k per camera.
- Negative or near-zero AGL would mean a datum mismatch or a wrong CSV height; see the AGL min column.
