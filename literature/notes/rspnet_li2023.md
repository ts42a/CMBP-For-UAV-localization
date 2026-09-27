# RSPNet — Li et al. 2023 (reading note)

- **Local:** `sources/RSPNet_Li2023_rs15174353_webfetch.txt`
- **Access:** 2026-09-22 (WebFetch HTML; MDPI PDF blocked)
- **Identity:** Xin Li, Guo Zhang, Hao Cui, Jinhao Ma, Wei Wang. Remote Sensing 15(17):4353. DOI 10.3390/rs15174353

## Protocol essentials
- Labels: simulated \(L_T=R_s/K\); SIFT-FLANN + AKAZE-BFM; threshold \(E\) numeric value **not stated** (§3.1)
- Classical baseline success: <3 px (§4.2)
- Split: 80/20; train/test in **two separate regions** (§4.1); coverage India/Tibet/Hubei (Fig. 7)
- Metrics: MAE/MSE/RMSE/R²; ROC at grade>5; Table 2 MAE 0.6581, R² 0.9154
- Calibration / scheduling: **not reported** in reviewed sections
- Code/data: **not reported** / WHU-SCENE-MATCHING **public path not found** on WHU data page
- Not PeihaoChen/RSPNet
