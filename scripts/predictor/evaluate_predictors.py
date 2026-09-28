#!/usr/bin/env python3
"""Stages 6-8: map-only predictors, source-only calibration, held-out-group evaluation.

Follows a frozen predictor protocol exactly (no tuning). Outer loop: leave one region group out
(splits_v1.json primary_7group). Inner loop: the fold's inner_folds give an out-of-fold (OOF)
prediction for every source row; calibrators are fitted on those OOF predictions only, then
applied to the untouched test group.

  v1 (registered pilot run): pooled Platt vs isotonic, chosen on source OOF.
  v2 (primary from 2026-09-28, deviation D3): fixed-effects Platt, whose slope comes from
      within-group variation only; v1 calibration still reported as cal_v1_registered.

  py -3 scripts/predictor/evaluate_predictors.py --protocol configs/predictor_protocol_v2.yaml \\
      --features results/stage6_7/pilot --out results/stage8/pilot_v2 --name stage6_8_pilot_v2
Writes <out>/{predictions,metrics_per_fold}.csv, <out>/run_manifest.json,
reports/<name>_summary.{md,json}, figures/<name>/reliability_<model>.png.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TEX = ("entropy", "edge_density", "grad_mean", "lap_var", "valid_frac")
TEX_COLS = [f"tex_{w}_{k}" for w in ("near", "win") for k in TEX] + ["log_radius", "n_window_tiles"]
EXTRA_COLS = ["log_radius", "n_window_tiles"]
DISTINCT_COLS = ["distinct_max", "distinct_top5", "no_competitor"]
MODELS = ("M0_constant", "M1_texture", "M2_dino", "M3_dino_distinct")
CALIBRATED = ("M1_texture", "M2_dino", "M3_dino_distinct")
METRIC_KEYS = ("auroc", "pr_auc", "brier", "log_loss", "ece")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


class Data:
    def __init__(self, feat_dir: Path):
        self.df = pd.read_csv(feat_dir / "features.csv", dtype={"region": str, "group_7": str})
        emb = np.load(feat_dir / "embeddings.npz")
        self.near, self.win = emb["near"].astype(np.float64), emb["window"].astype(np.float64)
        self.y = self.df["label"].to_numpy(int)
        self.group = self.df["group_7"].to_numpy(str)
        self.tex = self.df[TEX_COLS].to_numpy(float)
        self.extra = self.df[EXTRA_COLS].to_numpy(float)
        self.distinct = self.df[DISTINCT_COLS].to_numpy(float)


def fit_predict(kind: str, d: Data, tr: np.ndarray, te: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Fit `kind` on rows `tr` (labels y[tr]) and return P(success) for rows `te`."""
    y = d.y if y is None else y
    ytr = y[tr]
    if kind == "M0_constant" or len(np.unique(ytr)) < 2:
        return np.full(len(te), ytr.mean())
    if kind == "M1_texture":
        xtr, xte = d.tex[tr], d.tex[te]
    else:
        parts_tr, parts_te = [], []
        for vec in (d.near, d.win):
            sc = StandardScaler().fit(vec[tr])
            pca = PCA(n_components=16, random_state=0).fit(sc.transform(vec[tr]))
            parts_tr.append(pca.transform(sc.transform(vec[tr])))
            parts_te.append(pca.transform(sc.transform(vec[te])))
        parts_tr.append(d.extra[tr])
        parts_te.append(d.extra[te])
        if kind == "M3_dino_distinct":
            parts_tr.append(d.distinct[tr])
            parts_te.append(d.distinct[te])
        xtr, xte = np.hstack(parts_tr), np.hstack(parts_te)
    sc = StandardScaler().fit(xtr)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(xtr), ytr)
    return lr.predict_proba(sc.transform(xte))[:, 1]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(kind: str, p: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2:
        c = float(y.mean())
        return lambda q: np.full(len(q), c)
    if kind == "platt":
        m = LogisticRegression(C=1e6, max_iter=2000).fit(logit(p), y)
        return lambda q: m.predict_proba(logit(q))[:, 1]
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3).fit(p, y)
    return lambda q: m.predict(q)


def fit_fe_platt(p: np.ndarray, y: np.ndarray, g: np.ndarray):
    """Fixed-effects Platt (protocol v2): slope from within-group variation, pooled intercept.

    Returns (calibrate(q), slope, intercept). The slope is the coefficient on logit(p) in a
    logistic model with one intercept per group, fitted on groups holding both classes, and is
    clipped at 0 so the calibrator never inverts the predictor. The intercept is then fitted on
    all rows with the slope held fixed.
    """
    z = logit(p).ravel()
    both = [k for k in np.unique(g) if len(np.unique(y[g == k])) == 2]
    slope = 0.0
    if both:
        m = np.isin(g, both)
        dummies = (g[m][:, None] == np.array(both)[None, :]).astype(float)
        lr = LogisticRegression(C=10.0, fit_intercept=False, max_iter=5000).fit(np.hstack([z[m][:, None], dummies]), y[m])
        slope = max(0.0, float(lr.coef_[0, 0]))

    def nll(b: float) -> float:
        q = np.clip(expit(slope * z + b), 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    intercept = float(minimize_scalar(nll, bounds=(-20.0, 20.0), method="bounded").x)
    return (lambda q: expit(slope * logit(q).ravel() + intercept)), slope, intercept


def ll(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    two = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if two else None,
        "pr_auc": float(average_precision_score(y, p)) if two else None,
        "brier": float(brier_score_loss(y, p)), "log_loss": ll(y, p), "ece": ece(y, p),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/predictor_protocol_v2.yaml")
    ap.add_argument("--features", default="results/stage6_7/pilot")
    ap.add_argument("--out", default="results/stage8/pilot_v2")
    ap.add_argument("--name", default="stage6_8_pilot_v2")
    args = ap.parse_args()

    proto_path = ROOT / args.protocol
    proto = yaml.safe_load(proto_path.read_text(encoding="utf-8"))
    proto_sha = sha256(proto_path)
    # v2 and later share fixed-effects calibration; v3 adds the global-permutation leakage control.
    v2 = proto["schema_version"] in ("predictor_protocol_v2", "predictor_protocol_v3")
    # Calibrated variants: name -> prediction column.
    cal_cols = {"cal_v1_registered": "p_cal_v1", "cal_v2_fe": "p_cal_fe"} if v2 else {"calibrated": "p_cal"}
    v1_name = "cal_v1_registered" if v2 else "calibrated"
    n_perm = int(proto["controls"].get("shuffle_permutations", 1))
    n_leak = int(proto["controls"].get("leakage_permutations", 0))

    feat_dir = ROOT / args.features
    feat_manifest = json.loads((feat_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if feat_manifest["protocol_sha256"] != proto["inputs"].get("features_protocol_sha256", proto_sha):
        raise SystemExit("features were built under a different predictor protocol")
    splits = json.loads((ROOT / proto["inputs"]["splits"]).read_text(encoding="utf-8"))
    scheme = splits["schemes"][proto["inputs"]["scheme"]]
    d = Data(feat_dir)
    seed = int(proto["metrics"]["bootstrap"].split("seed ")[1].split(",")[0])
    rng = np.random.default_rng(seed)

    pred_rows, fold_rows, shuffle_auc, shuffle_perm, cal_choice = [], [], {}, {}, {}
    leak_perm: dict[str, list[float]] = {}
    for fold in scheme["outer_folds"]:
        test_g = fold["fold_id"]
        te = np.flatnonzero(d.group == test_g)
        src = np.flatnonzero(d.group != test_g)
        assert len(te) and not set(te) & set(src)

        oof = {m: np.full(len(d.y), np.nan) for m in MODELS}
        for inner in fold["inner_folds"]:
            assert test_g not in inner["train_groups"] and inner["val_group"] != test_g
            itr = np.flatnonzero(np.isin(d.group, inner["train_groups"]))
            iva = np.flatnonzero(d.group == inner["val_group"])
            assert not set(itr) & set(te) and not set(iva) & set(te)
            for m in MODELS:
                oof[m][iva] = fit_predict(m, d, itr, iva)
        for m in MODELS:
            assert not np.isnan(oof[m][src]).any() and np.isnan(oof[m][te]).all()

        raw = {m: fit_predict(m, d, src, te) for m in MODELS}
        cal = {v: {} for v in cal_cols}
        chosen, fe = {}, {}
        for m in CALIBRATED:
            losses = {}
            for kind in ("platt", "isotonic"):
                per_g = []
                for g in np.unique(d.group[src]):
                    fit_i = src[d.group[src] != g]
                    ev_i = src[d.group[src] == g]
                    f = fit_calibrator(kind, oof[m][fit_i], d.y[fit_i])
                    per_g.append(ll(d.y[ev_i], f(oof[m][ev_i])))
                losses[kind] = float(np.mean(per_g))
            chosen[m] = "platt" if losses["platt"] <= losses["isotonic"] else "isotonic"
            f = fit_calibrator(chosen[m], oof[m][src], d.y[src])
            cal[v1_name][m] = f(raw[m])
            entry = {"chosen": chosen[m], **{f"source_logloss_{k}": v for k, v in losses.items()}}
            if v2:
                f_fe, slope, icpt = fit_fe_platt(oof[m][src], d.y[src], d.group[src])
                cal["cal_v2_fe"][m] = f_fe(raw[m])
                fe[m] = (slope, icpt)
                entry.update({"fe_slope": slope, "fe_intercept": icpt})
                if len(np.unique(d.y[te])) == 2 and slope > 0:
                    gap = abs(roc_auc_score(d.y[te], cal["cal_v2_fe"][m]) - roc_auc_score(d.y[te], raw[m]))
                    assert gap < 1e-6, f"fe_platt changed AUROC in {test_g}/{m} by {gap}"
            cal_choice.setdefault(m, {})[test_g] = entry

        # Shuffle control: permute labels within each training group, refit M1.
        aucs = []
        for _ in range(n_perm):
            y_sh = d.y.copy()
            for g in np.unique(d.group[src]):
                gi = src[d.group[src] == g]
                y_sh[gi] = rng.permutation(d.y[gi])
            p_sh = fit_predict("M1_texture", d, src, te, y=y_sh)
            if len(np.unique(d.y[te])) == 2:
                aucs.append(float(roc_auc_score(d.y[te], p_sh)))
        shuffle_auc[test_g] = float(np.mean(aucs)) if aucs else None
        shuffle_perm[test_g] = aucs

        # Leakage control (v3): permute all source labels together, refit M1. Runs after the
        # within-group permutations so earlier protocol versions keep their random stream.
        leak = []
        for _ in range(n_leak):
            y_sh = d.y.copy()
            y_sh[src] = rng.permutation(d.y[src])
            p_sh = fit_predict("M1_texture", d, src, te, y=y_sh)
            if len(np.unique(d.y[te])) == 2:
                leak.append(float(roc_auc_score(d.y[te], p_sh)))
        leak_perm[test_g] = leak

        def cal_label(v: str, m: str) -> str:
            if v == "cal_v2_fe":
                return f"fe_platt slope={fe[m][0]:.3f}"
            return chosen.get(m, "")

        for m in MODELS:
            variants = {"raw": raw[m]} | ({v: cal[v][m] for v in cal_cols} if m in CALIBRATED else {})
            for v, p in variants.items():
                fold_rows.append({"fold": test_g, "regions": ",".join(sorted(set(d.df["region"].iloc[te]))),
                                  "localizer_dev_contaminated": bool(fold.get("localizer_dev_contaminated", False)),
                                  "model": m, "variant": v, "calibrator": cal_label(v, m) if v != "raw" else "",
                                  **metrics(d.y[te], p)})
        for j, i in enumerate(te):
            base = {k: d.df[k].iloc[i] for k in ("region", "group_7", "filename", "radius_m", "offset_k")}
            for m in MODELS:
                row = {**base, "fold": test_g, "model": m, "label": int(d.y[i]), "p_raw": float(raw[m][j])}
                if v2:
                    row.update({"p_cal_v1": float(cal[v1_name][m][j]) if m in CALIBRATED else "",
                                "calibrator_v1": chosen.get(m, ""),
                                "p_cal_fe": float(cal["cal_v2_fe"][m][j]) if m in CALIBRATED else "",
                                "fe_slope": fe[m][0] if m in fe else "", "fe_intercept": fe[m][1] if m in fe else ""})
                else:
                    row.update({"p_cal": float(cal[v1_name][m][j]) if m in CALIBRATED else "",
                                "calibrator": chosen.get(m, "")})
                row["protocol_sha256"] = proto_sha
                pred_rows.append(row)
        extra = {m: round(fe[m][0], 3) for m in fe} if v2 else ""
        print(f"fold {test_g}: test n={len(te)} prevalence={d.y[te].mean():.3f} calibrators={chosen}"
              + (f" fe_slopes={extra}" if v2 else ""), flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame(pred_rows)
    per_fold = pd.DataFrame(fold_rows)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    per_fold.to_csv(out_dir / "metrics_per_fold.csv", index=False)

    # Pooled metrics with region-group bootstrap CIs, and macro over folds where defined.
    folds = [f["fold_id"] for f in scheme["outer_folds"]]
    reps = int(proto["metrics"]["bootstrap"].split(" reps")[0].split(",")[-1].strip().split()[-1])
    summary_models = {}
    for m in MODELS:
        pm = preds[preds.model == m]
        for v, col in (("raw", "p_raw"), *cal_cols.items()):
            if v != "raw" and m not in CALIBRATED:
                continue
            y_all, p_all = pm["label"].to_numpy(int), pm[col].astype(float).to_numpy()
            pooled = metrics(y_all, p_all)
            by_fold = {f: (pm["fold"] == f).to_numpy() for f in folds}
            boots = {k: [] for k in ("auroc", "brier", "log_loss", "ece")}
            for _ in range(reps):
                pick = rng.choice(folds, size=len(folds), replace=True)
                mask_idx = np.concatenate([np.flatnonzero(by_fold[f]) for f in pick])
                mb = metrics(y_all[mask_idx], p_all[mask_idx])
                for k in boots:
                    if mb[k] is not None:
                        boots[k].append(mb[k])
            ci = {k: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] for k, b in boots.items() if b}
            pf = per_fold[(per_fold.model == m) & (per_fold.variant == v)]
            macro = {k: (float(pf[k].dropna().mean()) if pf[k].notna().any() else None) for k in METRIC_KEYS}
            pf_x = pf[~pf.localizer_dev_contaminated]
            macro_x = {k: (float(pf_x[k].dropna().mean()) if pf_x[k].notna().any() else None) for k in METRIC_KEYS}
            summary_models[f"{m}|{v}"] = {"pooled": pooled, "pooled_ci95": ci, "macro": macro,
                                         "macro_excluding_dev_fold": macro_x,
                                         "folds_with_auroc": int(pf["auroc"].notna().sum())}

    sh_vals = [v for v in shuffle_auc.values() if v is not None]
    shuffle = {"per_fold": shuffle_auc, "macro": float(np.mean(sh_vals)) if sh_vals else None}
    if n_perm > 1:
        two = [f for f, a in shuffle_perm.items() if a]
        per_perm_macro = [float(np.mean([shuffle_perm[f][k] for f in two])) for k in range(n_perm)]
        shuffle.update({"permutations": n_perm, "macro_min": min(per_perm_macro), "macro_max": max(per_perm_macro)})
    summary = {
        "scope": "pilot function check on Stage 5 pilot labels (10 frames per region); not a geographic result",
        "protocol_sha256": proto_sha, "features_manifest": feat_manifest,
        "n_rows": int(len(d.y)), "prevalence": float(d.y.mean()), "folds": folds,
        "models": summary_models, "calibrator_choice": cal_choice,
        "shuffle_control_auroc": shuffle,
    }
    if n_leak:
        two = [f for f, a in leak_perm.items() if a]
        per_perm_macro = [float(np.mean([leak_perm[f][k] for f in two])) for k in range(n_leak)]
        mean = float(np.mean(per_perm_macro))
        summary["leakage_control_auroc"] = {
            "mode": "global permutation of source labels", "permutations": n_leak,
            "per_fold_mean": {f: (float(np.mean(a)) if a else None) for f, a in leak_perm.items()},
            "macro_mean": mean, "macro_min": min(per_perm_macro), "macro_max": max(per_perm_macro),
            "pass_rule": proto["controls"]["leakage_pass"], "pass": abs(mean - 0.5) <= 0.05,
        }
    if v2:
        summary["protocol"] = args.protocol
        summary["h3_macro"] = {
            m: {k: {"raw": summary_models[f"{m}|raw"]["macro"][k], "cal_v2_fe": summary_models[f"{m}|cal_v2_fe"]["macro"][k],
                    "cal_v1_registered": summary_models[f"{m}|cal_v1_registered"]["macro"][k]}
                for k in ("brier", "log_loss", "ece")}
            for m in CALIBRATED}
    (ROOT / "reports" / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_scripts_or_configs": bool(git("status", "--porcelain", "--", "scripts", "configs")),
        "protocol_path": args.protocol, "protocol_sha256": proto_sha, "features_dir": args.features,
        "features_csv_sha256": sha256(feat_dir / "features.csv"),
    }, indent=2), encoding="utf-8")

    reliability_figs(preds, folds, ROOT / "figures" / args.name, cal_cols)
    write_md(summary, per_fold, args.name, args.out, v2)
    return 0


def reliability_figs(preds: pd.DataFrame, folds: list[str], fig_dir: Path, cal_cols: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    styles = [("p_raw", "o--", "raw")] + [(c, s, v) for (v, c), s in zip(cal_cols.items(), ("s-", "^-"))]
    for m in CALIBRATED:
        pm = preds[preds.model == m]
        panels = folds + ["pooled"]
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True, sharey=True)
        for ax, f in zip(axes.ravel(), panels):
            sub = pm if f == "pooled" else pm[pm.fold == f]
            y = sub["label"].to_numpy(int)
            for col, style, lab in styles:
                p = sub[col].astype(float).to_numpy()
                b = np.minimum((p * 10).astype(int), 9)
                xs = [p[b == i].mean() for i in range(10) if (b == i).sum() >= 3]
                ys = [y[b == i].mean() for i in range(10) if (b == i).sum() >= 3]
                ax.plot(xs, ys, style, ms=4, label=lab)
            ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
            ax.set_title(f"{f} (n={len(sub)}, prev={y.mean():.2f})", fontsize=9)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel("predicted P(success)")
        for ax in axes[:, 0]:
            ax.set_ylabel("observed rate")
        fig.suptitle(f"{m}: reliability per held-out group (bins with >= 3 rows)", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"reliability_{m}.png", dpi=110)
        plt.close(fig)


def fmt(x, nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_md(s: dict, per_fold: pd.DataFrame, name: str, out: str, v2: bool) -> None:
    md = [f"# {name} — Stages 6-8 predictor summary", "",
          "**Function check on the Stage 5 pilot labels (10 frames per region), not a geographic result.** "
          f"Protocol `{s.get('protocol', 'configs/predictor_protocol_v1.yaml')}` (sha256 `{s['protocol_sha256'][:12]}…`), "
          f"fixed hyperparameters, region 07 excluded (deviation D2). Rows {s['n_rows']}, prevalence {s['prevalence']:.3f}.", ""]
    if v2:
        md += ["Variants: `raw` (uncalibrated), `cal_v2_fe` (primary: fixed-effects Platt, slope from within-group "
               "variation), `cal_v1_registered` (the v1 pooled Platt/isotonic, kept for comparison).", ""]
    md += ["## Headline: macro over held-out groups", "",
           "| model | variant | macro AUROC | macro AUROC excl. G2 | macro PR-AUC | macro Brier | macro log loss | macro ECE |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, v in s["models"].items():
        m, var = key.split("|")
        mac, mx = v["macro"], v["macro_excluding_dev_fold"]
        md.append(f"| {m} | {var} | {fmt(mac['auroc'])} | {fmt(mx['auroc'])} | {fmt(mac['pr_auc'])} | "
                  f"{fmt(mac['brier'])} | {fmt(mac['log_loss'])} | {fmt(mac['ece'])} |")
    if v2:
        md += ["", "### H3 check: does source-fitted calibration improve held-out probability quality?", "",
               "| model | macro Brier raw → fe (v1) | macro log loss raw → fe (v1) | fe better on both |",
               "|---|---|---|---|"]
        for m, h in s["h3_macro"].items():
            b, l = h["brier"], h["log_loss"]
            better = b["cal_v2_fe"] < b["raw"] and l["cal_v2_fe"] < l["raw"]
            md.append(f"| {m} | {fmt(b['raw'])} → {fmt(b['cal_v2_fe'])} ({fmt(b['cal_v1_registered'])}) | "
                      f"{fmt(l['raw'])} → {fmt(l['cal_v2_fe'])} ({fmt(l['cal_v1_registered'])}) | {'yes' if better else 'no'} |")
    md += ["", "## Pooled probability quality over all held-out rows (95% CI: region-group bootstrap)", "",
           "| model | variant | Brier (CI) | log loss (CI) | ECE |", "|---|---|---|---|---:|"]
    for key, v in s["models"].items():
        m, var = key.split("|")
        p, ci = v["pooled"], v["pooled_ci95"]
        md.append(f"| {m} | {var} | {fmt(p['brier'])} ({fmt(ci['brier'][0])}–{fmt(ci['brier'][1])}) | "
                  f"{fmt(p['log_loss'])} ({fmt(ci['log_loss'][0])}–{fmt(ci['log_loss'][1])}) | {fmt(p['ece'])} |")
    md += ["", "## Secondary — pooled AUROC (unreliable under leave-one-group-out)", "",
           "Each fold's predictions carry the other groups' success rate, which moves opposite to the held-out "
           "group's own rate, so pooled ranking mixes in a between-group artefact (the constant model scores well "
           "below 0.5). Read the macro AUROC above instead.", "",
           "| model | variant | pooled AUROC (CI) |", "|---|---|---|"]
    for key, v in s["models"].items():
        m, var = key.split("|")
        p, ci = v["pooled"], v["pooled_ci95"]
        auc_ci = f"{fmt(p['auroc'])} ({fmt(ci['auroc'][0])}–{fmt(ci['auroc'][1])})" if "auroc" in ci else fmt(p["auroc"])
        md.append(f"| {m} | {var} | {auc_ci} |")
    sc = s["shuffle_control_auroc"]
    rng_txt = f", range over {sc['permutations']} permutations {fmt(sc['macro_min'])}–{fmt(sc['macro_max'])}" if "permutations" in sc else ""
    lk = s.get("leakage_control_auroc")
    if lk:
        md += ["", f"**Leakage control** (M1, all source labels permuted together, {lk['permutations']} permutations): "
               f"macro AUROC mean {fmt(lk['macro_mean'])}, range {fmt(lk['macro_min'])}–{fmt(lk['macro_max'])} — "
               f"**{'PASS' if lk['pass'] else 'FAIL'}** ({lk['pass_rule']})."]
    label = "Confound diagnostic" if lk else "Shuffle control"
    md += ["", f"{label} (M1, labels permuted within training groups): macro AUROC {fmt(sc['macro'])}{rng_txt} "
           f"(per fold: {', '.join(f'{k} {fmt(v)}' for k, v in sc['per_fold'].items())})"
           + (" — keeps each group's success rate, so it measures between-region confounding, not leakage." if lk else ".") , "",
           "## Per held-out group", "",
           "| fold | regions | model | variant | n | prev. | AUROC | PR-AUC | Brier | log loss | ECE | calibrator |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in per_fold.iterrows():
        flag = " (dev)" if r["localizer_dev_contaminated"] else ""
        md.append(f"| {r['fold']}{flag} | {r['regions']} | {r['model']} | {r['variant']} | {r['n']} | {r['prevalence']:.2f} | "
                  f"{fmt(r['auroc'])} | {fmt(r['pr_auc'])} | {r['brier']:.3f} | {r['log_loss']:.3f} | {r['ece']:.3f} | {r['calibrator']} |")
    md += ["", "AUROC and PR-AUC are undefined (—) for a held-out group with one outcome class. "
           "(dev) marks the fold containing localizer-dev region 03.", "",
           f"Reliability diagrams: `figures/{name}/reliability_<model>.png`. Per-row predictions and per-fold metrics: `{out}/`."]
    (ROOT / "reports" / f"{name}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
