#!/usr/bin/env python3
"""Stages 6-8: map-only predictors, source-only calibration, held-out-group evaluation.

Follows configs/predictor_protocol_v1.yaml exactly (no tuning). Outer loop: leave one region
group out (splits_v1.json primary_7group). Inner loop: the fold's inner_folds give an
out-of-fold (OOF) prediction for every source row; calibrators are chosen and fitted on those
OOF predictions only, then applied to the untouched test group.

  py -3 scripts/predictor/evaluate_predictors.py --features results/stage6_7/pilot --out results/stage8/pilot --name stage6_8_pilot
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
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "predictor_protocol_v1.yaml"
TEX = ("entropy", "edge_density", "grad_mean", "lap_var", "valid_frac")
TEX_COLS = [f"tex_{w}_{k}" for w in ("near", "win") for k in TEX] + ["log_radius", "n_window_tiles"]
EXTRA_COLS = ["log_radius", "n_window_tiles"]
DISTINCT_COLS = ["distinct_max", "distinct_top5", "no_competitor"]
MODELS = ("M0_constant", "M1_texture", "M2_dino", "M3_dino_distinct")
CALIBRATED = ("M1_texture", "M2_dino", "M3_dino_distinct")


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
    ap.add_argument("--features", default="results/stage6_7/pilot")
    ap.add_argument("--out", default="results/stage8/pilot")
    ap.add_argument("--name", default="stage6_8_pilot")
    args = ap.parse_args()

    proto = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    proto_sha = sha256(PROTOCOL)
    feat_dir = ROOT / args.features
    feat_manifest = json.loads((feat_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if feat_manifest["protocol_sha256"] != proto_sha:
        raise SystemExit("features were built under a different predictor protocol")
    splits = json.loads((ROOT / proto["inputs"]["splits"]).read_text(encoding="utf-8"))
    scheme = splits["schemes"][proto["inputs"]["scheme"]]
    d = Data(feat_dir)
    seed = int(proto["metrics"]["bootstrap"].split("seed ")[1].split(",")[0])
    rng = np.random.default_rng(seed)

    pred_rows, fold_rows, shuffle_auc, cal_choice = [], [], {}, {}
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
        cal, chosen = {}, {}
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
            cal[m] = f(raw[m])
            cal_choice.setdefault(m, {})[test_g] = {"chosen": chosen[m], **{f"source_logloss_{k}": v for k, v in losses.items()}}

        # Shuffle control: permute labels within each training group, refit M1.
        y_sh = d.y.copy()
        for g in np.unique(d.group[src]):
            gi = src[d.group[src] == g]
            y_sh[gi] = rng.permutation(d.y[gi])
        p_sh = fit_predict("M1_texture", d, src, te, y=y_sh)
        shuffle_auc[test_g] = float(roc_auc_score(d.y[te], p_sh)) if len(np.unique(d.y[te])) == 2 else None

        for m in MODELS:
            variants = {"raw": raw[m]} | ({"calibrated": cal[m]} if m in cal else {})
            for v, p in variants.items():
                fold_rows.append({"fold": test_g, "regions": ",".join(sorted(set(d.df["region"].iloc[te]))),
                                  "localizer_dev_contaminated": bool(fold.get("localizer_dev_contaminated", False)),
                                  "model": m, "variant": v, "calibrator": chosen.get(m, "") if v == "calibrated" else "",
                                  **metrics(d.y[te], p)})
        for j, i in enumerate(te):
            base = {k: d.df[k].iloc[i] for k in ("region", "group_7", "filename", "radius_m", "offset_k")}
            for m in MODELS:
                pred_rows.append({**base, "fold": test_g, "model": m, "label": int(d.y[i]), "p_raw": float(raw[m][j]),
                                  "p_cal": float(cal[m][j]) if m in cal else "", "calibrator": chosen.get(m, ""),
                                  "protocol_sha256": proto_sha})
        print(f"fold {test_g}: test n={len(te)} prevalence={d.y[te].mean():.3f} calibrators={chosen}", flush=True)

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
        for v, col in (("raw", "p_raw"), ("calibrated", "p_cal")):
            if v == "calibrated" and m not in CALIBRATED:
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
            macro = {k: (float(pf[k].dropna().mean()) if pf[k].notna().any() else None)
                     for k in ("auroc", "pr_auc", "brier", "log_loss", "ece")}
            pf_x = pf[~pf.localizer_dev_contaminated]
            macro_x = {k: (float(pf_x[k].dropna().mean()) if pf_x[k].notna().any() else None)
                       for k in ("auroc", "pr_auc", "brier", "log_loss", "ece")}
            summary_models[f"{m}|{v}"] = {"pooled": pooled, "pooled_ci95": ci, "macro": macro,
                                         "macro_excluding_dev_fold": macro_x,
                                         "folds_with_auroc": int(pf["auroc"].notna().sum())}

    sh_vals = [v for v in shuffle_auc.values() if v is not None]
    summary = {
        "scope": "pilot function check on Stage 5 pilot labels (10 frames per region); not a geographic result",
        "protocol_sha256": proto_sha, "features_manifest": feat_manifest,
        "n_rows": int(len(d.y)), "prevalence": float(d.y.mean()), "folds": folds,
        "models": summary_models, "calibrator_choice": cal_choice,
        "shuffle_control_auroc": {"per_fold": shuffle_auc, "macro": float(np.mean(sh_vals)) if sh_vals else None},
    }
    (ROOT / "reports" / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_scripts_or_configs": bool(git("status", "--porcelain", "--", "scripts", "configs")),
        "protocol_sha256": proto_sha, "features_dir": args.features,
        "features_csv_sha256": sha256(feat_dir / "features.csv"),
    }, indent=2), encoding="utf-8")

    reliability_figs(preds, folds, ROOT / "figures" / args.name)
    write_md(summary, per_fold, args.name)
    return 0


def reliability_figs(preds: pd.DataFrame, folds: list[str], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, 1, 11)
    for m in CALIBRATED:
        pm = preds[preds.model == m]
        panels = folds + ["pooled"]
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True, sharey=True)
        for ax, f in zip(axes.ravel(), panels):
            sub = pm if f == "pooled" else pm[pm.fold == f]
            y = sub["label"].to_numpy(int)
            for col, style, lab in (("p_raw", "o--", "raw"), ("p_cal", "s-", "calibrated")):
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


def write_md(s: dict, per_fold: pd.DataFrame, name: str) -> None:
    md = [f"# {name} — Stages 6-8 predictor summary", "",
          "**Function check on the Stage 5 pilot labels (10 frames per region), not a geographic result.** "
          f"Protocol `configs/predictor_protocol_v1.yaml` (sha256 `{s['protocol_sha256'][:12]}…`), fixed hyperparameters, "
          f"region 07 excluded (deviation D2). Rows {s['n_rows']}, prevalence {s['prevalence']:.3f}.", "",
          "## Pooled over held-out groups (95% CI: region-group bootstrap) and macro over folds", "",
          "| model | variant | pooled AUROC (CI) | pooled Brier (CI) | pooled log loss | pooled ECE | macro AUROC | macro AUROC excl. G2 | macro Brier |",
          "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for key, v in s["models"].items():
        m, var = key.split("|")
        p, ci, mac, mx = v["pooled"], v["pooled_ci95"], v["macro"], v["macro_excluding_dev_fold"]
        auc_ci = f"{fmt(p['auroc'])} ({fmt(ci['auroc'][0])}–{fmt(ci['auroc'][1])})" if "auroc" in ci else fmt(p["auroc"])
        br_ci = f"{fmt(p['brier'])} ({fmt(ci['brier'][0])}–{fmt(ci['brier'][1])})"
        md.append(f"| {m} | {var} | {auc_ci} | {br_ci} | {fmt(p['log_loss'])} | {fmt(p['ece'])} | "
                  f"{fmt(mac['auroc'])} | {fmt(mx['auroc'])} | {fmt(mac['brier'])} |")
    sc = s["shuffle_control_auroc"]
    md += ["", f"Shuffle control (M1, labels permuted within training groups): macro AUROC {fmt(sc['macro'])} "
           f"(per fold: {', '.join(f'{k} {fmt(v)}' for k, v in sc['per_fold'].items())}).", "",
           "## Per held-out group", "",
           "| fold | regions | model | variant | n | prev. | AUROC | PR-AUC | Brier | log loss | ECE | calibrator |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in per_fold.iterrows():
        flag = " (dev)" if r["localizer_dev_contaminated"] else ""
        md.append(f"| {r['fold']}{flag} | {r['regions']} | {r['model']} | {r['variant']} | {r['n']} | {r['prevalence']:.2f} | "
                  f"{fmt(r['auroc'])} | {fmt(r['pr_auc'])} | {r['brier']:.3f} | {r['log_loss']:.3f} | {r['ece']:.3f} | {r['calibrator']} |")
    md += ["", "AUROC and PR-AUC are undefined (—) for a held-out group with one outcome class. "
           "(dev) marks the fold containing localizer-dev region 03.", "",
           f"Reliability diagrams: `figures/{name}/reliability_<model>.png`. Per-row predictions and per-fold metrics: "
           "`results/stage8/pilot/`."]
    (ROOT / "reports" / f"{name}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
