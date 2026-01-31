#!/usr/bin/env python3
"""Generate additional PRL referee-facing figures.

This script is designed to be run from the PTPP repository root.
It reads processed CSV outputs (already produced by the sweep scripts)
and writes compact PDF figures into a chosen output directory.

Required CSV inputs:
  --w20_csv    joint_sweep_long.csv (W=20,L=50 sweep)
  --size2_csv  joint_sweep_size2_long.csv (W=28,L=70 fixed-injection replicate)
  --jitter_csv jitter_long.csv

Figures produced:
  figS1_roc.pdf                  ROC curves (Delta a* -> class)
  figS2_delta_distributions.pdf  Delta a* distributions (knot off vs on)
  figS3_auc_by_pf.pdf            AUC vs pf (robustness across density)
  figS4_plateau_bias.pdf         Plateau survival + excluded medians
  figS5_jitter_distributional.pdf neff/W and C vs jitter
  figS6_threshold_sweep.pdf      Balanced accuracy vs Delta-threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC via rank statistic (handles ties)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    sum_pos = float(ranks[y_true == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve_manual(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)  # descending
    y = y_true[order]
    s = y_score[order]
    # thresholds at unique score values
    thresh = np.r_[np.inf, np.unique(s)]
    tpr = []
    fpr = []
    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    for th in thresh:
        pred = (y_score >= th).astype(int)
        TP = ((pred == 1) & (y_true == 1)).sum()
        FP = ((pred == 1) & (y_true == 0)).sum()
        tpr.append(TP / P if P else 0.0)
        fpr.append(FP / N if N else 0.0)
    return np.array(fpr), np.array(tpr)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = y_true == 1
    neg = ~pos
    tpr = (y_pred[pos] == 1).mean() if pos.any() else float("nan")
    tnr = (y_pred[neg] == 0).mean() if neg.any() else float("nan")
    return float((tpr + tnr) / 2.0)


def prep_selection_w20(df: pd.DataFrame) -> pd.DataFrame:
    """Match the W=20 selection used in the Letter: knot-off inj=0 vs knot-on inj=9."""
    sel = (
        (df["mode"] == "steady")
        & (
            ((df["knot_on"] == 0) & (df["inject_channel"] == 0))
            | ((df["knot_on"] == 1) & (df["inject_channel"] == 9))
        )
    )
    out = df.loc[sel].copy()
    out["W"] = 20
    out["delta"] = -out["g_delta_a_star"]  # paper convention
    out["neffW"] = out["n_eff"] / 20.0
    return out


def prep_selection_size2(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed-injection control: injection=14 for both knot states (W=28)."""
    sel = (df["mode"] == "steady") & (df["inject_channel"] == 14)
    out = df.loc[sel].copy()
    out["W"] = 28
    out["delta"] = -out["g_delta_a_star"]
    out["neffW"] = out["n_eff"] / 28.0
    return out


def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w20_csv", required=True)
    ap.add_argument("--size2_csv", required=True)
    ap.add_argument("--jitter_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w20 = pd.read_csv(args.w20_csv)
    size2 = pd.read_csv(args.size2_csv)
    jitter = pd.read_csv(args.jitter_csv)

    w20s = prep_selection_w20(w20)
    s2s = prep_selection_size2(size2)

    # --- Fig S1: ROC curves (plateau-valid subset) ---
    plt.figure(figsize=(5.6, 3.2))
    for label, df in [("W=20 selection", w20s), ("W=28 fixed injection", s2s)]:
        d = df[df["g_plateau_ok"] == True].dropna(subset=["delta"])
        y = d["knot_on"].astype(int).to_numpy()
        score = -d["delta"].to_numpy()  # higher score => more defect-like
        auc = auc_rank(y, score)
        fpr, tpr = roc_curve_manual(y, score)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=8, loc="lower right")
    savefig(out_dir / "figS1_roc.pdf")

    # --- Fig S2: Delta distributions under fixed injection (size2) ---
    plt.figure(figsize=(5.6, 3.0))
    d = s2s[s2s["g_plateau_ok"] == True].dropna(subset=["delta"])
    d0 = d[d["knot_on"] == 0]["delta"].to_numpy()
    d1 = d[d["knot_on"] == 1]["delta"].to_numpy()
    bins = np.linspace(min(d0.min(), d1.min()), max(d0.max(), d1.max()), 18)
    plt.hist(d0, bins=bins, alpha=0.7, label="knot off")
    plt.hist(d1, bins=bins, alpha=0.7, label="knot on")
    plt.axvline(-0.25, linestyle="--", linewidth=1)
    plt.xlabel(r"$\Delta a^*$ (paper convention)")
    plt.ylabel("count")
    plt.legend(fontsize=8)
    savefig(out_dir / "figS2_delta_distributions.pdf")

    # --- Fig S3: AUC vs pf (plateau-valid) ---
    def auc_vs_pf(df: pd.DataFrame, label: str):
        rows = []
        for pf, g in df[df["g_plateau_ok"] == True].groupby("pf"):
            if g["knot_on"].nunique() < 2:
                continue
            y = g["knot_on"].astype(int).to_numpy()
            score = -g["delta"].to_numpy()
            rows.append((pf, auc_rank(y, score), len(g)))
        rows.sort(key=lambda x: x[0])
        return rows

    plt.figure(figsize=(5.6, 3.2))
    for label, df in [("W=20 selection", w20s), ("W=28 fixed injection", s2s)]:
        rows = auc_vs_pf(df, label)
        if not rows:
            continue
        pfs = [r[0] for r in rows]
        aucs = [r[1] for r in rows]
        plt.plot(pfs, aucs, marker="o", label=label)
    plt.ylim(0.45, 1.02)
    plt.xlabel(r"bulk coupling density $p_f$")
    plt.ylabel("ROC AUC (knot on vs off)")
    plt.legend(fontsize=8)
    savefig(out_dir / "figS3_auc_by_pf.pdf")

    # --- Fig S4: Plateau survival and excluded medians ---
    def plateau_summary(df: pd.DataFrame, W: int):
        out = []
        for knot in [0, 1]:
            sub = df[df["knot_on"] == knot]
            n = len(sub)
            ok = int(sub["g_plateau_ok"].sum())
            frac = ok / n if n else float("nan")
            med_excl = float((sub[sub["g_plateau_ok"] == False]["n_eff"] / W).median()) if (n - ok) else float("nan")
            out.append((knot, frac, med_excl, n))
        return out

    plt.figure(figsize=(5.6, 3.2))
    ax = plt.gca()
    ax2 = ax.twinx()
    xticks = [0, 1, 3, 4]
    xlabels = ["20 off", "20 on", "28 off", "28 on"]
    med_marks = []
    for x0, (label, df, W) in enumerate([("W=20 selection", w20s, 20), ("W=28 fixed injection", s2s, 28)]):
        summ = plateau_summary(df, W)
        # bars: plateau-valid fraction
        xs = [x0 * 3 + 0, x0 * 3 + 1]
        fracs = [summ[0][1], summ[1][1]]
        ax.bar(xs, fracs, width=0.8, alpha=0.8)
        # markers: median neff/W among plateau-fail instances (if any)
        for kidx, x in enumerate(xs):
            med_excl = summ[kidx][2]
            if np.isfinite(med_excl):
                med_marks.append((x, med_excl))
    if med_marks:
        ax2.plot([m[0] for m in med_marks], [m[1] for m in med_marks], marker="o", linestyle="None")
        ax2.set_ylabel(r"median $n_{\mathrm{eff}}/W$ (plateau-fail)")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("plateau-valid fraction")
    ax.set_ylim(0.0, 1.05)
    savefig(out_dir / "figS4_plateau_bias.pdf")

    # --- Fig S5: Jitter invariance for distributional observables ---
    jitter = jitter.copy()
    jitter["neffW"] = jitter["n_eff"] / 20.0
    plt.figure(figsize=(5.6, 3.2))
    for (knot, inj), g in jitter.groupby(["knot_on", "inject_channel"]):
        g2 = g.groupby("jitter_ps", as_index=False).agg(neffW=("neffW", "mean"), C=("contrast_C", "mean"))
        plt.plot(g2["jitter_ps"], g2["neffW"], marker="o", label=f"neff/W knot={int(knot)} inj={int(inj)}")
    plt.xlabel("timing jitter (ps)")
    plt.ylabel(r"$n_{\mathrm{eff}}/W$")
    plt.legend(fontsize=7, loc="best")
    savefig(out_dir / "figS5_jitter_neff.pdf")

    plt.figure(figsize=(5.6, 3.2))
    for (knot, inj), g in jitter.groupby(["knot_on", "inject_channel"]):
        g2 = g.groupby("jitter_ps", as_index=False).agg(C=("contrast_C", "mean"))
        plt.plot(g2["jitter_ps"], g2["C"], marker="s", label=f"C knot={int(knot)} inj={int(inj)}")
    plt.xlabel("timing jitter (ps)")
    plt.ylabel("contrast C")
    plt.legend(fontsize=7, loc="best")
    savefig(out_dir / "figS5_jitter_contrast.pdf")

    # --- Fig S6: Balanced accuracy vs Delta threshold ---
    plt.figure(figsize=(5.6, 3.2))
    for label, df in [("W=20 selection", w20s), ("W=28 fixed injection", s2s)]:
        d = df[df["g_plateau_ok"] == True].dropna(subset=["delta"])
        y = d["knot_on"].astype(int).to_numpy()
        thrs = np.linspace(d["delta"].min(), d["delta"].max(), 60)
        bas = []
        for thr in thrs:
            pred = (d["delta"].to_numpy() <= thr).astype(int)
            bas.append(balanced_accuracy(y, pred))
        plt.plot(thrs, bas, label=label)
    plt.axvline(-0.25, linestyle="--", linewidth=1)
    plt.axvline(-0.30, linestyle=":", linewidth=1)
    plt.xlabel(r"threshold on $\Delta a^*$")
    plt.ylabel("balanced accuracy")
    plt.legend(fontsize=8)
    savefig(out_dir / "figS6_threshold_sweep.pdf")


if __name__ == "__main__":
    main()
