#!/usr/bin/env python3
"""Generate the main-text figures for the PRL Letter.

This script is intentionally lightweight and reads only the processed CSV artifacts
produced by the sweep scripts.

Inputs (CSV):
  --w20_csv     results/joint_sweep_full/joint_sweep_long.csv
  --size2_csv   results/joint_sweep_size2/joint_sweep_long.csv
  --pknot_csv   results/pknot_sweep/pknot_sweep_long.csv
  --jitter_csv  results/jitter_0to50/jitter_long.csv
  --dict_csv    results/joint_sweep_full/dictionary_global.csv

Outputs (PDF) written into --out_dir:
  fig_w20_dictionary_scatter.pdf
  fig_size2_dictionary_scatter_global.pdf
  fig_pknot_vs_neff.pdf
  fig_jitter_invariance.pdf
  fig_dictionary_binned.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _prep_w20_selection(df: pd.DataFrame) -> pd.DataFrame:
    # Paper selection (Fig. 1 left): knot-off injected at edge (inj=0) vs knot-on injected in interior (inj=9)
    sel = (
        (df["mode"] == "steady")
        & (
            ((df["knot_on"] == 0) & (df["inject_channel"] == 0))
            | ((df["knot_on"] == 1) & (df["inject_channel"] == 9))
        )
    )
    out = df.loc[sel].copy()
    out["delta_paper"] = -out["g_delta_a_star"]
    out["neffW"] = out["n_eff"] / 20.0
    return out


def _prep_size2_selection(df: pd.DataFrame) -> pd.DataFrame:
    # Fixed injection control (Fig. 1 right): injection fixed to mid-channel (inj=14) for both knot states
    sel = (df["mode"] == "steady") & (df["inject_channel"] == 14)
    out = df.loc[sel].copy()
    out["delta_paper"] = -out["g_delta_a_star"]
    out["neffW"] = out["n_eff"] / 28.0
    return out


def make_fig_dictionary_scatter(w20: pd.DataFrame, size2: pd.DataFrame, out_dir: Path) -> None:
    # --- W=20 selection ---
    plt.figure()
    for knot in [0, 1]:
        sub = w20[w20["knot_on"] == knot]
        plt.scatter(sub["delta_paper"], sub["neffW"], s=80, alpha=0.7, label=f"knot_on={knot}")
    plt.axvline(-0.2, linestyle="--")
    plt.xlabel(r"$\Delta a^*_{paper}$")
    plt.ylabel(r"$n_{eff}/W$")
    plt.legend()
    savefig(out_dir / "fig_w20_dictionary_scatter.pdf")

    # --- size2 fixed injection ---
    plt.figure()
    sub0 = size2[size2["knot_on"] == 0]
    sub1 = size2[size2["knot_on"] == 1]
    plt.scatter(sub0["delta_paper"], sub0["neffW"], s=80, alpha=0.7, label="knot_on=0", marker="o")
    plt.scatter(sub1["delta_paper"], sub1["neffW"], s=80, alpha=0.7, label="knot_on=1", marker="^")
    plt.axvline(-0.2, linestyle="--")
    plt.xlabel(r"$\Delta a^*_{paper}$")
    plt.ylabel(r"$n_{eff}/W$")
    plt.legend()
    savefig(out_dir / "fig_size2_dictionary_scatter_global.pdf")


def make_fig_pknot(pknot_df: pd.DataFrame, out_dir: Path) -> None:
    # Expect knot_on==True and inject_channel fixed (interior)
    df = pknot_df.copy()
    # participation fraction
    df["neffW"] = df["n_eff"] / 20.0

    g = df.groupby("pknot")["neffW"]
    pk = np.array(sorted(g.groups.keys()), dtype=float)
    med = np.array([float(g.get_group(p).median()) for p in pk])
    q25 = np.array([float(g.get_group(p).quantile(0.25)) for p in pk])
    q75 = np.array([float(g.get_group(p).quantile(0.75)) for p in pk])

    plt.figure()
    plt.fill_between(pk, q25, q75, alpha=0.2)
    plt.plot(pk, med, marker="o")
    plt.xlabel(r"$p_{knot}$")
    plt.ylabel(r"Median $n_{eff}/W$")
    savefig(out_dir / "fig_pknot_vs_neff.pdf")


def make_fig_jitter(jitter_df: pd.DataFrame, out_dir: Path) -> None:
    df = jitter_df.copy()
    # Select the two reference conditions used in the paper figure
    df = df[(df["mode"] == "pulsed") & (((df["knot_on"] == False) & (df["inject_channel"] == 0)) | ((df["knot_on"] == True) & (df["inject_channel"] == 9)))]

    plt.figure()
    for (knot_on, inj), g in df.groupby(["knot_on", "inject_channel"]):
        g = g.sort_values("jitter_ps")
        y = g["out_sum"].astype(float).to_numpy()
        y_norm = y / float(np.mean(y)) if np.isfinite(y).all() and np.mean(y) != 0 else y
        plt.plot(g["jitter_ps"], y_norm, marker="o", label=f"knot_on={int(knot_on)}, inj={inj}")

    plt.xlabel("Timing jitter (ps)")
    plt.ylabel(r"Normalized integrated energy $\Sigma y$")
    plt.ylim(0.999, 1.001)
    plt.legend()
    savefig(out_dir / "fig_jitter_invariance.pdf")


def make_fig_dictionary_binned(dict_df: pd.DataFrame, out_dir: Path) -> None:
    df = dict_df.copy()
    df["delta_paper_binned"] = -df["delta_a_star_bin"].astype(float)
    df = df.sort_values("delta_paper_binned")

    plt.figure()
    plt.plot(df["delta_paper_binned"], df["neff_med"], marker="o")
    plt.xlabel(r"$\Delta a^*_{paper}$ (binned)")
    plt.ylabel(r"Median $n_{eff}$")
    savefig(out_dir / "fig_dictionary_binned.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w20_csv", required=True)
    ap.add_argument("--size2_csv", required=True)
    ap.add_argument("--pknot_csv", required=True)
    ap.add_argument("--jitter_csv", required=True)
    ap.add_argument("--dict_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w20 = pd.read_csv(args.w20_csv)
    size2 = pd.read_csv(args.size2_csv)
    pknot = pd.read_csv(args.pknot_csv)
    jitter = pd.read_csv(args.jitter_csv)
    dict_df = pd.read_csv(args.dict_csv)

    w20s = _prep_w20_selection(w20)
    size2s = _prep_size2_selection(size2)

    make_fig_dictionary_scatter(w20s, size2s, out_dir)
    make_fig_pknot(pknot, out_dir)
    make_fig_jitter(jitter, out_dir)
    make_fig_dictionary_binned(dict_df, out_dir)


if __name__ == "__main__":
    main()
