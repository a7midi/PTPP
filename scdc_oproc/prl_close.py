"""PRL closeout utilities for the SCDC Optical Processor v0 suite.

This module turns a sweep directory (produced by `python -m scdc_oproc.sweep_v0`) into
three PRL-style figures and a small `key_numbers.json` summary.

Design goals:
  - deterministic, scriptable reproduction
  - robust metrics (avoid max/median pathologies when many ports are zero)
  - minimal assumptions (reads design.json when present; otherwise uses safe defaults)

Usage:

  python -m scdc_oproc.prl_close \
    --in_root results/prl_submission_data \
    --out_dir results/prl_figures \
    --bulk_inject 0 \
    --knot_inject 9 \
    --fig1_knot_on 1 \
    --fig1_inject 9

Outputs (under --out_dir):
  - agg_sim_summary.csv
  - fig1_phase_transition.(png|pdf)
  - fig2_lensing.(png|pdf)
  - fig3_jitter.(png|pdf)  [only if pulsed data exists]
  - key_numbers.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_RUN_DIR_RE = re.compile(r"pf(?P<pf>[0-9.]+)_knot(?P<knot>[01])_seed(?P<seed>\d+)")


def _safe_float(x: object, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_design_W_L(run_dir: Path) -> Tuple[int, int]:
    """Return (W, L) from design.json if available; else (20, 50)."""
    p = run_dir / "design.json"
    if not p.exists():
        return (20, 50)
    try:
        d = json.loads(p.read_text())
        W = int(d.get("W", 20))
        L = int(d.get("L", 50))
        return (W, L)
    except Exception:
        return (20, 50)


def _parse_jitter_from_tag(tag: str) -> Optional[float]:
    # tag format: pulsed_in{ch}_jit{jit:.1f}
    if "_jit" not in tag:
        return None
    try:
        return float(tag.split("_jit", 1)[1])
    except Exception:
        return None


def collect_sweep(in_root: Path) -> pd.DataFrame:
    """Load all `sim_summary.csv` files under a sweep root into one dataframe."""
    rows: List[pd.DataFrame] = []
    for p in sorted(in_root.glob("pf*_knot*_seed*")):
        if not p.is_dir():
            continue
        csv_path = p / "sim_summary.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["run_dir"] = str(p)

        m = _RUN_DIR_RE.match(p.name)
        if m:
            df["seed"] = int(m.group("seed"))
            df["pf_dir"] = float(m.group("pf"))
            df["knot_dir"] = int(m.group("knot"))
        else:
            df["seed"] = np.nan
            df["pf_dir"] = np.nan
            df["knot_dir"] = np.nan

        W, L = _read_design_W_L(p)
        df["W"] = int(W)
        df["L"] = int(L)

        # Robust derived metrics
        eps = 1e-15
        df["out_mean"] = df["out_sum"] / df["W"].clip(lower=1)
        df["contrast_max_mean"] = df["out_max"] / (df["out_mean"] + eps)
        df["peak_fraction"] = df["out_max"] / (df["out_sum"] + eps)
        df["neff_ports"] = np.exp(df["entropy"].clip(lower=0))  # exp(H) effective #ports

        # Jitter (if present)
        df["jitter_ps"] = df["tag"].astype(str).map(_parse_jitter_from_tag)

        rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No sim_summary.csv files found under: {in_root}. "
            "Did you run scdc_oproc.sweep_v0 (or build_v0) with --out_root/--out_dir?"
        )

    return pd.concat(rows, ignore_index=True)


def _mean_sem(x: pd.Series) -> Tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return (float("nan"), float("nan"))
    mu = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(max(len(x), 1))) if len(x) > 1 else 0.0
    return (mu, sem)


def make_fig1_phase_transition(
    df: pd.DataFrame,
    out_path: Path,
    *,
    inject_channel: int = 0,
    knot_on: int = 0,
    ref_inject_channel: Optional[int] = None,
    ref_knot_on: Optional[int] = None,
) -> Dict[str, float]:
    """Phase / regime plot vs pf.

    For v0 PRL rigor, we recommend using a *robust* optical contrast metric:
        C = y_max / y_mean

    and a separate instability metric across seeds:
        CV = std(out_sum) / mean(out_sum)
    """
    sub = df[(df["mode"] == "steady") & (df["inject_channel"] == int(inject_channel)) & (df["knot_on"] == bool(knot_on))]
    if sub.empty:
        raise ValueError(
            "No steady-mode rows for the requested filter. "
            "Check your sweep used --mode steady or both."
        )

    stats: List[Tuple[float, float, float, float, float, float]] = []
    for pf, g in sub.groupby("pf"):
        mu_c, sem_c = _mean_sem(g["contrast_max_mean"])
        mu_s, sem_s = _mean_sem(g["out_sum"])
        cv = float(pd.to_numeric(g["out_sum"], errors="coerce").std(ddof=1) / (mu_s + 1e-15)) if len(g) > 1 else 0.0
        stats.append((float(pf), mu_c, sem_c, mu_s, sem_s, cv))

    stats.sort(key=lambda t: t[0])
    pf_vals = [t[0] for t in stats]
    c_mu = [t[1] for t in stats]
    c_sem = [t[2] for t in stats]
    cv_vals = [t[5] for t in stats]

    # identify empirical "critical" point as argmax contrast
    pf_crit = float(pf_vals[int(np.nanargmax(c_mu))])
    c_crit = float(np.nanmax(c_mu))

    # Optional reference curve (contrast only)
    ref_stats: Optional[List[Tuple[float, float, float]]] = None
    ref_pf_crit: Optional[float] = None
    ref_c_crit: Optional[float] = None
    if (ref_inject_channel is None) ^ (ref_knot_on is None):
        raise ValueError("Fig. 1 reference curve requires both ref_inject_channel and ref_knot_on.")
    if ref_inject_channel is not None and ref_knot_on is not None:
        ref_sub = df[
            (df["mode"] == "steady")
            & (df["inject_channel"] == int(ref_inject_channel))
            & (df["knot_on"] == bool(ref_knot_on))
        ]
        if not ref_sub.empty:
            tmp: List[Tuple[float, float, float]] = []
            for pf, g in ref_sub.groupby("pf"):
                mu_c, sem_c = _mean_sem(g["contrast_max_mean"])
                tmp.append((float(pf), mu_c, sem_c))
            tmp.sort(key=lambda t: t[0])
            ref_stats = tmp
            ref_pf_vals = [t[0] for t in tmp]
            ref_c_mu = [t[1] for t in tmp]
            ref_pf_crit = float(ref_pf_vals[int(np.nanargmax(ref_c_mu))])
            ref_c_crit = float(np.nanmax(ref_c_mu))

    fig = plt.figure(figsize=(6.0, 3.2))
    ax = fig.add_subplot(111)
    ax.errorbar(pf_vals, c_mu, yerr=c_sem, marker="o", linestyle="-", label=r"Contrast $C=y_{max}/\bar y$")

    if ref_stats is not None:
        r_pf = [t[0] for t in ref_stats]
        r_c = [t[1] for t in ref_stats]
        r_sem = [t[2] for t in ref_stats]
        ax.errorbar(
            r_pf,
            r_c,
            yerr=r_sem,
            marker="o",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"Reference (knot={int(ref_knot_on)}, inject={int(ref_inject_channel)})",
        )
    ax.set_xlabel("Bulk coupling density $p_f$")
    ax.set_ylabel("Optical contrast $C$")
    ax.set_title(f"Transport regime (knot={knot_on}, inject={int(inject_channel)})")
    ax.axvline(pf_crit, linestyle="--", linewidth=1.0)

    if ref_pf_crit is not None:
        ax.axvline(ref_pf_crit, linestyle=":", linewidth=1.0)

    ax2 = ax.twinx()
    ax2.plot(pf_vals, cv_vals, marker="s", linestyle=":", label=r"Run-to-run variability (CV of $\sum y$)")
    ax2.set_ylabel("Instability (coefficient of variation)")

    # Combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    out: Dict[str, float] = {
        "pf_crit": pf_crit,
        "contrast_at_pf_crit": c_crit,
        "inject_channel": float(inject_channel),
        "knot_on": float(knot_on),
    }
    if ref_pf_crit is not None and ref_c_crit is not None:
        out.update(
            {
                "ref_pf_crit": float(ref_pf_crit),
                "ref_contrast_at_pf_crit": float(ref_c_crit),
                "ref_inject_channel": float(ref_inject_channel),
                "ref_knot_on": float(ref_knot_on),
                "delta_pf": float(pf_crit - float(ref_pf_crit)),
            }
        )
    return out


def _select_example_run(df: pd.DataFrame, in_root: Path) -> Path:
    """Pick a representative run directory for figure 2."""
    # prefer pf≈0.12, knot_on=1, seed=1
    cand = df[(df["knot_on"] == True) & (df["mode"] == "steady")]
    if cand.empty:
        cand = df[df["mode"] == "steady"]
    # compute distance to 0.12
    cand = cand.copy()
    cand["pf_dist"] = (cand["pf"].astype(float) - 0.12).abs()
    cand = cand.sort_values(["pf_dist", "seed"], ascending=[True, True])
    run_dir = Path(str(cand.iloc[0]["run_dir"]))
    if run_dir.exists():
        return run_dir
    # fallback
    for p in sorted(in_root.glob("pf*_knot*_seed*")):
        if (p / "sim_summary.csv").exists():
            return p
    raise FileNotFoundError("Could not find an example run directory.")


def _pick_knot_inject_from_df(df: pd.DataFrame, run_dir: Path, bulk_inject: int) -> int:
    """Choose a likely knot injection channel.

    Heuristic: within that run_dir, pick the steady injection channel that is not bulk
    and has the highest contrast.
    """
    sub = df[(df["run_dir"] == str(run_dir)) & (df["mode"] == "steady")]
    if sub.empty:
        return bulk_inject
    W = int(sub["W"].iloc[0]) if "W" in sub.columns else 20
    bulk_set = {int(bulk_inject), 0, max(W - 1, 0)}
    cand = sub[~sub["inject_channel"].isin(list(bulk_set))]
    if cand.empty:
        # if only bulk injections exist, just use bulk
        return int(sub["inject_channel"].iloc[0])
    cand = cand.sort_values(["contrast_max_mean"], ascending=[False])
    return int(cand.iloc[0]["inject_channel"])


def make_fig2_lensing_panels(
    df: pd.DataFrame,
    in_root: Path,
    out_path: Path,
    *,
    example_run: Optional[Path] = None,
    bulk_inject: int = 0,
    knot_inject: Optional[int] = None,
) -> Dict[str, object]:
    """Create a 2x2 panel figure from existing PNGs: outputs + active profiles."""
    if example_run is None:
        example_run = _select_example_run(df, in_root)
    if knot_inject is None:
        knot_inject = _pick_knot_inject_from_df(df, example_run, bulk_inject=bulk_inject)

    plots_dir = example_run / "plots"
    paths = {
        "out_bulk": plots_dir / f"outputs_steady_in{bulk_inject}.png",
        "out_knot": plots_dir / f"outputs_steady_in{knot_inject}.png",
        "act_bulk": plots_dir / f"active_steady_in{bulk_inject}.png",
        "act_knot": plots_dir / f"active_steady_in{knot_inject}.png",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected plot images in example run. "
            f"Run: {example_run}. Missing: {missing}. "
            "Tip: make sure you ran build_v0/sweep_v0 with --mode steady or both."
        )

    imgs = {k: plt.imread(str(p)) for k, p in paths.items()}

    fig = plt.figure(figsize=(6.4, 4.8))
    gs = fig.add_gridspec(2, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    ax00.imshow(imgs["out_bulk"])
    ax00.set_title(f"Bulk injection (ch={bulk_inject})")
    ax00.axis("off")

    ax01.imshow(imgs["out_knot"])
    ax01.set_title(f"Knot injection (ch={knot_inject})")
    ax01.axis("off")

    ax10.imshow(imgs["act_bulk"])
    ax10.set_title("Active profile (bulk)")
    ax10.axis("off")

    ax11.imshow(imgs["act_knot"])
    ax11.set_title("Active profile (knot)")
    ax11.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    # record a couple of numbers for the paper
    sub = df[(df["run_dir"] == str(example_run)) & (df["mode"] == "steady")]
    out = {
        "example_run": str(example_run),
        "bulk_inject": int(bulk_inject),
        "knot_inject": int(knot_inject),
    }
    if not sub.empty:
        bulk_row = sub[sub["inject_channel"] == bulk_inject].head(1)
        knot_row = sub[sub["inject_channel"] == knot_inject].head(1)
        if not bulk_row.empty:
            out["bulk_contrast"] = float(bulk_row["contrast_max_mean"].iloc[0])
            out["bulk_neff_ports"] = float(bulk_row["neff_ports"].iloc[0])
        if not knot_row.empty:
            out["knot_contrast"] = float(knot_row["contrast_max_mean"].iloc[0])
            out["knot_neff_ports"] = float(knot_row["neff_ports"].iloc[0])
    return out


def make_fig3_jitter(
    df: pd.DataFrame,
    out_path: Path,
    *,
    inject_channel: Optional[int] = None,
    knot_on: Optional[bool] = None,
) -> Dict[str, float]:
    """Plot output energy and contrast vs jitter (requires pulsed rows).

    If no pulsed rows exist, this function raises ValueError.
    """
    pulsed = df[(df["mode"] == "pulsed") & df["jitter_ps"].notna()].copy()
    if pulsed.empty:
        raise ValueError(
            "No pulsed/jitter data found. Re-run at least one condition with --mode pulsed (or both)."
        )

    if knot_on is not None:
        pulsed = pulsed[pulsed["knot_on"] == bool(knot_on)]
    if inject_channel is not None:
        pulsed = pulsed[pulsed["inject_channel"] == int(inject_channel)]

    # fallback selection: pick the (knot_on=1) row with most contrast at jitter=0
    if pulsed.empty:
        pulsed = df[(df["mode"] == "pulsed") & df["jitter_ps"].notna()].copy()

    stats = []
    for jit, g in pulsed.groupby("jitter_ps"):
        mu_e, sem_e = _mean_sem(g["out_sum"])
        mu_c, sem_c = _mean_sem(g["contrast_max_mean"])
        stats.append((float(jit), mu_e, sem_e, mu_c, sem_c))
    stats.sort(key=lambda t: t[0])

    jit = [t[0] for t in stats]
    e_mu = [t[1] for t in stats]
    e_sem = [t[2] for t in stats]
    c_mu = [t[3] for t in stats]
    c_sem = [t[4] for t in stats]

    # quantify variation
    rel_var_energy = float(np.nanmax(e_mu) - np.nanmin(e_mu)) / (float(np.nanmean(e_mu)) + 1e-15)
    rel_var_contrast = float(np.nanmax(c_mu) - np.nanmin(c_mu)) / (float(np.nanmean(c_mu)) + 1e-15)

    fig = plt.figure(figsize=(6.0, 3.2))
    ax = fig.add_subplot(111)
    ax.errorbar(jit, e_mu, yerr=e_sem, marker="o", linestyle="-", label=r"Integrated output energy $\sum y$")
    ax.set_xlabel("Injection timing jitter (ps)")
    ax.set_ylabel("Integrated output energy")
    ax.set_title("Timing-jitter robustness (pulsed)")

    ax2 = ax.twinx()
    ax2.errorbar(jit, c_mu, yerr=c_sem, marker="s", linestyle=":", label=r"Contrast $C=y_{max}/\bar y$")
    ax2.set_ylabel("Contrast")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return {
        "jitter_rel_variation_energy": rel_var_energy,
        "jitter_rel_variation_contrast": rel_var_contrast,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PRL closeout figures for scdc_oproc v0")
    ap.add_argument("--in_root", type=str, required=True, help="Sweep root containing pf*_knot*_seed* folders")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for figures + aggregated CSV")

    ap.add_argument(
        "--bulk_inject",
        type=int,
        default=0,
        help=(
            "Reference injection channel for Fig 2 (left panel). "
            "For historical compatibility this was also used for Fig 1; use --fig1_inject to decouple."
        ),
    )
    ap.add_argument(
        "--fig1_inject",
        type=int,
        default=None,
        help=(
            "Injection channel for Fig 1 (phase/transport curve). "
            "If unset, defaults to --knot_inject (if provided) else --bulk_inject."
        ),
    )
    ap.add_argument("--fig1_knot_on", type=int, default=0, choices=[0, 1], help="Use knot=0 or knot=1 data for Fig 1")
    ap.add_argument(
        "--fig1_ref_inject",
        type=int,
        default=None,
        help=(
            "Optional: overlay a reference contrast curve on Fig 1 using this injection channel "
            "(requires --fig1_ref_knot_on as well). Useful to quantify a shift in the critical point."
        ),
    )
    ap.add_argument(
        "--fig1_ref_knot_on",
        type=int,
        default=None,
        choices=[0, 1],
        help="Optional: knot flag (0/1) for the Fig 1 reference curve (requires --fig1_ref_inject).",
    )

    ap.add_argument("--example_run", type=str, default=None, help="Explicit run dir for Fig 2 (e.g. pf0.120_knot1_seed1)")
    ap.add_argument(
        "--knot_inject",
        type=int,
        default=None,
        help=(
            "Device injection channel for Fig 2 (right panel). "
            "If unset, a heuristic will pick the highest-contrast interior injection."
        ),
    )

    ap.add_argument("--write_pdf", action="store_true", help="Also write PDF versions of figures")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_sweep(in_root)
    df.to_csv(out_dir / "agg_sim_summary.csv", index=False)

    key_numbers: Dict[str, object] = {
        "in_root": str(in_root),
        "n_rows": int(len(df)),
        "n_runs": int(df["run_dir"].nunique()),
    }

    # Figure 1
    fig1_png = out_dir / "fig1_phase_transition.png"
    # Figure 1 uses its own injection choice to avoid conflating wall-reference
    # behavior (often dominated by boundary guidance/percolation) with the interior
    # "device" behavior used in Fig. 2.
    fig1_inject = int(args.fig1_inject) if args.fig1_inject is not None else (
        int(args.knot_inject) if args.knot_inject is not None else int(args.bulk_inject)
    )
    key_numbers["fig1"] = make_fig1_phase_transition(
        df,
        fig1_png,
        inject_channel=fig1_inject,
        knot_on=int(args.fig1_knot_on),
        ref_inject_channel=args.fig1_ref_inject,
        ref_knot_on=args.fig1_ref_knot_on,
    )
    if args.write_pdf:
        make_fig1_phase_transition(
            df,
            out_dir / "fig1_phase_transition.pdf",
            inject_channel=fig1_inject,
            knot_on=int(args.fig1_knot_on),
            ref_inject_channel=args.fig1_ref_inject,
            ref_knot_on=args.fig1_ref_knot_on,
        )

    # Figure 2
    ex = Path(args.example_run) if args.example_run else None
    if args.knot_inject is not None and int(args.knot_inject) == int(args.bulk_inject):
        raise ValueError(
            "For Fig. 2, --knot_inject must differ from --bulk_inject. "
            "Otherwise the two panels are identical."
        )
    fig2_png = out_dir / "fig2_lensing.png"
    key_numbers["fig2"] = make_fig2_lensing_panels(
        df,
        in_root,
        fig2_png,
        example_run=ex,
        bulk_inject=int(args.bulk_inject),
        knot_inject=args.knot_inject,
    )
    if args.write_pdf:
        make_fig2_lensing_panels(
            df,
            in_root,
            out_dir / "fig2_lensing.pdf",
            example_run=ex,
            bulk_inject=int(args.bulk_inject),
            knot_inject=args.knot_inject,
        )

    # Figure 3 (optional)
    try:
        fig3_png = out_dir / "fig3_jitter.png"
        key_numbers["fig3"] = make_fig3_jitter(df, fig3_png)
        if args.write_pdf:
            make_fig3_jitter(df, out_dir / "fig3_jitter.pdf")
    except ValueError as e:
        key_numbers["fig3"] = {"note": str(e)}

    (out_dir / "key_numbers.json").write_text(json.dumps(key_numbers, indent=2))

    print("Wrote:")
    print(f"  {out_dir / 'agg_sim_summary.csv'}")
    print(f"  {fig1_png}")
    print(f"  {fig2_png}")
    if "fig3" in key_numbers and isinstance(key_numbers["fig3"], dict) and "note" not in key_numbers["fig3"]:
        print(f"  {out_dir / 'fig3_jitter.png'}")
    print(f"  {out_dir / 'key_numbers.json'}")


if __name__ == "__main__":
    main()
