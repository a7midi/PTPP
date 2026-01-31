from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def make_dictionary(
    joint_sweep_path: Path,
    *,
    use: Literal["global", "local"] = "global",
    a_round: int = 3,
    da_round: int = 3,
) -> pd.DataFrame:
    """
    Build a lookup table mapping (a*, Δa*) -> transport statistics.

    This produces a *binned* dictionary by rounding (a*, Δa*) to control cardinality.
    For a journal submission we typically report medians and IQRs across seeds.
    """
    df = pd.read_csv(joint_sweep_path)

    if use == "global":
        a = df["g_a_star"]
        da = df["g_delta_a_star"]
    elif use == "local":
        a = df["l_a_star"]
        da = df["l_delta_a_star"]
    else:
        raise ValueError("use must be 'global' or 'local'")

    df = df.copy()
    df["a_star_bin"] = a.round(a_round)
    df["delta_a_star_bin"] = da.round(da_round)

    group_cols = ["a_star_bin", "delta_a_star_bin"]
    agg = df.groupby(group_cols).agg(
        n=("contrast_C", "size"),
        contrast_med=("contrast_C", "median"),
        contrast_q25=("contrast_C", lambda x: float(np.quantile(x, 0.25))),
        contrast_q75=("contrast_C", lambda x: float(np.quantile(x, 0.75))),
        neff_med=("n_eff", "median"),
        neff_q25=("n_eff", lambda x: float(np.quantile(x, 0.25))),
        neff_q75=("n_eff", lambda x: float(np.quantile(x, 0.75))),
        sumy_med=("out_sum", "median"),
        active_med=("active_total", "median"),
    ).reset_index()

    # Helpful derived "regime" indicator
    agg["neff_iqr"] = agg["neff_q75"] - agg["neff_q25"]
    agg["contrast_iqr"] = agg["contrast_q75"] - agg["contrast_q25"]

    return agg.sort_values(["a_star_bin", "delta_a_star_bin"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build (a*, Δa*) -> transport dictionary from joint sweep output.")
    ap.add_argument("--joint_sweep_csv", type=str, required=True, help="Path to joint_sweep_long.csv")
    ap.add_argument("--out_csv", type=str, required=True, help="Output lookup table CSV")
    ap.add_argument("--use", choices=["global", "local"], default="global")
    ap.add_argument("--a_round", type=int, default=3)
    ap.add_argument("--da_round", type=int, default=3)
    args = ap.parse_args()

    out = make_dictionary(
        Path(args.joint_sweep_csv),
        use=args.use,
        a_round=int(args.a_round),
        da_round=int(args.da_round),
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[ptpp] wrote dictionary to: {args.out_csv}")


if __name__ == "__main__":
    main()
