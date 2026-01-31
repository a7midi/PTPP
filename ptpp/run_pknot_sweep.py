from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from .build_transport_no_plots import build_transport_instance


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PknotSweepConfig:
    pknot_sweep: Dict[str, Any]
    outputs: Dict[str, Any]


def _load_config(path: Path) -> PknotSweepConfig:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PknotSweepConfig(pknot_sweep=dict(cfg["pknot_sweep"]), outputs=dict(cfg["outputs"]))


def run(config_path: Path) -> Path:
    cfg = _load_config(config_path)

    out_root = _repo_root() / str(cfg.outputs["out_root"])
    if bool(cfg.outputs.get("overwrite", False)) and out_root.exists():
        import shutil

        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    L = int(cfg.pknot_sweep["L"])
    W = int(cfg.pknot_sweep["W"])
    pf = float(cfg.pknot_sweep["pf"])
    inject_channel = int(cfg.pknot_sweep["inject_channel"])

    pknot_list = [float(x) for x in cfg.pknot_sweep["pknot_list"]]
    seeds = [int(s) for s in cfg.pknot_sweep["seeds"]]

    knot_layers = (int(cfg.pknot_sweep["knot_layers"][0]), int(cfg.pknot_sweep["knot_layers"][1]))
    knot_channels = (int(cfg.pknot_sweep["knot_channels"][0]), int(cfg.pknot_sweep["knot_channels"][1]))

    rows = []
    for pknot in pknot_list:
        for seed in seeds:
            run_dir = out_root / f"pknot{pknot:.2f}_seed{seed}"
            build_transport_instance(
                out_dir=run_dir,
                L=L,
                W=W,
                pf=pf,
                knot_on=True,
                pknot=float(pknot),
                knot_layers=knot_layers,
                knot_channels=knot_channels,
                seed=int(seed),
                mode="steady",
                inject_list=[inject_channel],
            )

            df = pd.read_csv(run_dir / "sim_summary.csv")
            df = df[df["mode"] == "steady"].copy()
            # only one row expected for inject_list length 1
            denom = df["out_sum"].astype(float).replace(0, np.nan)
            df["contrast_C"] = (df["out_max"].astype(float) * float(W)) / denom
            df["n_eff"] = np.exp(df["entropy"].astype(float))
            df["pknot"] = float(pknot)
            df["seed"] = int(seed)

            keep = [
                "tag",
                "inject_channel",
                "mode",
                "pf",
                "knot_on",
                "out_sum",
                "out_max",
                "out_median",
                "focus_gain",
                "gini",
                "entropy",
                "hill_alpha_k5",
                "active_total",
                "regime",
                "pknot",
                "seed",
                "contrast_C",
                "n_eff",
            ]
            rows.append(df[keep])

    out = pd.concat(rows, ignore_index=True)
    out.sort_values(["pknot", "seed"], inplace=True)

    out_csv = out_root / "pknot_sweep_long.csv"
    out.to_csv(out_csv, index=False)

    (out_root / "config.resolved.json").write_text(
        json.dumps({"pknot_sweep": cfg.pknot_sweep, "outputs": cfg.outputs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return out_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate pknot_sweep_long.csv for knot-strength sweep")
    ap.add_argument("--config", type=str, required=True, help="YAML config (see configs/pknot_sweep.yaml)")
    args = ap.parse_args()
    out_root = run(Path(args.config))
    print(f"[ptpp] wrote pknot sweep outputs to: {out_root}")


if __name__ == "__main__":
    main()
