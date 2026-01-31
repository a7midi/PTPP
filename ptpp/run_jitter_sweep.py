from __future__ import annotations

import argparse
import json
import re
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
class JitterSweepConfig:
    jitter: Dict[str, Any]
    outputs: Dict[str, Any]


def _load_config(path: Path) -> JitterSweepConfig:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return JitterSweepConfig(jitter=dict(cfg["jitter"]), outputs=dict(cfg["outputs"]))


_JIT_RE = re.compile(r"_jit([0-9]+(?:\.[0-9]+)?)$")


def _parse_jitter_ps(tag: str) -> float:
    """Parse jitter from tag like: pulsed_in9_jit25.0 -> 25.0"""
    m = _JIT_RE.search(str(tag))
    if not m:
        return float("nan")
    return float(m.group(1))


def run(config_path: Path) -> Path:
    cfg = _load_config(config_path)

    out_root = _repo_root() / str(cfg.outputs["out_root"])
    if bool(cfg.outputs.get("overwrite", False)) and out_root.exists():
        import shutil

        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    L = int(cfg.jitter["L"])
    W = int(cfg.jitter["W"])
    pf = float(cfg.jitter["pf"])
    pknot = float(cfg.jitter.get("pknot", 0.90))
    knot_layers = (int(cfg.jitter["knot_layers"][0]), int(cfg.jitter["knot_layers"][1]))
    knot_channels = (int(cfg.jitter["knot_channels"][0]), int(cfg.jitter["knot_channels"][1]))
    jitter_ps_list = [float(x) for x in cfg.jitter["jitter_ps_list"]]

    all_rows = []

    for case in cfg.jitter["cases"]:
        knot_on = bool(case["knot_on"])
        seed = int(case["seed"])
        inject_channel = int(case["inject_channel"])

        run_dir = out_root / f"pf{pf:.3f}_knot{1 if knot_on else 0}_seed{seed}"

        build_transport_instance(
            out_dir=run_dir,
            L=L,
            W=W,
            pf=pf,
            knot_on=knot_on,
            pknot=pknot,
            knot_layers=knot_layers,
            knot_channels=knot_channels,
            seed=seed,
            mode="pulsed",
            inject_list=[inject_channel],
            jitter_ps_list=jitter_ps_list,
        )

        df = pd.read_csv(run_dir / "sim_summary.csv")
        df = df[df["mode"] == "pulsed"].copy()
        df["jitter_ps"] = df["tag"].map(_parse_jitter_ps)

        denom = df["out_sum"].astype(float).replace(0, np.nan)
        df["contrast_C"] = (df["out_max"].astype(float) * float(W)) / denom
        df["n_eff"] = np.exp(df["entropy"].astype(float))

        # record run_dir as repo-relative path for portability
        df["run_dir"] = (Path(cfg.outputs["out_root"]) / run_dir.name).as_posix()

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
            "jitter_ps",
            "contrast_C",
            "n_eff",
            "run_dir",
        ]
        all_rows.append(df[keep])

    out = pd.concat(all_rows, ignore_index=True)
    out.sort_values(["knot_on", "inject_channel", "jitter_ps"], inplace=True)

    out_csv = out_root / "jitter_long.csv"
    out.to_csv(out_csv, index=False)

    # provenance
    (out_root / "config.resolved.json").write_text(
        json.dumps({"jitter": cfg.jitter, "outputs": cfg.outputs}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return out_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate jitter_long.csv for 0..50ps timing jitter sweep")
    ap.add_argument("--config", type=str, required=True, help="YAML config (see configs/jitter_0to50.yaml)")
    args = ap.parse_args()
    out_root = run(Path(args.config))
    print(f"[ptpp] wrote jitter sweep outputs to: {out_root}")


if __name__ == "__main__":
    main()
