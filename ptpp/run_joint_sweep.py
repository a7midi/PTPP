from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from .build_transport_no_plots import build_transport_instance

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_external_on_path() -> None:
    """No-op. Engine and geometry code are vendored in this repository."""
    return


@dataclass(frozen=True)
class JointSweepConfig:
    scdc: Dict[str, Any]
    geometry: Dict[str, Any]
    outputs: Dict[str, Any]


def _load_config(path: Path) -> JointSweepConfig:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return JointSweepConfig(
        scdc=dict(cfg["scdc"]),
        geometry=dict(cfg["geometry"]),
        outputs=dict(cfg["outputs"]),
    )


def _run_build_v0(
    *,
    out_dir: Path,
    L: int,
    W: int,
    pf: float,
    knot_on: bool,
    pknot: float,
    knot_layers: Tuple[int, int],
    knot_channels: Tuple[int, int],
    seed: int,
    mode: str,
) -> None:
    """Call the reference build pipeline from the transport suite (vendored)."""
    cmd = [
        "python",
        "-m",
        "scdc_oproc.build_v0",
        "--L",
        str(int(L)),
        "--W",
        str(int(W)),
        "--pf",
        str(float(pf)),
        "--seed",
        str(int(seed)),
        "--mode",
        str(mode),
        "--out_dir",
        str(out_dir),
    ]
    if knot_on:
        cmd += [
            "--knot_on",
            "--pknot",
            str(float(pknot)),
            "--knot_layers",
            f"{int(knot_layers[0])},{int(knot_layers[1])}",
            "--knot_channels",
            f"{int(knot_channels[0])},{int(knot_channels[1])}",
        ]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "build_v0.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, check=True, env=env, stdout=log, stderr=log)


def _adjacency_to_digraph(adj: np.ndarray) -> "nx.DiGraph":
    import networkx as nx  # local import to keep requirements minimal

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Expected square adjacency; got shape={adj.shape}")
    n = adj.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(adj > 0)
    G.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return G


def _induced_subgraph_from_bbox(
    *,
    adj: np.ndarray,
    nodes: List[List[int]],
    layers: Tuple[int, int],
    channels: Tuple[int, int],
    margin_layers: int = 1,
    margin_channels: int = 1,
) -> np.ndarray:
    """Return adjacency matrix for the induced subgraph in a (layer,channel) bounding box."""
    loL = min(layers) - margin_layers
    hiL = max(layers) + margin_layers
    loC = min(channels) - margin_channels
    hiC = max(channels) + margin_channels

    idx = [
        i
        for i, (layer, ch) in enumerate(nodes)
        if (loL <= int(layer) <= hiL) and (loC <= int(ch) <= hiC)
    ]
    if not idx:
        # Return a tiny empty graph rather than failing hard.
        return adj[:1, :1].copy()
    sub = adj[np.ix_(idx, idx)]
    return sub


def _compute_geometry(
    *,
    run_dir: Path,
    adjacency_npy: Path,
    nodes_json: Path,
    geom_cfg: Dict[str, Any],
    knot_layers: Tuple[int, int],
    knot_channels: Tuple[int, int],
) -> Dict[str, Any]:
    """Compute global+local geometry diagnostics in a subprocess to avoid memory growth."""
    out_path = run_dir / "geometry_diagnostics.json"
    cmd = [
        "python",
        "-m",
        "ptpp.compute_geometry_cli",
        "--adj",
        str(adjacency_npy),
        "--nodes",
        str(nodes_json),
        "--out",
        str(out_path),
        "--r_scales",
        *[str(int(x)) for x in geom_cfg["r_scales"]],
        "--domain_eta",
        str(float(geom_cfg.get("domain_eta", 0.05))),
        "--domain_min_blocks",
        str(int(geom_cfg.get("domain_min_blocks", 20))),
        "--rho_min",
        str(float(geom_cfg.get("rho_min", 0.0))),
        "--plateau_window_k",
        str(int(geom_cfg.get("plateau_window_k", 3))),
        "--plateau_rel_tol",
        str(float(geom_cfg.get("plateau_rel_tol", 0.15))),
        "--plateau_min_domain_blocks",
        str(int(geom_cfg.get("plateau_min_domain_blocks", 20))),
        "--knot_layers",
        str(int(knot_layers[0])),
        str(int(knot_layers[1])),
        "--knot_channels",
        str(int(knot_channels[0])),
        str(int(knot_channels[1])),
        "--local_margin_layers",
        str(int(geom_cfg.get("local_margin_layers", 1))),
        "--local_margin_channels",
        str(int(geom_cfg.get("local_margin_channels", 1))),
    ]
    env = os.environ.copy()
    # Make the current repo importable for the subprocess (python -m ptpp.compute_geometry_cli).
    root = _repo_root()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) + (os.pathsep + prev if prev else "")
    subprocess.run(cmd, check=True, env=env)

    return json.loads(out_path.read_text(encoding="utf-8"))


def _transport_rows_from_sim_summary(sim_summary_csv: Path, W: int) -> pd.DataFrame:
    df = pd.read_csv(sim_summary_csv)

    # Contrast C = y_max / mean(y) = out_max / (out_sum / W)
    denom = df["out_sum"].astype(float).replace(0, np.nan)
    df["contrast_C"] = (df["out_max"].astype(float) * float(W)) / denom

    # n_eff = exp(H) where H is Shannon entropy of normalized outputs (saved as "entropy")
    df["n_eff"] = np.exp(df["entropy"].astype(float))
    return df


def run(config_path: Path) -> Path:
    _ensure_external_on_path()
    cfg = _load_config(config_path)

    out_root = _repo_root() / str(cfg.outputs["out_root"])
    if bool(cfg.outputs.get("overwrite", False)) and out_root.exists():
        for p in sorted(out_root.glob("*")):
            if p.is_dir():
                import shutil

                shutil.rmtree(p)
            else:
                p.unlink()
    out_root.mkdir(parents=True, exist_ok=True)

    L = int(cfg.scdc["L"])
    W = int(cfg.scdc["W"])
    pknot = float(cfg.scdc.get("pknot", 0.90))
    knot_layers = (int(cfg.scdc["knot_layers"][0]), int(cfg.scdc["knot_layers"][1]))
    knot_channels = (int(cfg.scdc["knot_channels"][0]), int(cfg.scdc["knot_channels"][1]))
    mode = str(cfg.scdc.get("mode", "steady"))

    rows: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []

    for pf in cfg.scdc["pf_list"]:
        for knot_on in cfg.scdc["knot_on_list"]:
            for seed in cfg.scdc["seeds"]:
                tag = f"pf{float(pf):.3f}_knot{1 if knot_on else 0}_seed{int(seed)}"
                run_dir = out_root / tag
                run_dir.mkdir(parents=True, exist_ok=True)

                # Build + simulate transport instance (no plots) to keep joint sweeps lean.
                inject_list = cfg.scdc.get("inject_list", None)
                build_transport_instance(
                    out_dir=run_dir,
                    L=L,
                    W=W,
                    pf=float(pf),
                    knot_on=bool(knot_on),
                    pknot=pknot,
                    knot_layers=knot_layers,
                    knot_channels=knot_channels,
                    seed=int(seed),
                    mode=mode,
                    inject_list=inject_list,
                )


                # Geometry (global + local)
                geom = _compute_geometry(
                    run_dir=run_dir,
                    adjacency_npy=run_dir / "adjacency.npy",
                    nodes_json=run_dir / "nodes.json",
                    geom_cfg=cfg.geometry,
                    knot_layers=knot_layers,
                    knot_channels=knot_channels,
                )

                # Transport metrics (per injection row)
                tdf = _transport_rows_from_sim_summary(run_dir / "sim_summary.csv", W=W)
                for _, r in tdf.iterrows():
                    row = {
                        "pf": float(pf),
                        "knot_on": bool(knot_on),
                        "seed": int(seed),
                        "mode": str(r["mode"]),
                        "tag": str(r["tag"]),
                        "inject_channel": int(r["inject_channel"]),
                        "out_sum": float(r["out_sum"]),
                        "out_max": float(r["out_max"]),
                        "entropy": float(r["entropy"]),
                        "contrast_C": float(r["contrast_C"]),
                        "n_eff": float(r["n_eff"]),
                        "gini": float(r["gini"]),
                        "active_total": int(r["active_total"]),
                        "regime": str(r["regime"]),
                    }
                    row.update(geom)
                    rows.append(row)

                runs.append(
                    {
                        "run_tag": tag,
                        "run_dir": str(run_dir),
                        "pf": float(pf),
                        "knot_on": bool(knot_on),
                        "seed": int(seed),
                    }
                )

    df_all = pd.DataFrame(rows)
    df_runs = pd.DataFrame(runs)

    # Deterministic sort for diffs / hashing
    df_all.sort_values(["pf", "knot_on", "seed", "inject_channel", "tag"], inplace=True)
    df_runs.sort_values(["pf", "knot_on", "seed"], inplace=True)

    out_csv = out_root / "joint_sweep_long.csv"
    df_all.to_csv(out_csv, index=False)
    df_runs.to_csv(out_root / "runs_index.csv", index=False)

    if bool(cfg.outputs.get("write_parquet", True)):
        out_parquet = out_root / "joint_sweep_long.parquet"
        try:
            df_all.to_parquet(out_parquet, index=False)
        except ImportError as e:
            # Parquet requires optional engines (pyarrow or fastparquet).
            # For strict reproducibility CSV remains the primary artifact.
            print(f"[ptpp] parquet skipped (missing optional dependency): {e}")

    # Copy resolved config for provenance
    with (out_root / "config.resolved.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"scdc": cfg.scdc, "geometry": cfg.geometry, "outputs": cfg.outputs},
            f,
            indent=2,
            sort_keys=True,
        )

    return out_root


def main() -> None:
    ap = argparse.ArgumentParser(description="PTPP joint sweep: transport + (a*, Δa*) geometry diagnostics")
    ap.add_argument("--config", type=str, required=True, help="YAML config (see configs/joint_sweep_*.yaml)")
    args = ap.parse_args()
    out_root = run(Path(args.config))
    print(f"[ptpp] wrote joint sweep outputs to: {out_root}")


if __name__ == "__main__":
    main()
