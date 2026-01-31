from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _adjacency_to_digraph(adj: np.ndarray) -> "nx.DiGraph":
    import networkx as nx

    n = int(adj.shape[0])
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
        return adj[:1, :1].copy()
    return adj[np.ix_(idx, idx)]


def compute_geometry(
    *,
    adjacency_npy: Path,
    nodes_json: Path,
    r_scales: List[int],
    domain_eta: float,
    domain_min_blocks: int,
    rho_min: float,
    plateau_window_k: int,
    plateau_rel_tol: float,
    plateau_min_domain_blocks: int,
    knot_layers: Tuple[int, int],
    knot_channels: Tuple[int, int],
    local_margin_layers: int,
    local_margin_channels: int,
) -> Dict[str, Any]:
    from emergence_v3.geometry import compute_geometry_diagnostics  # type: ignore

    adj = np.load(adjacency_npy)
    nodes = json.loads(nodes_json.read_text(encoding="utf-8"))

    G_global = _adjacency_to_digraph(adj)
    diag_g = compute_geometry_diagnostics(
        G_global,
        r_scales=r_scales,
        domain_eta=domain_eta,
        domain_min_blocks=domain_min_blocks,
        rho_min=rho_min,
        plateau_window_k=plateau_window_k,
        plateau_rel_tol=plateau_rel_tol,
        plateau_min_domain_blocks=plateau_min_domain_blocks,
    )

    adj_local = _induced_subgraph_from_bbox(
        adj=adj,
        nodes=nodes,
        layers=knot_layers,
        channels=knot_channels,
        margin_layers=local_margin_layers,
        margin_channels=local_margin_channels,
    )
    G_local = _adjacency_to_digraph(adj_local)
    diag_l = compute_geometry_diagnostics(
        G_local,
        r_scales=r_scales,
        domain_eta=domain_eta,
        domain_min_blocks=max(5, domain_min_blocks // 2),
        rho_min=rho_min,
        plateau_window_k=max(2, plateau_window_k - 1),
        plateau_rel_tol=plateau_rel_tol,
        plateau_min_domain_blocks=max(5, plateau_min_domain_blocks // 2),
    )

    def _pack(d: Any, prefix: str) -> Dict[str, Any]:
        return {
            f"{prefix}a_star": float(getattr(d, "a_star", float("nan"))),
            f"{prefix}a_star_ls": float(getattr(d, "a_star_ls", float("nan"))),
            f"{prefix}delta_a_star": float(getattr(d, "delta_a_star", float("nan"))),
            f"{prefix}plateau_ok": bool(getattr(d, "plateau_ok", False)),
            f"{prefix}plateau_Rs": list(getattr(d, "plateau_Rs", [])),
            f"{prefix}plateau_rel_var": float(getattr(d, "plateau_rel_var", float("nan"))),
            f"{prefix}num_scc": int(getattr(d, "num_scc", 0)),
            f"{prefix}max_depth": int(getattr(d, "max_depth", 0)),
        }

    out: Dict[str, Any] = {}
    out.update(_pack(diag_g, prefix="g_"))
    out.update(_pack(diag_l, prefix="l_"))
    return out


def main() -> None:
    # Reduce noisy numerical warnings; all raw data are saved for auditing anyway.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    ap = argparse.ArgumentParser(description="Compute (a*, Δa*) geometry diagnostics for a mesh adjacency.")
    ap.add_argument("--adj", type=str, required=True)
    ap.add_argument("--nodes", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--r_scales", type=int, nargs="+", required=True)
    ap.add_argument("--domain_eta", type=float, default=0.05)
    ap.add_argument("--domain_min_blocks", type=int, default=20)
    ap.add_argument("--rho_min", type=float, default=0.0)
    ap.add_argument("--plateau_window_k", type=int, default=3)
    ap.add_argument("--plateau_rel_tol", type=float, default=0.15)
    ap.add_argument("--plateau_min_domain_blocks", type=int, default=20)

    ap.add_argument("--knot_layers", type=int, nargs=2, required=True)
    ap.add_argument("--knot_channels", type=int, nargs=2, required=True)
    ap.add_argument("--local_margin_layers", type=int, default=1)
    ap.add_argument("--local_margin_channels", type=int, default=1)

    args = ap.parse_args()

    geom = compute_geometry(
        adjacency_npy=Path(args.adj),
        nodes_json=Path(args.nodes),
        r_scales=[int(x) for x in args.r_scales],
        domain_eta=float(args.domain_eta),
        domain_min_blocks=int(args.domain_min_blocks),
        rho_min=float(args.rho_min),
        plateau_window_k=int(args.plateau_window_k),
        plateau_rel_tol=float(args.plateau_rel_tol),
        plateau_min_domain_blocks=int(args.plateau_min_domain_blocks),
        knot_layers=(int(args.knot_layers[0]), int(args.knot_layers[1])),
        knot_channels=(int(args.knot_channels[0]), int(args.knot_channels[1])),
        local_margin_layers=int(args.local_margin_layers),
        local_margin_channels=int(args.local_margin_channels),
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(geom, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
