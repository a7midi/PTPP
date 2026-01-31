from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def build_transport_instance(
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
    mode: str = "steady",
    inject_list: List[int] | None = None,
    jitter_ps_list: List[float] | None = None,
) -> None:
    """
    Build + simulate a mesh instance *without plotting*.

    This is a memory-lean variant of `scdc_oproc.build_v0` for use in large joint sweeps.
    It intentionally skips matplotlib figures to reduce runtime + memory pressure.
    """
    root = Path(__file__).resolve().parents[1]
    from scdc_oproc.config import MeshDesign  # type: ignore
    from scdc_oproc.graph import generate_couplers, build_layered_graph, graph_to_adjacency  # type: ignore
    from scdc_oproc.layout import place_components, estimate_routes, annotate_graph_with_routes  # type: ignore
    from scdc_oproc.confluence import list_diamonds, diamond_mismatch_report  # type: ignore
    from scdc_oproc.sim import simulate_steady, simulate_pulsed  # type: ignore
    from scdc_oproc.metrics import summarize  # type: ignore
    from scdc_oproc.export import (  # type: ignore
        write_json,
        write_components_csv,
        write_routes_csv,
        write_netlist_json,
        write_gdsfactory_stub,
    )
    from scdc_oproc.repro import write_meta  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

    design = MeshDesign(
        L=int(L),
        W=int(W),
        pf=float(pf),
        knot_on=bool(knot_on),
        pknot=float(pknot),
        knot_layers=tuple(knot_layers),
        knot_channels=tuple(knot_channels),
        seed=int(seed),
    )

    couplers = generate_couplers(design)
    G = build_layered_graph(design, couplers)

    # adjacency + nodes (audit)
    nodes = list(G.nodes())
    A = graph_to_adjacency(G, design)
    np.save(out_dir / "adjacency.npy", A)
    write_json(out_dir / "nodes.json", [list(n) for n in nodes])

    comps = place_components(design, couplers)
    routes = estimate_routes(design, G)
    annotate_graph_with_routes(G, routes)

    diamonds = list_diamonds(G)
    conf = diamond_mismatch_report(G, diamonds)

    # injections: default mirrors build_v0 logic but can be overridden
    if inject_list is None:
        bulk = [0] if design.W <= 1 else [0, design.W - 1]
        kc0 = max(0, min(int(design.knot_channels[0]), design.W - 1))
        kc1 = max(0, min(int(design.knot_channels[1]), design.W))
        if kc1 <= kc0:
            knot_mid = int(design.W // 2)
        else:
            knot_mid = int((kc0 + kc1 - 1) / 2)
        knot = [knot_mid]
        inject_list = (
            sorted(set(bulk + knot)) if design.knot_on
            else sorted(set([0, design.W // 3, 2 * design.W // 3, design.W - 1]))
        )
        inject_list = [ch for ch in inject_list if 0 <= ch < design.W]

    rows: List[Dict[str, Any]] = []
    for ch in inject_list:
        if mode in ("steady", "both"):
            sim = simulate_steady(design, G, inject_channel=ch, power_in=1.0)
            summ = summarize(design, sim)
            summ["design"] = {"pf": design.pf, "knot_on": design.knot_on}
            summ["tag"] = f"steady_in{ch}"
            rows.append(summ)

        if mode in ("pulsed", "both"):
            # Jitter values (ps). If not provided, default mirrors scdc_oproc.build_v0.
            jits = jitter_ps_list if jitter_ps_list is not None else [0.0, 10.0, 25.0, 50.0]
            for jit in jits:
                sim = simulate_pulsed(design, G, inject_channel=ch, power_in=1.0, jitter_ps=float(jit))
                summ = summarize(design, sim)
                summ["design"] = {"pf": design.pf, "knot_on": design.knot_on}
                summ["tag"] = f"pulsed_in{ch}_jit{float(jit):.1f}"
                rows.append(summ)

    # Write artifacts
    write_json(out_dir / "design.json", design.to_dict())
    write_json(out_dir / "confluence_report.json", conf)
    write_components_csv(out_dir / "components.csv", comps)
    write_routes_csv(out_dir / "routes.csv", routes)
    write_netlist_json(out_dir / "netlist.json", design, comps, routes)
    write_gdsfactory_stub(out_dir / "gdsfactory_stub.py", out_dir / "netlist.json")

    # Flatten summary
    flat_rows = []
    for r in rows:
        flat = {
            "tag": r["tag"],
            "inject_channel": r["inject_channel"],
            "mode": r["mode"],
            "pf": r["design"]["pf"],
            "knot_on": r["design"]["knot_on"],
            "out_sum": r["outputs"]["sum"],
            "out_max": r["outputs"]["max"],
            "out_median": r["outputs"]["median"],
            "focus_gain": r["outputs"]["focus_gain"],
            "gini": r["outputs"]["gini"],
            "entropy": r["outputs"]["entropy"],
            "hill_alpha_k5": r["outputs"]["hill_alpha_k5"],
            "active_total": r["active"]["total"],
            "regime": r["regime"],
        }
        flat_rows.append(flat)

    pd.DataFrame(flat_rows).to_csv(out_dir / "sim_summary.csv", index=False)

    write_meta(
        out_dir / "meta.json",
        extra={
            "args": {
                "L": int(L),
                "W": int(W),
                "pf": float(pf),
                "knot_on": bool(knot_on),
                "pknot": float(pknot),
                "knot_layers": list(knot_layers),
                "knot_channels": list(knot_channels),
                "seed": int(seed),
                "mode": mode,
                "inject_list": list(inject_list),
            },
            "graph_nodes": int(G.number_of_nodes()),
            "graph_edges": int(G.number_of_edges()),
        },
        project_root=root,
    )
