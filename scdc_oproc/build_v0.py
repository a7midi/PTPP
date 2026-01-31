from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from .config import MeshDesign
from .graph import generate_couplers, build_layered_graph, graph_to_adjacency, count_diamonds
from .layout import place_components, estimate_routes, annotate_graph_with_routes
from .confluence import list_diamonds, diamond_mismatch_report
from .sim import simulate_steady, simulate_pulsed
from .metrics import summarize
from .export import write_json, write_components_csv, write_routes_csv, write_netlist_json, write_gdsfactory_stub
from .plots import plot_layout, plot_outputs, plot_active_profile, plot_impulse_heatmap
from .repro import write_meta


def _parse_tuple_int(s: str) -> tuple[int, int]:
    parts = s.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected 'a,b'")
    return (int(parts[0]), int(parts[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description="SCDC Optical Processor v0 build pipeline (photonic-realistic mesh)")
    ap.add_argument("--L", type=int, default=50)
    ap.add_argument("--W", type=int, default=20)
    ap.add_argument("--pf", type=float, default=0.12)
    ap.add_argument("--knot_on", action="store_true", help="Enable dense knot region")
    ap.add_argument("--pknot", type=float, default=0.90)
    ap.add_argument("--knot_layers", type=_parse_tuple_int, default="10,18")
    ap.add_argument("--knot_channels", type=_parse_tuple_int, default="6,14")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="results/oproc_v0")

    ap.add_argument("--mode", choices=["steady", "pulsed", "both"], default="both")
    ap.add_argument("--dt_ps", type=float, default=5.0)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--jitter_ps", type=float, nargs="*", default=[0.0, 10.0, 25.0, 50.0],
                    help="Jitter offsets to test (ps) in pulsed mode")

    ap.add_argument("--inject", type=int, nargs="*", default=None, help="Override injection channels list")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    design = MeshDesign(
        L=int(args.L),
        W=int(args.W),
        pf=float(args.pf),
        knot_on=bool(args.knot_on),
        pknot=float(args.pknot),
        knot_layers=tuple(args.knot_layers),
        knot_channels=tuple(args.knot_channels),
        dt_ps=float(args.dt_ps),
        max_time_bins=int(args.T),
        seed=int(args.seed),
    )

    # --- Compile graph ---
    couplers = generate_couplers(design)
    G = build_layered_graph(design, couplers)

    # adjacency (analysis/audit): dense transport adjacency for the full layered DAG
    nodes = list(G.nodes())
    A = graph_to_adjacency(G, design)
    np.save(out_dir / "adjacency.npy", A)
    write_json(out_dir / "nodes.json", [list(n) for n in nodes])


    # --- Placement & routing estimates ---
    comps = place_components(design, couplers)
    routes = estimate_routes(design, G)
    annotate_graph_with_routes(G, routes)

    # --- Confluence audit ---
    diamonds = list_diamonds(G)
    conf = diamond_mismatch_report(G, diamonds)

    # --- Choose injections ---
    if args.inject is not None and len(args.inject) > 0:
        inject_list = [int(x) for x in args.inject]
    else:
        # default: 2 bulk + 1 knot (if knot enabled) else 4 spread
        bulk = [0] if design.W <= 1 else [0, design.W - 1]

        # clamp knot channel range to [0, W]
        kc0 = max(0, min(int(design.knot_channels[0]), design.W - 1))
        kc1 = max(0, min(int(design.knot_channels[1]), design.W))
        if kc1 <= kc0:
            knot_mid = int(design.W // 2)
        else:
            knot_mid = int((kc0 + kc1 - 1) / 2)
        knot = [knot_mid]

        inject_list = (
            sorted(set(bulk + knot)) if design.knot_on
            else sorted(set([0, design.W//3, 2*design.W//3, design.W-1]))
        )
        inject_list = [ch for ch in inject_list if 0 <= ch < design.W]

    # --- Simulate ---
    rows: List[Dict[str, Any]] = []

    for ch in inject_list:
        if args.mode in ("steady", "both"):
            sim = simulate_steady(design, G, inject_channel=ch, power_in=1.0)
            summ = summarize(design, sim)
            summ["design"] = {"pf": design.pf, "knot_on": design.knot_on}
            summ["tag"] = f"steady_in{ch}"
            rows.append(summ)

            # plots
            plot_outputs(sim.outputs_energy, out_dir / f"plots/outputs_steady_in{ch}.png",
                         title=f"Outputs (steady) inject={ch}")
            plot_active_profile(summ["active"]["per_stage"], out_dir / f"plots/active_steady_in{ch}.png",
                                title=f"Active channels vs stage (steady) inject={ch}")

        if args.mode in ("pulsed", "both"):
            for jit in args.jitter_ps:
                sim = simulate_pulsed(design, G, inject_channel=ch, power_in=1.0, jitter_ps=float(jit), T=int(args.T))
                summ = summarize(design, sim)
                summ["design"] = {"pf": design.pf, "knot_on": design.knot_on}
                summ["tag"] = f"pulsed_in{ch}_jit{float(jit):.1f}"
                rows.append(summ)

                plot_outputs(sim.outputs_energy, out_dir / f"plots/outputs_pulsed_in{ch}_jit{float(jit):.1f}.png",
                             title=f"Outputs (pulsed) inject={ch} jitter={jit}ps")
                plot_active_profile(summ["active"]["per_stage"], out_dir / f"plots/active_pulsed_in{ch}_jit{float(jit):.1f}.png",
                                    title=f"Active vs stage (pulsed) inject={ch} jitter={jit}ps")
                plot_impulse_heatmap(sim, out_dir / f"plots/impulse_heatmap_in{ch}_jit{float(jit):.1f}.png",
                                     title=f"Output impulse heatmap inject={ch} jitter={jit}ps")

    # --- Write artifacts ---
    write_json(out_dir / "design.json", design.to_dict())
    write_json(out_dir / "confluence_report.json", conf)

    write_components_csv(out_dir / "components.csv", comps)
    write_routes_csv(out_dir / "routes.csv", routes)
    write_netlist_json(out_dir / "netlist.json", design, comps, routes)
    write_gdsfactory_stub(out_dir / "gdsfactory_stub.py", out_dir / "netlist.json")

    # summary CSV (flatten a bit)
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
    df = pd.DataFrame(flat_rows)
    df.to_csv(out_dir / "sim_summary.csv", index=False)

    # layout plot
    plot_layout(design, comps, out_dir / "plots/layout_couplers.png")

    # meta / reproducibility
    write_meta(out_dir / "meta.json", extra={
        "args": vars(args),
        "diamonds_count": int(conf.get("diamonds", 0)),
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
    }, project_root=Path(__file__).resolve().parents[1])

    print(f"Wrote artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
