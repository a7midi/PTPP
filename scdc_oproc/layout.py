from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import networkx as nx

from .config import MeshDesign
from .graph import CouplerSite, _stage_pairs


@dataclass
class Component:
    name: str
    kind: str
    x_um: float
    y_um: float
    props: Dict[str, Any]


@dataclass
class RouteEdge:
    u: Tuple[int, int]
    v: Tuple[int, int]
    length_um: float
    delay_ps: float
    loss_dB: float
    kind: str
    crosses: int


def place_components(design: MeshDesign, couplers: List[CouplerSite]) -> List[Component]:
    """Place couplers on a simple stage/channel grid.

    This produces a *placement* for a photonic CAD to later route.
    For v0, this is a planar nearest-neighbor mesh, so routing can be monotone in x.
    """
    comps: List[Component] = []

    # IO ports
    for c in range(design.W):
        comps.append(Component(
            name=f"in_{c}",
            kind="input_port",
            x_um=0.0,
            y_um=c * design.channel_pitch_um,
            props={"channel": c}
        ))
        comps.append(Component(
            name=f"out_{c}",
            kind="output_port",
            x_um=design.L * design.stage_pitch_um,
            y_um=c * design.channel_pitch_um,
            props={"channel": c}
        ))

    # Couplers
    for cp in couplers:
        if not cp.present:
            continue
        x = (cp.stage + 0.5) * design.stage_pitch_um
        y = (cp.a + 0.5) * design.channel_pitch_um
        comps.append(Component(
            name=f"cpl_s{cp.stage}_a{cp.a}_b{cp.b}",
            kind="coupler_2x2",
            x_um=float(x),
            y_um=float(y),
            props={"stage": cp.stage, "a": cp.a, "b": cp.b, "eta": cp.eta, "il_dB": cp.il_dB}
        ))

    return comps


def estimate_routes(design: MeshDesign, G: nx.DiGraph) -> List[RouteEdge]:
    """Estimate routing lengths for edges using a Manhattan metric.

    We assume each stage transition consumes ~stage_pitch in x, plus a vertical jog
    proportional to channel difference. This is an *estimator* for compile-time audits.
    Real routing must use the foundry PDK and DRC rules.

    Crossings are estimated as 0 for nearest-neighbor mesh edges; for general graphs,
    a segment intersection test would be required.
    """
    # speed of light in vacuum: 299792458 m/s
    c_m_per_s = 299792458.0
    n_g = float(design.wg_group_index)
    v_m_per_s = c_m_per_s / n_g  # group velocity

    wg_loss_dB_per_um = float(design.wg_loss_dB_per_cm) / 1e4  # 1 cm = 1e4 um

    routes: List[RouteEdge] = []

    for u, v, d in G.edges(data=True):
        (su, cu) = u
        (sv, cv) = v

        dx = abs(sv - su) * design.stage_pitch_um
        dy = abs(cv - cu) * design.channel_pitch_um
        length_um = float(dx + dy)

        # delay = length / v
        length_m = length_um * 1e-6
        delay_s = length_m / v_m_per_s
        delay_ps = delay_s * 1e12

        # loss: propagation + (optional) coupler insertion loss already on edge metadata
        loss_prop = length_um * wg_loss_dB_per_um
        loss = float(loss_prop + float(d.get("il_dB", 0.0)))

        routes.append(RouteEdge(
            u=u, v=v,
            length_um=length_um,
            delay_ps=delay_ps,
            loss_dB=loss,
            kind=str(d.get("kind", "edge")),
            crosses=0
        ))
    return routes


def annotate_graph_with_routes(G: nx.DiGraph, routes: List[RouteEdge]) -> None:
    """Attach route length/delay/loss to G edges in-place."""
    lookup = {(r.u, r.v): r for r in routes}
    for u, v in G.edges():
        r = lookup[(u, v)]
        G.edges[u, v]["length_um"] = r.length_um
        G.edges[u, v]["delay_ps"] = r.delay_ps
        G.edges[u, v]["loss_dB"] = r.loss_dB
        G.edges[u, v]["crosses"] = r.crosses
