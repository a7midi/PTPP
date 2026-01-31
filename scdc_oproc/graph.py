from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, Iterable
import numpy as np
import networkx as nx

from .config import MeshDesign


@dataclass
class CouplerSite:
    stage: int
    a: int
    b: int
    present: bool
    eta: float          # cross-coupling fraction in [0,1]
    il_dB: float        # insertion loss (dB)

    def key(self) -> Tuple[int, int, int]:
        return (self.stage, self.a, self.b)


def _stage_pairs(W: int, stage: int) -> List[Tuple[int, int]]:
    """Nearest-neighbor coupling pairs at a given stage.

    Even stages couple (0,1), (2,3), ...
    Odd stages couple (1,2), (3,4), ...
    This is a standard planar mesh pattern.
    """
    pairs: List[Tuple[int, int]] = []
    if stage % 2 == 0:
        start = 0
    else:
        start = 1
    i = start
    while i + 1 < W:
        pairs.append((i, i + 1))
        i += 2
    return pairs


def generate_couplers(design: MeshDesign) -> List[CouplerSite]:
    rng = np.random.default_rng(design.seed)
    couplers: List[CouplerSite] = []
    for s in range(design.L):
        for (a, b) in _stage_pairs(design.W, s):
            # determine whether this site is within the knot region
            in_knot = (
                design.knot_on and
                (design.knot_layers[0] <= s < design.knot_layers[1]) and
                (design.knot_channels[0] <= a < design.knot_channels[1]) and
                (design.knot_channels[0] < b <= design.knot_channels[1])
            )
            p = design.pknot if in_knot else design.pf
            present = bool(rng.random() < p)

            # small per-coupler variation in splitting
            eta = float(np.clip(design.split_eta + rng.normal(0.0, design.split_sigma), 0.0, 1.0))
            couplers.append(CouplerSite(stage=s, a=a, b=b, present=present, eta=eta, il_dB=float(design.il_dB)))
    return couplers


def build_layered_graph(design: MeshDesign, couplers: List[CouplerSite]) -> nx.DiGraph:
    """Build a directed layered graph for intensity transport.

    Nodes are (stage, channel) pairs, stage in [0..L] (L+1 layers of nodes).
    Edges go from stage s to s+1.

    - If no coupler on (a,b) at stage s: straight edges a->a and b->b.
    - If coupler present: a splits into (a,b), b splits into (a,b).
      Edge weights are intensity fractions; insertion loss is recorded on edges.
    """
    G = nx.DiGraph()

    # add nodes
    for s in range(design.L + 1):
        for c in range(design.W):
            G.add_node((s, c), stage=s, channel=c)

    # map couplers by stage and pair
    by_stage: Dict[int, Dict[Tuple[int, int], CouplerSite]] = {}
    for cp in couplers:
        by_stage.setdefault(cp.stage, {})[(cp.a, cp.b)] = cp

    # helper: add edge with metadata
    def add_edge(u, v, weight, kind, eta=None, il_dB=0.0):
        G.add_edge(u, v,
                   w=float(weight),
                   kind=str(kind),
                   eta=None if eta is None else float(eta),
                   il_dB=float(il_dB))

    # build edges stage by stage
    for s in range(design.L):
        pairs = _stage_pairs(design.W, s)
        paired = set()
        for (a, b) in pairs:
            paired.add(a); paired.add(b)
            cp = by_stage[s][(a, b)]
            if not cp.present:
                add_edge((s, a), (s+1, a), 1.0, kind="straight", il_dB=0.0)
                add_edge((s, b), (s+1, b), 1.0, kind="straight", il_dB=0.0)
            else:
                # intensity mixing with insertion loss
                t = 1.0 - cp.eta
                x = cp.eta
                # a input
                add_edge((s, a), (s+1, a), t, kind="coupler", eta=cp.eta, il_dB=cp.il_dB)
                add_edge((s, a), (s+1, b), x, kind="coupler", eta=cp.eta, il_dB=cp.il_dB)
                # b input
                add_edge((s, b), (s+1, a), x, kind="coupler", eta=cp.eta, il_dB=cp.il_dB)
                add_edge((s, b), (s+1, b), t, kind="coupler", eta=cp.eta, il_dB=cp.il_dB)

        # any unpaired channels propagate straight
        for c in range(design.W):
            if c not in paired:
                add_edge((s, c), (s+1, c), 1.0, kind="straight", il_dB=0.0)

    return G


def graph_to_adjacency(G: nx.DiGraph, design: MeshDesign) -> np.ndarray:
    """Return a dense adjacency / transport matrix representation.

    We return a (N x N) matrix for the full layered graph (nodes include stage).
    Entry A[i,j] is the weight from node i to node j.
    """
    nodes = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for u, v, d in G.edges(data=True):
        A[idx[u], idx[v]] = float(d.get("w", 0.0))
    return A


def count_diamonds(G: nx.DiGraph) -> int:
    """Count directed 'diamonds': x->y, x->z, y->w, z->w with y!=z."""
    # brute force but fine at v0 sizes (<= ~1000 nodes, ~2000 edges)
    diamonds = 0
    for x in G.nodes():
        succ = list(G.successors(x))
        if len(succ) < 2:
            continue
        for i in range(len(succ)):
            for j in range(i+1, len(succ)):
                y = succ[i]; z = succ[j]
                if y == z:
                    continue
                wy = set(G.successors(y))
                wz = set(G.successors(z))
                common = wy.intersection(wz)
                diamonds += len(common)
    return diamonds
