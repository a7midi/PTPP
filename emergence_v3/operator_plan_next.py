from __future__ import annotations

"""
Next-operator plan utilities.

CRITICAL: Blocking must match the main suite:
- blocks scale ~ N/R (topological-order chunks), not ~ max_depth/R (depth-interval bins)

This file provides:
- stable condensation DAG + stable topological order
- curvature proxy κ_r on condensation edges
- memory density ρ_mem on condensation nodes
- per-R diagnostics with blocks = topo chunks of size R
- review-proof plateau selection on a_R
- Null-2 / Null-2.5 bucketed edge swaps
- allocation metrics on condensation DAG edges (global + depth-conditioned)

Dependencies: numpy, networkx; optional scipy (if installed) for fast Theil–Sen.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import heapq
import math

import numpy as np
import networkx as nx


# ----------------------------
# Stable iteration
# ----------------------------

def iter_edges_stable(G: nx.Graph) -> Iterable[Tuple[int, int]]:
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, _k in sorted(G.edges(keys=True)):
            yield int(u), int(v)
    else:
        for u, v in sorted(G.edges()):
            yield int(u), int(v)


# ----------------------------
# Condensation DAG (stable SCC ids)
# ----------------------------

@dataclass(frozen=True)
class Condensation:
    C: nx.DiGraph
    node_to_scc: Dict[int, int]
    scc_nodes: List[List[int]]
    edge_weight: Dict[Tuple[int, int], int]   # multiplicity of original edges across SCCs


def stable_condensation(G: nx.DiGraph) -> Condensation:
    """
    Deterministic SCC condensation:
    - SCC ids assigned by sorting SCCs by min node id.
    - edge_weight records multiplicity across SCCs.
    """
    base = nx.DiGraph()
    base.add_nodes_from(int(n) for n in G.nodes())
    for u, v in iter_edges_stable(G):
        base.add_edge(int(u), int(v))

    comps = list(nx.strongly_connected_components(base))
    comps_sorted = sorted((sorted(map(int, c)) for c in comps), key=lambda c: c[0])

    node_to_scc: Dict[int, int] = {}
    scc_nodes: List[List[int]] = []
    for sid, nodes in enumerate(comps_sorted):
        scc_nodes.append(nodes)
        for n in nodes:
            node_to_scc[int(n)] = int(sid)

    C = nx.DiGraph()
    C.add_nodes_from(range(len(scc_nodes)))

    edge_weight: Dict[Tuple[int, int], int] = {}
    for u, v in iter_edges_stable(G):
        su = node_to_scc[int(u)]
        sv = node_to_scc[int(v)]
        if su == sv:
            continue
        C.add_edge(su, sv)
        edge_weight[(su, sv)] = edge_weight.get((su, sv), 0) + 1

    return Condensation(C=C, node_to_scc=node_to_scc, scc_nodes=scc_nodes, edge_weight=edge_weight)


# ----------------------------
# Stable topo order and depth (used for null constraints + metrics)
# ----------------------------

def stable_topological_order(DAG: nx.DiGraph) -> List[int]:
    indeg = {int(u): int(DAG.in_degree(u)) for u in DAG.nodes()}
    heap = [u for u, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    adj = {int(u): sorted(int(v) for v in DAG.successors(u)) for u in DAG.nodes()}

    out: List[int] = []
    while heap:
        u = heapq.heappop(heap)
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)

    if len(out) != DAG.number_of_nodes():
        raise ValueError("Topological order failed: graph is not a DAG.")
    return out


def scc_depths(C: nx.DiGraph) -> Dict[int, int]:
    topo = stable_topological_order(C)
    depth = {u: 0 for u in topo}
    preds = {u: sorted(int(p) for p in C.predecessors(u)) for u in C.nodes()}
    for u in topo:
        if preds[u]:
            depth[u] = max(depth[p] + 1 for p in preds[u])
    return depth


# ----------------------------
# Geometry proxies on condensation DAG
# ----------------------------

def cone_volume_truncated(C: nx.DiGraph, r: int) -> Dict[int, int]:
    r = int(r)
    if r < 0:
        raise ValueError("r must be >= 0")
    adj = {int(u): [int(v) for v in C.successors(u)] for u in C.nodes()}

    if r == 0:
        return {u: 1 for u in adj}
    if r == 1:
        return {u: 1 + len(set(adj[u])) for u in adj}
    if r == 2:
        out: Dict[int, int] = {}
        for u in adj:
            s = set(adj[u])
            for v in adj[u]:
                s.update(adj.get(v, []))
            s.add(u)
            out[u] = int(len(s))
        return out

    out: Dict[int, int] = {}
    for s in adj:
        seen = {s}
        stack = [(s, 0)]
        while stack:
            u, d = stack.pop()
            if d == r:
                continue
            for v in adj[u]:
                if v in seen:
                    continue
                seen.add(v)
                stack.append((v, d + 1))
        out[s] = int(len(seen))
    return out


def curvature_edge_proxy(C: nx.DiGraph, Vr: Dict[int, int]) -> Dict[Tuple[int, int], float]:
    return {(int(u), int(v)): float(Vr[int(v)] - Vr[int(u)]) for u, v in iter_edges_stable(C)}


def rho_mem(C: nx.DiGraph) -> Dict[int, float]:
    # ρ_mem(X) = outdeg(X) - 1 on condensation DAG nodes
    return {int(u): float(C.out_degree(int(u)) - 1) for u in C.nodes()}


# ----------------------------
# Blocking (MATCHES SUITE): topo-order chunks of size R
# ----------------------------

def blocks_by_topo_chunks(topo: List[int], R: int) -> List[List[int]]:
    R = int(R)
    if R <= 0:
        raise ValueError("R must be >= 1")
    return [topo[i:i + R] for i in range(0, len(topo), R)]


# ----------------------------
# Fits: Theil–Sen + LS
# ----------------------------

try:
    from scipy.stats import theilslopes as _theilslopes  # type: ignore

    def theil_sen_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2:
            return float("nan"), float("nan")
        res = _theilslopes(y, x)  # slope, intercept, lo, hi
        return float(res[0]), float(res[1])

except Exception:
    def theil_sen_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.size
        if n < 2:
            return float("nan"), float("nan")
        slopes: List[float] = []
        for i in range(n - 1):
            xi = x[i]
            yi = y[i]
            for j in range(i + 1, n):
                dx = x[j] - xi
                if dx == 0:
                    continue
                slopes.append((y[j] - yi) / dx)
        if not slopes:
            return float("nan"), float("nan")
        a = float(np.median(np.asarray(slopes, dtype=float)))
        b = float(np.median(y - a * x))
        return a, b


def ls_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan"), float("nan")
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


# ----------------------------
# Per-R diagnostics
# ----------------------------

@dataclass(frozen=True)
class ScaleDiagnostics:
    R: int
    blocks_total: int
    blocks_with_edges: int
    domain_blocks: int
    domain_ok: bool
    mean_rho_pos: float
    a_R: float
    b_R: float
    a_R_LS: float
    b_R_LS: float
    g_R: float


def compute_scale_diagnostics(
    C: nx.DiGraph,
    topo: List[int],
    rho: Dict[int, float],
    kappa_edge: Dict[Tuple[int, int], float],
    *,
    R: int,
    domain_min_blocks: int,
    min_blocks_total: int,
    edge_weight: Optional[Dict[Tuple[int, int], int]] = None,
) -> ScaleDiagnostics:
    """
    Blocks = contiguous chunks of topo order, size R.

    ρ_R(block) = mean_{X in block} ρ_mem(X)
    κ_R(block) = mean_{edges X->Y with SOURCE X in block} κ(X->Y)
               = outgoing-edge average (edge-weighted by multiplicity if provided)

    Domain blocks: ρ_R > 0 and κ_R finite (requires at least one outgoing edge from the block).
    """
    blocks = blocks_by_topo_chunks(topo, int(R))
    blocks_total = int(len(blocks))
    if blocks_total < int(min_blocks_total):
        return ScaleDiagnostics(
            R=int(R), blocks_total=blocks_total, blocks_with_edges=0,
            domain_blocks=0, domain_ok=False,
            mean_rho_pos=float("nan"),
            a_R=float("nan"), b_R=float("nan"),
            a_R_LS=float("nan"), b_R_LS=float("nan"),
            g_R=float("nan"),
        )

    node_block: Dict[int, int] = {}
    for bi, block in enumerate(blocks):
        for u in block:
            node_block[int(u)] = int(bi)

    ksum = np.zeros(blocks_total, dtype=float)
    kcnt = np.zeros(blocks_total, dtype=float)

    # Edge-driven κ, by source block
    for (u, v), kval in kappa_edge.items():
        bu = node_block.get(int(u), None)
        if bu is None:
            continue
        w = 1.0
        if edge_weight is not None:
            w = float(edge_weight.get((int(u), int(v)), 1))
        ksum[bu] += w * float(kval)
        kcnt[bu] += w

    rho_block = np.zeros(blocks_total, dtype=float)
    for bi, block in enumerate(blocks):
        rho_block[bi] = float(np.mean([rho[int(u)] for u in block])) if block else float("nan")

    blocks_with_edges = int(np.sum(kcnt > 0))
    domain_mask = (rho_block > 0) & (kcnt > 0)
    domain_blocks = int(np.sum(domain_mask))
    domain_ok = domain_blocks >= int(domain_min_blocks)

    if not domain_ok:
        mean_rho_pos = float(np.mean(rho_block[domain_mask])) if domain_blocks > 0 else float("nan")
        return ScaleDiagnostics(
            R=int(R), blocks_total=blocks_total, blocks_with_edges=blocks_with_edges,
            domain_blocks=domain_blocks, domain_ok=False,
            mean_rho_pos=mean_rho_pos,
            a_R=float("nan"), b_R=float("nan"),
            a_R_LS=float("nan"), b_R_LS=float("nan"),
            g_R=float("nan"),
        )

    rho_dom = rho_block[domain_mask]
    kappa_dom = (ksum[domain_mask] / kcnt[domain_mask])

    ratio = kappa_dom / rho_dom
    g_R = float(np.median(ratio[np.isfinite(ratio)])) if ratio.size else float("nan")

    a_R, b_R = theil_sen_fit(rho_dom, kappa_dom)
    a_LS, b_LS = ls_fit(rho_dom, kappa_dom)

    mean_rho_pos = float(np.mean(rho_dom)) if rho_dom.size else float("nan")

    return ScaleDiagnostics(
        R=int(R),
        blocks_total=blocks_total,
        blocks_with_edges=blocks_with_edges,
        domain_blocks=domain_blocks,
        domain_ok=True,
        mean_rho_pos=mean_rho_pos,
        a_R=float(a_R), b_R=float(b_R),
        a_R_LS=float(a_LS), b_R_LS=float(b_LS),
        g_R=float(g_R),
    )


# ----------------------------
# Plateau selection
# ----------------------------

@dataclass(frozen=True)
class PlateauResult:
    ok: bool
    R_window: List[int]
    rel_var: float
    a_star: float
    a_star_ls: float
    delta_a_star: float


def select_plateau_aR(
    per_R: List[ScaleDiagnostics],
    *,
    window_k: int,
    rel_var_thresh: float,
    median_eps: float,
) -> PlateauResult:
    elig = [d for d in per_R if d.domain_ok and np.isfinite(d.a_R)]
    if len(elig) < int(window_k):
        return PlateauResult(False, [], float("inf"), float("nan"), float("nan"), float("nan"))

    elig.sort(key=lambda d: d.R)
    Rs = [d.R for d in elig]
    a = np.asarray([d.a_R for d in elig], dtype=float)
    a_ls = np.asarray([d.a_R_LS for d in elig], dtype=float)

    k = int(window_k)
    for end in range(len(Rs) - 1, k - 2, -1):
        start = end - k + 1
        w = a[start:end + 1]
        med = float(np.median(w))
        denom = max(abs(med), float(median_eps))
        rel = float((np.max(w) - np.min(w)) / denom)
        if rel <= float(rel_var_thresh):
            a_star = float(np.median(w))
            a_star_ls = float(np.median(a_ls[start:end + 1]))
            return PlateauResult(True, Rs[start:end + 1], rel, a_star, a_star_ls, float(a_star_ls - a_star))

    return PlateauResult(False, [], float("inf"), float("nan"), float("nan"), float("nan"))


# ----------------------------
# a* wrapper
# ----------------------------

def compute_a_star(
    G: nx.MultiDiGraph,
    *,
    r_curv: int,
    R_values: List[int],
    domain_min_blocks: int,
    min_blocks_total: int,
    plateau_k: int,
    plateau_relvar: float,
    plateau_eps: float,
) -> Tuple[PlateauResult, List[ScaleDiagnostics], Condensation, Dict[int, int]]:
    cond = stable_condensation(nx.DiGraph(G))
    C = cond.C

    topo = stable_topological_order(C)
    depth = scc_depths(C)  # used for null constraints + depth-conditioned metrics

    Vr = cone_volume_truncated(C, int(r_curv))
    kappa_edge = curvature_edge_proxy(C, Vr)
    rho = rho_mem(C)

    per_R: List[ScaleDiagnostics] = []
    for R in R_values:
        per_R.append(
            compute_scale_diagnostics(
                C, topo, rho, kappa_edge,
                R=int(R),
                domain_min_blocks=int(domain_min_blocks),
                min_blocks_total=int(min_blocks_total),
                edge_weight=cond.edge_weight,
            )
        )

    plat = select_plateau_aR(
        per_R,
        window_k=int(plateau_k),
        rel_var_thresh=float(plateau_relvar),
        median_eps=float(plateau_eps),
    )
    return plat, per_R, cond, depth


# ----------------------------
# Node labels for null bucket constraints
# ----------------------------

@dataclass(frozen=True)
class NodeLabels:
    node_depth: Dict[int, int]
    node_scc: Dict[int, int]
    scc_depth: Dict[int, int]
    scc_kout: Dict[int, int]
    scc_kin: Dict[int, int]


def compute_node_labels(G: nx.MultiDiGraph) -> NodeLabels:
    cond = stable_condensation(nx.DiGraph(G))
    C = cond.C
    d = scc_depths(C)
    scc_kout = {int(u): int(C.out_degree(int(u))) for u in C.nodes()}
    scc_kin = {int(u): int(C.in_degree(int(u))) for u in C.nodes()}
    node_depth = {int(n): int(d[cond.node_to_scc[int(n)]]) for n in G.nodes()}
    node_scc = {int(n): int(cond.node_to_scc[int(n)]) for n in G.nodes()}
    return NodeLabels(node_depth=node_depth, node_scc=node_scc, scc_depth=d, scc_kout=scc_kout, scc_kin=scc_kin)


def make_null2_edge_type(labels: NodeLabels) -> Callable[[int, int], Tuple[int, int]]:
    def f(u: int, v: int) -> Tuple[int, int]:
        return (int(labels.node_depth[int(u)]), int(labels.node_depth[int(v)]))
    return f


def make_null25_hubcore_edge_type(labels: NodeLabels, *, top_q: float = 0.10) -> Callable[[int, int], Tuple[int, int, int, int]]:
    """
    Edge type:
      (depth(u), depth(v), src_is_topq_kout_in_depth, dst_is_topq_kin_in_depth)
    where hubness is defined at the SCC level within each depth.
    """
    top_q = float(top_q)
    if not (0.0 < top_q <= 0.5):
        raise ValueError("top_q must be in (0,0.5]")

    sccs_by_depth: Dict[int, List[int]] = {}
    for scc, d in labels.scc_depth.items():
        sccs_by_depth.setdefault(int(d), []).append(int(scc))
    for d in sccs_by_depth:
        sccs_by_depth[d].sort()

    src_hubs: Dict[int, set[int]] = {}
    dst_hubs: Dict[int, set[int]] = {}
    for d, sccs in sccs_by_depth.items():
        if not sccs:
            src_hubs[d] = set()
            dst_hubs[d] = set()
            continue
        k = max(1, int(math.ceil(top_q * len(sccs))))
        src_sorted = sorted(sccs, key=lambda s: (-labels.scc_kout[int(s)], int(s)))
        dst_sorted = sorted(sccs, key=lambda s: (-labels.scc_kin[int(s)], int(s)))
        src_hubs[d] = set(src_sorted[:k])
        dst_hubs[d] = set(dst_sorted[:k])

    def f(u: int, v: int) -> Tuple[int, int, int, int]:
        du = int(labels.node_depth[int(u)])
        dv = int(labels.node_depth[int(v)])
        su = int(labels.node_scc[int(u)])
        sv = int(labels.node_scc[int(v)])
        return (du, dv, int(su in src_hubs[du]), int(sv in dst_hubs[dv]))
    return f


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    n_bins = int(n_bins)
    if n_bins <= 1:
        return np.zeros(values.shape[0], dtype=np.int32)
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    qs = np.asarray(qs, dtype=float)
    out = np.zeros(values.shape[0], dtype=np.int32)
    for i, x in enumerate(values):
        b = 0
        for j in range(n_bins):
            if x >= qs[j]:
                b = j
        out[i] = int(min(b, n_bins - 1))
    return out


def make_null25b_bins2d_edge_type(labels: NodeLabels, *, n_bins: int = 4) -> Callable[[int, int], Tuple[int, int, int, int]]:
    """
    Depth-conditioned 2D bins:
      (depth(u), depth(v), src_kout_bin_within_depth, dst_kin_bin_within_depth)
    """
    n_bins = int(max(2, n_bins))

    sccs_by_depth: Dict[int, List[int]] = {}
    for scc, d in labels.scc_depth.items():
        sccs_by_depth.setdefault(int(d), []).append(int(scc))
    for d in sccs_by_depth:
        sccs_by_depth[d].sort()

    src_bin_scc: Dict[int, int] = {}
    dst_bin_scc: Dict[int, int] = {}
    for d, sccs in sccs_by_depth.items():
        kout_vals = np.asarray([labels.scc_kout[int(s)] for s in sccs], dtype=float)
        kin_vals = np.asarray([labels.scc_kin[int(s)] for s in sccs], dtype=float)
        kout_bins = _quantile_bins(kout_vals, n_bins=n_bins)
        kin_bins = _quantile_bins(kin_vals, n_bins=n_bins)
        for i, s in enumerate(sccs):
            src_bin_scc[int(s)] = int(kout_bins[i])
            dst_bin_scc[int(s)] = int(kin_bins[i])

    def f(u: int, v: int) -> Tuple[int, int, int, int]:
        du = int(labels.node_depth[int(u)])
        dv = int(labels.node_depth[int(v)])
        su = int(labels.node_scc[int(u)])
        sv = int(labels.node_scc[int(v)])
        return (du, dv, int(src_bin_scc[su]), int(dst_bin_scc[sv]))
    return f


# ----------------------------
# Degree-preserving bucketed directed double-edge swap
# ----------------------------

@dataclass(frozen=True)
class SwapStats:
    nswap: int
    max_tries: int
    swaps_done: int
    tries: int


def directed_double_edge_swap_bucketed(
    G: nx.MultiDiGraph,
    rng: np.random.Generator,
    *,
    nswap: int,
    max_tries: int,
    edge_type_func: Callable[[int, int], Tuple[Any, ...]],
    allow_self_loops: bool = False,
) -> Tuple[nx.MultiDiGraph, SwapStats]:
    edges = [(int(u), int(v)) for u, v in iter_edges_stable(G)]
    m = len(edges)
    if m < 2 or int(nswap) <= 0:
        return G.copy(), SwapStats(int(nswap), int(max_tries), 0, 0)

    buckets: Dict[Tuple[Any, ...], List[int]] = {}
    for i, (u, v) in enumerate(edges):
        t = tuple(edge_type_func(int(u), int(v)))
        buckets.setdefault(t, []).append(i)

    swaps_done = 0
    tries = 0
    while swaps_done < int(nswap) and tries < int(max_tries):
        tries += 1
        i = int(rng.integers(0, m))
        u, v = edges[i]
        t = tuple(edge_type_func(int(u), int(v)))
        idxs = buckets.get(t, [])
        if len(idxs) < 2:
            continue
        j = int(idxs[int(rng.integers(0, len(idxs)))])
        if j == i:
            continue
        x, y = edges[j]

        # swap targets: u->y and x->v
        if (not allow_self_loops) and (u == y or x == v):
            continue

        edges[i] = (u, y)
        edges[j] = (x, v)
        swaps_done += 1

    H = nx.MultiDiGraph()
    H.add_nodes_from(sorted(int(n) for n in G.nodes()))
    for u, v in edges:
        if (not allow_self_loops) and u == v:
            continue
        H.add_edge(int(u), int(v))

    return H, SwapStats(int(nswap), int(max_tries), swaps_done, tries)


# ----------------------------
# Allocation metrics on condensation DAG edges
# ----------------------------

@dataclass(frozen=True)
class DepthPairMetric:
    d: int
    dp: int
    n_edges: int
    c_assort: float
    c_rich: float


def _weighted_corr(n: float, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    if n <= 1.0:
        return float("nan")
    vx = sxx - (sx * sx) / n
    vy = syy - (sy * sy) / n
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    cov = sxy - (sx * sy) / n
    return float(cov / math.sqrt(vx * vy))


def allocation_metrics_condensation(
    cond: Condensation,
    depth: Dict[int, int],
    *,
    top_q: float = 0.10,
    min_edges_per_pair: int = 25,
    domain_weighted: bool = True,
) -> Tuple[List[DepthPairMetric], float, float]:
    """
    c_assort(d->d') = corr(log1p(kout(src)), log1p(kin(dst))) on condensation edges
    c_rich(d->d')   = hub-core edge fraction on condensation edges
    Weighted by edge multiplicity.
    """
    C = cond.C
    kout = {int(u): int(C.out_degree(int(u))) for u in C.nodes()}
    kin = {int(u): int(C.in_degree(int(u))) for u in C.nodes()}

    # hub sets by depth (source hubs by kout, target hubs by kin)
    sccs_by_depth: Dict[int, List[int]] = {}
    for u in C.nodes():
        u = int(u)
        sccs_by_depth.setdefault(int(depth[u]), []).append(u)
    for d in sccs_by_depth:
        sccs_by_depth[d].sort()

    top_q = float(top_q)
    src_hubs: Dict[int, set[int]] = {}
    dst_hubs: Dict[int, set[int]] = {}
    for d, sccs in sccs_by_depth.items():
        if not sccs:
            src_hubs[d] = set()
            dst_hubs[d] = set()
            continue
        k = max(1, int(math.ceil(top_q * len(sccs))))
        src_sorted = sorted(sccs, key=lambda u: (-kout[u], u))
        dst_sorted = sorted(sccs, key=lambda u: (-kin[u], u))
        src_hubs[d] = set(src_sorted[:k])
        dst_hubs[d] = set(dst_sorted[:k])

    # sums per (d,dp)
    sums: Dict[Tuple[int, int], List[float]] = {}  # [w, sx, sy, sxx, syy, sxy, edge_count, rich_w, tot_w]

    for u, v in iter_edges_stable(C):
        u = int(u); v = int(v)
        if domain_weighted and kout[u] < 2:
            continue
        d = int(depth[u]); dp = int(depth[v])
        w = float(cond.edge_weight.get((u, v), 1))
        key = (d, dp)
        if key not in sums:
            sums[key] = [0.0] * 9

        x = float(np.log1p(kout[u]))
        y = float(np.log1p(kin[v]))
        s = sums[key]
        s[0] += w
        s[1] += w * x
        s[2] += w * y
        s[3] += w * x * x
        s[4] += w * y * y
        s[5] += w * x * y
        s[6] += 1.0
        s[8] += w
        if (u in src_hubs[d]) and (v in dst_hubs[dp]):
            s[7] += w

    metrics: List[DepthPairMetric] = []
    for (d, dp), s in sorted(sums.items()):
        n_edges = int(s[6])
        cA = float("nan")
        cR = float("nan")
        if n_edges >= int(min_edges_per_pair) and s[0] > 1.0:
            cA = _weighted_corr(s[0], s[1], s[2], s[3], s[4], s[5])
            cR = float(s[7] / s[8]) if s[8] > 0 else float("nan")
        metrics.append(DepthPairMetric(int(d), int(dp), n_edges, float(cA), float(cR)))

    def wavg(vals: List[Tuple[float, int]]) -> float:
        num = 0.0
        den = 0.0
        for v, n in vals:
            if np.isfinite(v) and n > 0:
                num += float(v) * float(n)
                den += float(n)
        return float(num / den) if den > 0 else float("nan")

    c_assort_global = wavg([(m.c_assort, m.n_edges) for m in metrics])
    c_rich_global = wavg([(m.c_rich, m.n_edges) for m in metrics])
    return metrics, float(c_assort_global), float(c_rich_global)


# ----------------------------
# Null stats (z + empirical two-sided p)
# ----------------------------

@dataclass(frozen=True)
class NullStats:
    mean: float
    std: float
    z: float
    p_emp: float
    n_eff: int


def null_stats(obs: float, null_vals: List[float]) -> NullStats:
    xs = np.asarray([x for x in null_vals if np.isfinite(x)], dtype=float)
    n = int(xs.size)
    if (not np.isfinite(obs)) or n == 0:
        return NullStats(float("nan"), float("nan"), float("nan"), float("nan"), n)
    mu = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1)) if n >= 2 else 0.0
    z = float((obs - mu) / sd) if sd > 0 else float("nan")
    dev = abs(obs - mu)
    p = float((1 + int(np.sum(np.abs(xs - mu) >= dev))) / (n + 1))
    return NullStats(mu, sd, z, p, n)
