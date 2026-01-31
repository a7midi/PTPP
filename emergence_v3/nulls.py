from __future__ import annotations

"""Null-model rewiring operators (directed) used for universality stress tests.

We implement a nested family of constrained directed double-edge swaps on a *simple* nx.DiGraph:

Null-1 (degree-preserving):
  - preserves in/out degree sequence exactly

Null-2 (degree + depth-profile preserving):
  - preserves in/out degree sequence
  - preserves a fixed node depth label derived from SCC-depth in the *original* graph
  - preserves edge-count matrix between depth layers by swapping only within (depth(src), depth(dst)) buckets

Null-3 (degree + community preserving):
  - preserves in/out degree sequence
  - preserves a fixed community partition (detected on the original graph)
  - preserves edge-count matrix between community pairs by swapping only within (comm(src), comm(dst)) buckets

Null-2.5 (degree + depth-profile + allocation operator):
  A refinement of Null-2 designed to preserve *coarse within-bucket allocation statistics* that Null-2 destroys.
  Two implemented variants:

  - deg_bins: preserve (depth(src), depth(dst), out_bin(src), in_bin(dst))
      where out_bin / in_bin are degree-quantile-like bins computed *within each depth slice* on the ORIGINAL graph.

  - hub_core: preserve (depth(src), depth(dst), core_out(src), core_in(dst))
      where core_out/core_in indicate whether a node lies in the top-q% out-/in-degree set within its depth slice
      on the ORIGINAL graph.

This module also provides two allocation diagnostics that are sensitive to within-bucket wiring:

  c_assort: weighted average (over depth pairs) of edge-level correlation
      corr(log(1+k_out(u)), log(1+k_in(v))) restricted to edges u->v in each depth pair (d,d').

  c_rich: fraction of edges that connect depth-slice "core" sources to "core" targets
      (top-q% out-degree within source depth) -> (top-q% in-degree within target depth).

Important design choices (for reproducibility / review-proofness):
  - all randomness is driven by a provided NumPy Generator
  - SCC labeling + depths are stable via emergence_v3.geometry.condensation_dag_stable
  - community labels are stabilized by sorting communities by min node id
  - degree-bin and hub-core labels are deterministic (tie-broken by node id)
"""

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np
import networkx as nx
import math

from emergence_v3.geometry import condensation_dag_stable, dag_depths_longest_path_ending


@dataclass(frozen=True)
class RewireStats:
    nswap_target: int
    max_tries: int
    tries: int
    swaps_done: int

    @property
    def swap_success_rate(self) -> float:
        return float(self.swaps_done / max(1, self.tries))


# --------------------------------------------------------------------------------------
# Depth + community labels (frozen on the ORIGINAL graph)
# --------------------------------------------------------------------------------------


def compute_node_depth_labels_from_scc_depth(G: nx.DiGraph) -> Dict[int, int]:
    """Assign each node a depth label using SCC-depth in the condensation DAG.

    Depth definition matches emergence_v3.geometry.dag_depths_longest_path_ending:
      depth(X) = length of a longest directed path ending at SCC X.

    The returned mapping is deterministic for a fixed node labeling.
    """
    C, scc_map = condensation_dag_stable(nx.DiGraph(G))
    if C.number_of_nodes() == 0:
        return {int(v): 0 for v in G.nodes()}
    scc_depth = dag_depths_longest_path_ending(C)
    depth_label: Dict[int, int] = {}
    for v in G.nodes():
        sv = int(scc_map.get(int(v), 0))
        depth_label[int(v)] = int(scc_depth.get(sv, 0))
    return depth_label


def detect_communities_louvain(
    G: nx.DiGraph,
    *,
    seed: int,
    resolution: float = 1.0,
) -> Dict[int, int]:
    """Detect communities using Louvain on an undirected projection.

    NetworkX's louvain_communities is seedable; we stabilize labels by
    sorting communities by min node id.
    """
    U = nx.Graph()
    U.add_nodes_from(int(v) for v in G.nodes())
    U.add_edges_from((int(u), int(v)) for (u, v) in G.edges())

    comms = nx.algorithms.community.louvain_communities(
        U,
        seed=int(seed),
        resolution=float(resolution),
    )
    comms_sorted = sorted((set(int(x) for x in c) for c in comms), key=lambda c: min(c) if c else -1)
    mapping: Dict[int, int] = {}
    for i, c in enumerate(comms_sorted):
        for v in c:
            mapping[int(v)] = int(i)
    # Any isolated / missed nodes (should not happen) fall back to singleton labels.
    next_id = len(comms_sorted)
    for v in U.nodes():
        vv = int(v)
        if vv not in mapping:
            mapping[vv] = next_id
            next_id += 1
    return mapping


# --------------------------------------------------------------------------------------
# Directed degree-preserving double-edge swap (core rewire engine)
# --------------------------------------------------------------------------------------


def directed_double_edge_swap(
    G: nx.DiGraph,
    *,
    rng: np.random.Generator,
    nswap: int,
    max_tries: int,
    allow_self_loops: bool = False,
    edge_type_func: Optional[Callable[[int, int], Hashable]] = None,
) -> Tuple[nx.DiGraph, RewireStats]:
    """Directed double-edge swaps preserving in/out degrees.

    If edge_type_func is provided, swaps are constrained to preserve the edge 'type':
      - only pick two edges with the same edge_type_func(u, v)
      - after swapping endpoints, the new edges remain in the same bucket

    This is the workhorse for:
      - Null-1 (edge_type_func=None)
      - Null-2 (edge_type_func=(depth(src), depth(dst)))
      - Null-3 (edge_type_func=(comm(src), comm(dst)))
      - Null-2.5 (edge_type_func=(depth,depth,out_bin,in_bin) or (depth,depth,core_out,core_in))

    Notes:
      - Operates on simple DiGraph (parallel edges are rejected).
      - Rejects swaps that would create existing edges (no multiedges).
    """
    H = nx.DiGraph(G)

    edges: List[Tuple[int, int]] = [(int(u), int(v)) for (u, v) in H.edges()]
    m = len(edges)
    if m < 2:
        return H, RewireStats(int(nswap), int(max_tries), 0, 0)

    edge_set = set(edges)
    pos = {e: i for i, e in enumerate(edges)}

    buckets: Optional[Dict[Hashable, List[int]]] = None
    eligible_bucket_keys: Optional[List[Hashable]] = None
    edge_types: Optional[List[Hashable]] = None
    if edge_type_func is not None:
        edge_types = [edge_type_func(u, v) for (u, v) in edges]
        buckets = {}
        for idx, t in enumerate(edge_types):
            buckets.setdefault(t, []).append(idx)
        eligible_bucket_keys = [k for k, idxs in buckets.items() if len(idxs) >= 2]
        if not eligible_bucket_keys:
            # No buckets with enough edges to swap.
            return H, RewireStats(int(nswap), int(max_tries), 0, 0)

    swaps = 0
    tries = 0
    nswap = int(max(0, nswap))
    max_tries = int(max(0, max_tries))

    while swaps < nswap and tries < max_tries:
        tries += 1

        if buckets is None:
            i = int(rng.integers(0, m))
            j = int(rng.integers(0, m))
        else:
            # Choose a bucket (uniform over eligible buckets), then choose two distinct indices inside.
            k = eligible_bucket_keys[int(rng.integers(0, len(eligible_bucket_keys)))]
            idxs = buckets[k]
            if len(idxs) < 2:
                continue
            a = int(rng.integers(0, len(idxs)))
            b = int(rng.integers(0, len(idxs)))
            if a == b:
                continue
            i = int(idxs[a])
            j = int(idxs[b])

        if i == j:
            continue
        e1 = edges[i]
        e2 = edges[j]
        if e1 == e2:
            continue

        u, v = e1
        x, y = e2
        # Avoid degenerate overlaps to keep the simple-graph guard clean.
        if len({u, v, x, y}) < 4:
            continue

        new1 = (u, y)
        new2 = (x, v)
        if (not allow_self_loops) and (new1[0] == new1[1] or new2[0] == new2[1]):
            continue

        # Preserve edge types if requested.
        if edge_type_func is not None and edge_types is not None:
            t1 = edge_types[i]
            t2 = edge_types[j]
            if t1 != t2:
                continue
            if edge_type_func(new1[0], new1[1]) != t1:
                continue
            if edge_type_func(new2[0], new2[1]) != t1:
                continue

        # Parallel-edge guard (simple digraph)
        if new1 in edge_set or new2 in edge_set:
            continue

        # Execute
        H.remove_edge(u, v)
        H.remove_edge(x, y)
        H.add_edge(*new1)
        H.add_edge(*new2)

        # Update bookkeeping
        for old, new in ((e1, new1), (e2, new2)):
            edge_set.remove(old)
            edge_set.add(new)
            idx = pos.pop(old)
            edges[idx] = new
            pos[new] = idx
            if edge_types is not None and edge_type_func is not None:
                # Type must be unchanged; still, refresh to keep invariants explicit.
                edge_types[idx] = edge_type_func(new[0], new[1])
        swaps += 1

    return H, RewireStats(int(nswap), int(max_tries), int(tries), int(swaps))


def edge_type_from_node_labels(node_label: Dict[int, int]) -> Callable[[int, int], Tuple[int, int]]:
    """Return edge_type_func(u,v) = (label(u), label(v))."""
    return lambda u, v: (int(node_label.get(int(u), 0)), int(node_label.get(int(v), 0)))


# --------------------------------------------------------------------------------------
# Allocation diagnostics (computed w.r.t. frozen ORIGINAL depth labels)
# --------------------------------------------------------------------------------------


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation; return nan if undefined."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or y.size < 3:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if not (np.isfinite(sx) and np.isfinite(sy)) or sx <= 0.0 or sy <= 0.0:
        return float("nan")
    c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else float("nan")


def compute_c_assort(
    G: nx.DiGraph,
    *,
    depth_label: Dict[int, int],
    min_edges_per_pair: int = 25,
) -> Dict[str, float]:
    """Compute depth-pair endpoint assortativity operator c_assort.

    For each depth pair (d,d'), compute correlation over edges u->v with depth(u)=d, depth(v)=d':
      corr(log(1+k_out(u)), log(1+k_in(v))).

    Return a weighted average over depth pairs with at least min_edges_per_pair edges.

    Returns a dict with:
      c_assort, c_assort_pairs_used, c_assort_edges_used
    """
    kout = {int(v): int(G.out_degree(int(v))) for v in G.nodes()}
    kin = {int(v): int(G.in_degree(int(v))) for v in G.nodes()}

    # Bucket by depth pair
    buckets: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for (u, v) in G.edges():
        uu = int(u)
        vv = int(v)
        d = int(depth_label.get(uu, 0))
        dp = int(depth_label.get(vv, 0))
        buckets.setdefault((d, dp), []).append((math.log1p(kout[uu]), math.log1p(kin[vv])))

    used_pairs = 0
    used_edges = 0
    num = 0.0
    den = 0.0
    for (d, dp), pairs in buckets.items():
        n = len(pairs)
        if n < int(min_edges_per_pair):
            continue
        xs = np.array([p[0] for p in pairs], dtype=float)
        ys = np.array([p[1] for p in pairs], dtype=float)
        c = _safe_corrcoef(xs, ys)
        if not np.isfinite(c):
            continue
        used_pairs += 1
        used_edges += n
        num += float(n) * float(c)
        den += float(n)

    c_assort = float(num / den) if den > 0 else float("nan")
    return {
        "c_assort": c_assort,
        "c_assort_pairs_used": float(used_pairs),
        "c_assort_edges_used": float(used_edges),
    }


def _top_set_within_depth(
    degrees: Dict[int, int],
    depth_label: Dict[int, int],
    *,
    top_q: float,
) -> Dict[int, set]:
    """Return mapping depth -> set(nodes) of top-q by degree inside each depth slice.

    Deterministic tie-breaking: sort by (-deg, node_id).
    """
    top_q = float(top_q)
    if not (0.0 < top_q <= 1.0):
        raise ValueError("top_q must be in (0,1]")
    nodes_by_depth: Dict[int, List[int]] = {}
    for v, d in depth_label.items():
        nodes_by_depth.setdefault(int(d), []).append(int(v))

    out: Dict[int, set] = {}
    for d, nodes in nodes_by_depth.items():
        nodes_sorted = sorted(nodes, key=lambda x: (-int(degrees.get(int(x), 0)), int(x)))
        n = len(nodes_sorted)
        if n == 0:
            out[d] = set()
            continue
        k = int(max(1, math.ceil(top_q * n)))
        out[d] = set(nodes_sorted[:k])
    return out


def compute_c_rich(
    G: nx.DiGraph,
    *,
    depth_label: Dict[int, int],
    top_q: float = 0.1,
) -> Dict[str, float]:
    """Compute within-depth rich-club allocation operator c_rich.

    We define "core" sources within each source depth d as the top-q% out-degree nodes in that depth,
    and "core" targets within each target depth d' as the top-q% in-degree nodes in that depth.

    c_rich is the fraction of edges u->v for which u is a core_out node and v is a core_in node,
    where core membership is defined within each depth slice.

    Returns a dict with:
      c_rich, c_rich_top_q, c_rich_edges_rich, c_rich_edges_total
    """
    kout = {int(v): int(G.out_degree(int(v))) for v in G.nodes()}
    kin = {int(v): int(G.in_degree(int(v))) for v in G.nodes()}

    core_out_by_depth = _top_set_within_depth(kout, depth_label, top_q=float(top_q))
    core_in_by_depth = _top_set_within_depth(kin, depth_label, top_q=float(top_q))

    rich = 0
    total = 0
    for (u, v) in G.edges():
        uu = int(u)
        vv = int(v)
        du = int(depth_label.get(uu, 0))
        dv = int(depth_label.get(vv, 0))
        total += 1
        if uu in core_out_by_depth.get(du, set()) and vv in core_in_by_depth.get(dv, set()):
            rich += 1

    c_rich = float(rich / total) if total > 0 else float("nan")
    return {
        "c_rich": c_rich,
        "c_rich_top_q": float(top_q),
        "c_rich_edges_rich": float(rich),
        "c_rich_edges_total": float(total),
    }


# --------------------------------------------------------------------------------------
# Null-2.5 edge-type builders (frozen labels from ORIGINAL graph)
# --------------------------------------------------------------------------------------


def _bin_labels_within_depth(
    degrees: Dict[int, int],
    depth_label: Dict[int, int],
    *,
    n_bins: int,
) -> Dict[int, int]:
    """Assign each node a bin index in {0..n_bins-1} based on its degree rank within its depth slice.

    Deterministic: sort nodes by (degree, node_id), assign bins by rank.
    Bins are approximately equal-sized; ties are resolved by node id.
    """
    n_bins = int(n_bins)
    if n_bins <= 1:
        return {int(v): 0 for v in degrees.keys()}

    nodes_by_depth: Dict[int, List[int]] = {}
    for v in degrees.keys():
        d = int(depth_label.get(int(v), 0))
        nodes_by_depth.setdefault(d, []).append(int(v))

    out: Dict[int, int] = {}
    for d, nodes in nodes_by_depth.items():
        nodes_sorted = sorted(nodes, key=lambda x: (int(degrees.get(int(x), 0)), int(x)))
        n = len(nodes_sorted)
        if n == 0:
            continue
        for idx, v in enumerate(nodes_sorted):
            b = int(min(n_bins - 1, (n_bins * idx) // max(1, n)))
            out[int(v)] = b
    return out


def build_null25_edge_type_depth_deg_bins(
    G: nx.DiGraph,
    *,
    depth_label: Dict[int, int],
    n_bins: int = 4,
) -> Callable[[int, int], Tuple[int, int, int, int]]:
    """Edge type for Null-2.5a (degree-bin constrained within depth pairs).

    type(u->v) = (depth(u), depth(v), out_bin(u), in_bin(v)),
    where out_bin and in_bin are degree-rank bins computed within each depth slice on the ORIGINAL graph.
    """
    kout = {int(v): int(G.out_degree(int(v))) for v in G.nodes()}
    kin = {int(v): int(G.in_degree(int(v))) for v in G.nodes()}
    out_bin = _bin_labels_within_depth(kout, depth_label, n_bins=int(n_bins))
    in_bin = _bin_labels_within_depth(kin, depth_label, n_bins=int(n_bins))

    def et(u: int, v: int) -> Tuple[int, int, int, int]:
        uu = int(u)
        vv = int(v)
        return (
            int(depth_label.get(uu, 0)),
            int(depth_label.get(vv, 0)),
            int(out_bin.get(uu, 0)),
            int(in_bin.get(vv, 0)),
        )

    return et


def build_null25_edge_type_depth_hub_core(
    G: nx.DiGraph,
    *,
    depth_label: Dict[int, int],
    top_q: float = 0.1,
) -> Callable[[int, int], Tuple[int, int, int, int]]:
    """Edge type for Null-2.5b (hub-core constrained within depth pairs).

    type(u->v) = (depth(u), depth(v), core_out(u), core_in(v)),
    where core_out/core_in indicate membership in top-q% out-/in-degree sets within each depth slice
    on the ORIGINAL graph.
    """
    kout = {int(v): int(G.out_degree(int(v))) for v in G.nodes()}
    kin = {int(v): int(G.in_degree(int(v))) for v in G.nodes()}

    core_out_by_depth = _top_set_within_depth(kout, depth_label, top_q=float(top_q))
    core_in_by_depth = _top_set_within_depth(kin, depth_label, top_q=float(top_q))

    core_out = {int(v): 0 for v in G.nodes()}
    core_in = {int(v): 0 for v in G.nodes()}
    for d, s in core_out_by_depth.items():
        for v in s:
            core_out[int(v)] = 1
    for d, s in core_in_by_depth.items():
        for v in s:
            core_in[int(v)] = 1

    def et(u: int, v: int) -> Tuple[int, int, int, int]:
        uu = int(u)
        vv = int(v)
        return (
            int(depth_label.get(uu, 0)),
            int(depth_label.get(vv, 0)),
            int(core_out.get(uu, 0)),
            int(core_in.get(vv, 0)),
        )

    return et
