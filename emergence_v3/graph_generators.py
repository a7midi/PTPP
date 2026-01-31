from __future__ import annotations

"""
Graph family generators with deterministic seeding and explicit orientation schemes.

The geometry module (G1) is defined on the *condensation DAG* G↓. If the input graph is
strongly connected (or nearly), G↓ collapses to a tiny DAG and block observables become
ill-defined. For PRL-ready diagnostics we therefore expose an explicit "orientation" knob that
can turn dense cyclic substrates into causal (DAG-like) substrates in a *declared* and reproducible way.

This file is intentionally self-contained and avoids "magic numbers":
all family parameters are supplied via experiment.yaml/config.yaml.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import math
import networkx as nx
import numpy as np


def _seed_int(rng: np.random.Generator) -> int:
    """Convert numpy Generator state into a deterministic 32-bit seed for networkx functions."""
    return int(rng.integers(0, 2**32 - 1, dtype=np.uint32))


def _random_order(N: int, rng: np.random.Generator) -> np.ndarray:
    order = np.arange(int(N), dtype=np.int32)
    rng.shuffle(order)
    return order


def _rank_from_order(order: np.ndarray) -> np.ndarray:
    """rank[node] = position in order"""
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size, dtype=np.int32)
    return rank


def causalize_directed_flip(DG: nx.DiGraph, rng: np.random.Generator, *, allow_self_loops: bool) -> nx.DiGraph:
    """Make a directed graph acyclic by flipping edges to point forward along a random node order."""
    N = DG.number_of_nodes()
    order = _random_order(N, rng)
    rank = _rank_from_order(order)

    out = nx.DiGraph()
    out.add_nodes_from(DG.nodes())
    for u, v in sorted(DG.edges()):
        u = int(u)
        v = int(v)
        if u == v:
            if allow_self_loops:
                out.add_edge(u, v)
            continue
        if int(rank[u]) < int(rank[v]):
            out.add_edge(u, v)
        else:
            out.add_edge(v, u)
    return out


def orient_undirected_causal_order(UG: nx.Graph, rng: np.random.Generator, *, allow_self_loops: bool) -> nx.DiGraph:
    """Orient an undirected graph into a DAG using a random node order (one directed edge per undirected edge)."""
    N = UG.number_of_nodes()
    order = _random_order(N, rng)
    rank = _rank_from_order(order)

    out = nx.DiGraph()
    out.add_nodes_from(UG.nodes())
    for u, v in sorted(UG.edges()):
        u = int(u)
        v = int(v)
        if u == v:
            if allow_self_loops:
                out.add_edge(u, v)
            continue
        if int(rank[u]) < int(rank[v]):
            out.add_edge(u, v)
        else:
            out.add_edge(v, u)
    return out


def _to_multidigraph(DG: nx.DiGraph, *, allow_self_loops: bool) -> nx.MultiDiGraph:
    MG = nx.MultiDiGraph()
    MG.add_nodes_from(int(v) for v in DG.nodes())
    for u, v in DG.edges():
        u = int(u)
        v = int(v)
        if (not allow_self_loops) and u == v:
            continue
        MG.add_edge(u, v)
    return MG


def _meta_basic(G: nx.MultiDiGraph, family: str, params: Dict[str, Any], orientation: str) -> Dict[str, Any]:
    N = int(G.number_of_nodes())
    E = int(G.number_of_edges())
    mean_out = float(E / N) if N > 0 else 0.0
    return {
        "family": family,
        "N": N,
        "E": E,
        "mean_out_degree": mean_out,
        "orientation": orientation,
        "params": dict(params),
    }


def _generate_undirected_sbm(
    *,
    sizes: List[int],
    p_matrix: np.ndarray,
    rng: np.random.Generator,
    allow_self_loops: bool,
) -> nx.Graph:
    """Generate an undirected SBM using numpy RNG (deterministic).

    Parameters
    ----------
    sizes:
        List of block sizes (must sum to N).
    p_matrix:
        KxK matrix of connection probabilities.
    rng:
        numpy Generator.
    allow_self_loops:
        Whether to include self-loops when i==j (diagonal sampling).

    Notes
    -----
    - Nodes are labeled 0..N-1 in block order.
    - For i<j we sample the full bipartite adjacency once.
    - For i==j we sample the upper triangle; include diagonal only if allow_self_loops.
    """
    sizes = [int(s) for s in sizes]
    if any(s < 0 for s in sizes):
        raise ValueError(f"SBM sizes must be nonnegative, got {sizes!r}")
    N = int(sum(sizes))
    K = int(len(sizes))

    p_matrix = np.asarray(p_matrix, dtype=float)
    if p_matrix.shape != (K, K):
        raise ValueError(f"SBM p_matrix must have shape ({K},{K}), got {p_matrix.shape!r}")
    if np.any(p_matrix < 0.0) or np.any(p_matrix > 1.0):
        raise ValueError("SBM probabilities must be in [0,1]")

    UG = nx.Graph()
    UG.add_nodes_from(range(N))

    # Block index ranges
    blocks: List[np.ndarray] = []
    start = 0
    for s in sizes:
        blocks.append(np.arange(start, start + s, dtype=np.int32))
        start += s

    for i in range(K):
        nodes_i = blocks[i]
        ni = int(nodes_i.size)
        if ni == 0:
            continue
        for j in range(i, K):
            nodes_j = blocks[j]
            nj = int(nodes_j.size)
            if nj == 0:
                continue
            p = float(p_matrix[i, j])
            if p <= 0.0:
                continue

            if i == j:
                # Upper triangle within block
                mat = rng.random((ni, ni))
                if allow_self_loops:
                    mask = np.triu(mat < p, k=0)
                else:
                    mask = np.triu(mat < p, k=1)
                rr, cc = np.where(mask)
                for a, b in zip(rr.tolist(), cc.tolist()):
                    u = int(nodes_i[a])
                    v = int(nodes_i[b])
                    if (not allow_self_loops) and u == v:
                        continue
                    UG.add_edge(u, v)
            else:
                # Bipartite block
                mat = rng.random((ni, nj))
                rr, cc = np.where(mat < p)
                for a, b in zip(rr.tolist(), cc.tolist()):
                    UG.add_edge(int(nodes_i[a]), int(nodes_j[b]))

    return UG


def generate_graph(
    family: str,
    *,
    N: int,
    mean_out_degree: float,
    params: Dict[str, Any],
    allow_self_loops: bool,
    rng: np.random.Generator,
) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
    """
    Generate a seeded graph for the universality sweep.

    Parameters
    ----------
    family:
        One of:
          - "Erdos_Renyi" (aliases: ER, Erdos-Renyi, etc.)
          - "Barabasi_Albert" (aliases: BA, Barabasi-Albert, etc.)
          - "Watts_Strogatz" (aliases: WS, Watts-Strogatz, etc.)
          - "Random_Geometric" (aliases: RGG, Random-Geometric, etc.)
          - "SBM" (aliases: Stochastic_Block_Model, etc.)
    N:
        Number of nodes.
    mean_out_degree:
        Target mean out-degree used either directly or indirectly (depending on family).
        For SBM, this is informational unless you omit probabilities (not supported).
    params:
        Family-specific parameters from experiment.yaml.
        May include:
          - orientation: "none" | "causalize_flip" | "causal_order"
        For SBM:
          - sizes: list[int] summing to N (default [N])
          - p: KxK probability matrix, OR (p_in and p_out)
          - p_in: within-block probability (if using p_in/p_out form)
          - p_out: between-block probability (if using p_in/p_out form)
    allow_self_loops:
        Whether to keep self-loops.
    rng:
        numpy Generator used for reproducibility.

    Returns
    -------
    (G, meta)
      G: networkx.MultiDiGraph with nodes 0..N-1
      meta: dict describing the realization
    """
    family = str(family).strip()

    # Friendly aliases to avoid CLI surprises.
    fam_upper = family.upper()
    if fam_upper in {"ER", "ERDOS_RENYI", "ERDOS-RENYI", "ERDOSRENYI"}:
        family = "Erdos_Renyi"
    elif fam_upper in {"BA", "BARABASI_ALBERT", "BARABASI-ALBERT", "BARABASIALBERT"}:
        family = "Barabasi_Albert"
    elif fam_upper in {"WS", "WATTS_STROGATZ", "WATTS-STROGATZ", "WATTSSTROGATZ"}:
        family = "Watts_Strogatz"
    elif fam_upper in {"RGG", "RANDOM_GEOMETRIC", "RANDOM-GEOMETRIC", "RANDOMGEOMETRIC"}:
        family = "Random_Geometric"
    elif fam_upper in {"SBM", "STOCHASTIC_BLOCK_MODEL", "STOCHASTIC-BLOCK-MODEL", "STOCHASTICBLOCKMODEL"}:
        family = "SBM"

    N = int(N)
    mean_out_degree = float(mean_out_degree)
    orientation = str(params.get("orientation", "causalize_flip" if family in {"Erdos_Renyi"} else "causal_order"))

    if family == "Erdos_Renyi":
        # Generate a directed ER graph with probability p for each ordered pair.
        # If params['p'] exists, treat it as the base probability; otherwise compute from mean_out_degree.
        p = float(params.get("p", mean_out_degree / max(1, (N - 1))))
        # networkx uses python's random; we pass an integer seed for determinism
        DG0 = nx.gnp_random_graph(N, p, seed=_seed_int(rng), directed=True)
        if not allow_self_loops:
            DG0.remove_edges_from(list(nx.selfloop_edges(DG0)))

        if orientation in {"causalize_flip", "causal_order"}:
            DG = causalize_directed_flip(nx.DiGraph(DG0), rng, allow_self_loops=allow_self_loops)
        else:
            DG = nx.DiGraph(DG0)

        MG = _to_multidigraph(DG, allow_self_loops=allow_self_loops)
        return MG, _meta_basic(MG, family, {"p": p, **params}, orientation)

    if family == "Barabasi_Albert":
        m = int(params.get("m", max(1, int(round(mean_out_degree)))))
        UG = nx.barabasi_albert_graph(N, m, seed=_seed_int(rng))
        if orientation in {"causal_order", "causalize_flip"}:
            DG = orient_undirected_causal_order(UG, rng, allow_self_loops=allow_self_loops)
        else:
            # fallback: arbitrarily direct each edge both ways (NOT recommended; likely SCC collapse)
            DG = nx.DiGraph()
            DG.add_nodes_from(UG.nodes())
            for u, v in UG.edges():
                DG.add_edge(u, v)
                DG.add_edge(v, u)
        MG = _to_multidigraph(DG, allow_self_loops=allow_self_loops)
        return MG, _meta_basic(MG, family, {"m": m, **params}, orientation)

    if family == "Watts_Strogatz":
        # params.k is interpreted as target mean out-degree after causal orientation,
        # so undirected degree is 2k and must be even.
        k = int(params.get("k", max(1, int(round(mean_out_degree)))))
        p_rewire = float(params.get("p", 0.1))
        k_und = int(2 * k)
        if k_und >= N:
            k_und = max(2, N - 1) if (N - 1) % 2 == 0 else max(2, N - 2)
        UG = nx.watts_strogatz_graph(N, k_und, p_rewire, seed=_seed_int(rng))
        DG = (
            orient_undirected_causal_order(UG, rng, allow_self_loops=allow_self_loops)
            if orientation in {"causal_order", "causalize_flip"}
            else nx.DiGraph(UG)
        )
        MG = _to_multidigraph(DG, allow_self_loops=allow_self_loops)
        return MG, _meta_basic(MG, family, {"k": k, "p": p_rewire, **params}, orientation)

    if family == "Random_Geometric":
        # If radius not provided, set approximate radius to target mean out-degree under causal orientation.
        # Expected undirected degree ~ (N-1)*pi*r^2 (ignoring boundaries).
        # After orienting each undirected edge once, expected out-degree ~ 0.5 * E[deg_und].
        radius = params.get("radius", None)
        if radius is None:
            radius = math.sqrt(max(1e-12, (2.0 * mean_out_degree) / (math.pi * max(1.0, (N - 1)))))
        radius = float(radius)
        UG = nx.random_geometric_graph(N, radius, seed=_seed_int(rng))
        DG = (
            orient_undirected_causal_order(UG, rng, allow_self_loops=allow_self_loops)
            if orientation in {"causal_order", "causalize_flip"}
            else nx.DiGraph(UG)
        )
        MG = _to_multidigraph(DG, allow_self_loops=allow_self_loops)
        return MG, _meta_basic(MG, family, {"radius": radius, **params}, orientation)

    if family == "SBM":
        # Stochastic Block Model generated as an UNDIRECTED SBM, then oriented into a DAG (recommended)
        # unless orientation == "none".
        sizes = params.get("sizes", None)
        if sizes is None:
            sizes = [N]
        sizes = [int(s) for s in sizes]
        if int(sum(sizes)) != N:
            raise ValueError(f"SBM sizes must sum to N={N}, got sizes={sizes!r} (sum={sum(sizes)})")

        # Accept either a full probability matrix `p` or a (p_in, p_out) pair.
        p = params.get("p", None)
        if p is not None:
            p_matrix = np.asarray(p, dtype=float)
        else:
            if ("p_in" not in params) or ("p_out" not in params):
                raise ValueError("SBM requires either params['p'] (KxK matrix) or both params['p_in'] and params['p_out']")
            p_in = float(params["p_in"])
            p_out = float(params["p_out"])
            K = int(len(sizes))
            p_matrix = np.full((K, K), p_out, dtype=float)
            np.fill_diagonal(p_matrix, p_in)

        UG = _generate_undirected_sbm(
            sizes=sizes,
            p_matrix=p_matrix,
            rng=rng,
            allow_self_loops=allow_self_loops,
        )

        if orientation in {"causal_order", "causalize_flip"}:
            DG = orient_undirected_causal_order(UG, rng, allow_self_loops=allow_self_loops)
        else:
            # WARNING: doubling edges typically creates a giant SCC and collapses the condensation DAG.
            DG = nx.DiGraph()
            DG.add_nodes_from(UG.nodes())
            for u, v in UG.edges():
                DG.add_edge(int(u), int(v))
                DG.add_edge(int(v), int(u))
            if not allow_self_loops:
                DG.remove_edges_from(list(nx.selfloop_edges(DG)))

        MG = _to_multidigraph(DG, allow_self_loops=allow_self_loops)
        meta_params = {"sizes": sizes, "p_matrix": np.asarray(p_matrix, dtype=float).tolist(), **params}
        return MG, _meta_basic(MG, family, meta_params, orientation)

    raise ValueError(f"Unknown graph family: {family!r}")
