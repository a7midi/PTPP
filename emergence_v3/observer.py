from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import networkx as nx

from .rules import essential_predecessors, single_coordinate_degeneracy
from .world import WorldInstance
from .scdc import condensation_dag, dag_depths


@dataclass(frozen=True)
class PocketSpec:
    """Observer pocket specification.

    pocket_nodes: P ⊆ V
    visible_panel: B(P) ⊆ P
    """
    pocket_nodes: Set[int]
    visible_panel: Set[int]


def predecessor_hull(G: nx.MultiDiGraph, S: Set[int]) -> Set[int]:
    """PredCl(S): predecessor closure (Definition 4.2)."""
    hull: Set[int] = set()
    stack = list(S)
    while stack:
        v = int(stack.pop())
        if v in hull:
            continue
        hull.add(v)
        for p in G.predecessors(v):
            p = int(p)
            if p not in hull:
                stack.append(p)
    return hull


def compute_ess_sets(world: WorldInstance) -> Dict[int, Set[int]]:
    """Ess(v) for every node v (Definition 4.1)."""
    return {v: set(essential_predecessors(world.rules[v])) for v in world.node_ids}


def compute_sigma(world: WorldInstance) -> Dict[int, int]:
    """σ(d) for every node d (Definition 5.4)."""
    sigma: Dict[int, int] = {}
    for v in world.node_ids:
        out_size = len(world.alphabets[v])
        sigma[v] = single_coordinate_degeneracy(world.rules[v], output_size=out_size)
    return sigma


def closure_operator(world: WorldInstance, S: Set[int], *, ess: Dict[int, Set[int]]) -> Set[int]:
    """Cl(S): closure as least fixed point of C(S)=PredCl(S)∪Force(S) (Definition 4.3).

    Notes:
    - We implement this as an incremental fixed-point over the full node set.
    - Correctness matters more than micro-optimizations here; pockets are tiny in the Phase-2 suite.
    """
    G = world.graph

    # reverse map: u -> nodes v for which u is an essential predecessor (u ∈ Ess(v))
    essential_succs: Dict[int, List[int]] = {u: [] for u in world.node_ids}
    for v, ess_set in ess.items():
        for u in ess_set:
            essential_succs[u].append(v)

    ess_count = {v: len(ess[v]) for v in world.node_ids}
    satisfied = {v: 0 for v in world.node_ids}

    # Initialize with predecessor hull of S
    P: Set[int] = set()
    stack = list(S)
    while stack:
        u = int(stack.pop())
        if u in P:
            continue
        P.add(u)

        # update satisfied counts for nodes for which u is essential
        for v in essential_succs.get(u, []):
            satisfied[v] += 1

        # expand predecessors
        for p in G.predecessors(u):
            p = int(p)
            if p not in P:
                stack.append(p)

    # Iterate forcing until fixed point
    changed = True
    while changed:
        changed = False
        # Find any new forced nodes whose essential predecessors are already in PredCl(P) (=P here)
        newly_forced = [v for v in world.node_ids if v not in P and satisfied[v] >= ess_count[v]]
        if not newly_forced:
            break
        changed = True
        for v in newly_forced:
            # Add v and its predecessor hull into P
            stack = [v]
            while stack:
                u = int(stack.pop())
                if u in P:
                    continue
                P.add(u)
                for s in essential_succs.get(u, []):
                    satisfied[s] += 1
                for p in G.predecessors(u):
                    p = int(p)
                    if p not in P:
                        stack.append(p)

    return P


def _is_connected_induced_undirected(G: nx.MultiDiGraph, P: Set[int]) -> bool:
    """Check connectivity of the induced *undirected* subgraph on P (Definition 5.1(ii))."""
    if not P:
        return False
    if len(P) == 1:
        return True

    UG = nx.Graph()
    UG.add_nodes_from(P)
    for u, v in G.edges():
        u = int(u)
        v = int(v)
        if u in P and v in P:
            UG.add_edge(u, v)

    # If there are isolated vertices, Graph() may still have them; connectivity fails.
    try:
        return nx.is_connected(UG)
    except nx.NetworkXPointlessConcept:
        return False


def pocket_diagnostics(world: WorldInstance, pocket: PocketSpec) -> Dict[str, object]:
    """Return a strict Definition-5.1 diagnostic for a pocket."""
    ess = compute_ess_sets(world)
    P = set(int(x) for x in pocket.pocket_nodes)

    # Closure invariance: Cl(P) == P
    ClP = closure_operator(world, P, ess=ess)
    closure_invariant = (ClP == P)

    connected = _is_connected_induced_undirected(world.graph, P)

    B = set(int(x) for x in pocket.visible_panel)
    hidden = P - B

    return {
        "closure_invariant": bool(closure_invariant),
        "connected": bool(connected),
        "pocket_ok": bool(closure_invariant and connected),
        "pocket_size": int(len(P)),
        "visible_size": int(len(B)),
        "hidden_size": int(len(hidden)),
    }


def make_observer_pocket(
    world: WorldInstance,
    *,
    rng: np.random.Generator,
    seed_size: int,
    max_nodes: int,
    visible_fraction: float,
    max_hidden_vars: int | None = None,
    max_attempts: int | None = None,
) -> PocketSpec:
    """Construct a pocket intended to satisfy Definition 5.1 (observer pocket).

    The critical fix (vs common buggy implementations):
    - NEVER truncate a closure result (BFS cut, random subset, etc.), because that can introduce
      external predecessors and violate closure invariance.
    - Instead, resample seeds to find a *small* closure (Cl(S)).

    Strategy:
    - Sample seed sets S, compute P = Cl(S).
    - Prefer (i) |P| <= max_nodes and (ii) P connected in the induced undirected graph.
    - If no connected pocket within max_nodes is found, fall back to the smallest closure found
      (still closure-invariant, but may not satisfy connectedness).
    """
    ess = compute_ess_sets(world)
    nodes = sorted(int(v) for v in world.node_ids)
    if not nodes:
        raise ValueError("World has no nodes")

    best_any: Optional[Set[int]] = None
    best_any_size = 10**18
    best_connected: Optional[Set[int]] = None
    best_connected_size = 10**18

    # Robust default: allow max_attempts=None (older configs / callers)
    try:
        max_attempts_eff = 500 if max_attempts is None else int(max_attempts)
    except Exception:
        max_attempts_eff = 500
    attempts = max(1, int(max_attempts_eff))

    for _ in range(attempts):
        seeds = set(
            int(x)
            for x in rng.choice(nodes, size=min(int(seed_size), len(nodes)), replace=False).tolist()
        )

        cand = closure_operator(world, seeds, ess=ess)

        # track smallest closure (any)
        if len(cand) < best_any_size:
            best_any = cand
            best_any_size = len(cand)

        # prefer connected candidates
        if _is_connected_induced_undirected(world.graph, cand):
            if len(cand) < best_connected_size:
                best_connected = cand
                best_connected_size = len(cand)

            # accept immediately if within size budget
            if len(cand) <= int(max_nodes):
                P = cand
                break
    else:
        # If loop didn't break, pick best candidate according to priority.
        if best_connected is not None and best_connected_size <= int(max_nodes):
            P = best_connected
        elif best_connected is not None:
            # still connected, but too big: better physics-valid than disconnected
            P = best_connected
        elif best_any is not None:
            P = best_any
        else:
            raise ValueError("Failed to build any pocket (unexpected)")

    P_list = sorted(int(x) for x in P)

    # Visible panel size target
    target_vis = int(round(len(P_list) * float(visible_fraction)))

    # Ensure exact counting is feasible by limiting hidden horizon size
    if max_hidden_vars is not None:
        target_vis = max(target_vis, len(P_list) - int(max_hidden_vars))

    # Avoid trivial hidden horizon if possible (helps SL1 be non-vacuous)
    if len(P_list) > 1:
        target_vis = min(target_vis, len(P_list) - 1)

    vis_count = max(1, min(len(P_list), target_vis))
    visible = set(int(x) for x in rng.choice(P_list, size=vis_count, replace=False).tolist())

    return PocketSpec(pocket_nodes=set(P_list), visible_panel=visible)


def check_observer_pocket(world: WorldInstance, pocket: PocketSpec) -> Dict[str, object]:
    """Compute checkable hypotheses for Theorem 5.1 (Definition 5.1).

    Returns fields suitable for CSV reporting:
      - closure_invariant: Cl(P) == P
      - connected: induced undirected graph on P is connected
      - pocket_ok: closure_invariant and connected
      - pocket_size / visible_size / hidden_size
    """

    P = set(int(x) for x in pocket.pocket_nodes)
    B = set(int(x) for x in pocket.visible_panel)
    H = P - B

    ess = compute_ess_sets(world)
    closure = closure_operator(world, P, ess=ess)
    closure_invariant = closure == P
    connected = _is_connected_induced_undirected(world.graph, P) if len(P) > 0 else False

    return {
        "closure_invariant": bool(closure_invariant),
        "connected": bool(connected),
        "pocket_ok": bool(closure_invariant and connected),
        "pocket_size": int(len(P)),
        "visible_size": int(len(B)),
        "hidden_size": int(len(H)),
    }


def _build_constraints_for_one_step(
    world: WorldInstance,
    P: Set[int],
    B: Set[int],
    x_t: Dict[int, int],
    x_t1: Dict[int, int],
) -> Tuple[List[int], List[Tuple[List[int], Set[Tuple[int, ...]]]]]:
    """Build a #CSP for hidden assignments m in Mt(P) (Definition 5.2).

    Returns:
      H: ordered list of hidden nodes
      constraints: list of (scope_indices, allowed_tuples) where scope_indices indexes into H
    """
    H = sorted(P - B)
    hidden_pos = {h: i for i, h in enumerate(H)}

    constraints: List[Tuple[List[int], Set[Tuple[int, ...]]]] = []

    for b in sorted(B):
        rule = world.rules[b]
        preds = rule.pred_nodes

        # Safety: autonomy requires all rule predecessors are inside P.
        # This is guaranteed if Cl(P)=P, but we keep the check to catch configuration bugs early.
        for p in preds:
            if p not in P:
                raise ValueError("Pocket is not closure-invariant: predecessor outside pocket")

        hidden_vars = [p for p in preds if p in hidden_pos]
        if not hidden_vars:
            # constraint depends only on visible values
            pred_vals = [x_t[p] for p in preds]
            out = int(rule.eval(pred_vals))
            if out != int(x_t1[b]):
                constraints.append(([], set()))  # impossible
            else:
                constraints.append(([], {tuple()}))
            continue

        scope = [hidden_pos[h] for h in hidden_vars]
        allowed: Set[Tuple[int, ...]] = set()

        # Enumerate assignments on just the variables in scope (often small).
        sizes = [len(world.alphabets[h]) for h in hidden_vars]
        for vals in np.ndindex(*sizes):
            assign_map = {hidden_vars[i]: int(vals[i]) for i in range(len(hidden_vars))}
            pred_vals = []
            for p in preds:
                pred_vals.append(assign_map[p] if p in assign_map else int(x_t[p]))
            out = int(rule.eval(pred_vals))
            if out == int(x_t1[b]):
                allowed.add(tuple(int(vals[i]) for i in range(len(hidden_vars))))
        constraints.append((scope, allowed))

    return H, constraints


def count_compatible_hidden_assignments_one_step(
    world: WorldInstance,
    P: Set[int],
    B: Set[int],
    x_t: Dict[int, int],
    x_t1: Dict[int, int],
    *,
    max_hidden_vars: int,
) -> int:
    """Exact count |Mt(P)| via backtracking (Definition 5.2)."""
    H, constraints = _build_constraints_for_one_step(world, P, B, x_t, x_t1)
    n_hidden = len(H)
    if n_hidden == 0:
        # no hidden degrees of freedom
        for scope, allowed in constraints:
            if len(scope) == 0 and len(allowed) == 0:
                return 0
        return 1

    if n_hidden > int(max_hidden_vars):
        raise ValueError(
            f"Hidden horizon too large for exact counting: {n_hidden} > max_hidden_vars={max_hidden_vars}"
        )

    assignment: List[Optional[int]] = [None] * n_hidden

    # Precompute which constraints mention each variable, for incremental checks
    var_to_constraints: List[List[int]] = [[] for _ in range(n_hidden)]
    for ci, (scope, _) in enumerate(constraints):
        for v in scope:
            var_to_constraints[v].append(ci)

    def is_consistent(var_idx: int) -> bool:
        for ci in var_to_constraints[var_idx]:
            scope, allowed = constraints[ci]
            if not allowed:
                return False
            if all(assignment[s] is not None for s in scope):
                tup = tuple(int(assignment[s]) for s in scope)
                if tup not in allowed:
                    return False
        return True

    def backtrack(i: int) -> int:
        if i == n_hidden:
            # All assigned; check any impossible empty-scope constraints
            for scope, allowed in constraints:
                if len(scope) == 0 and len(allowed) == 0:
                    return 0
            return 1

        h_node = H[i]
        total = 0
        for val in range(len(world.alphabets[h_node])):
            assignment[i] = int(val)
            if is_consistent(i):
                total += backtrack(i + 1)
            assignment[i] = None
        return total

    return int(backtrack(0))


def observer_entropy_one_step(
    world: WorldInstance,
    pocket: PocketSpec,
    x_t: Dict[int, int],
    x_t1: Dict[int, int],
    *,
    max_hidden_vars: int,
) -> int:
    """St(P) = ceil(log2 |Mt(P)|) (Definition 5.3)."""
    count = count_compatible_hidden_assignments_one_step(
        world,
        pocket.pocket_nodes,
        pocket.visible_panel,
        x_t,
        x_t1,
        max_hidden_vars=max_hidden_vars,
    )
    if count <= 0:
        return 0
    return int(ceil(log2(count)))


def sl1_check(
    world: WorldInstance,
    pocket: PocketSpec,
    *,
    ticks: int,
    max_hidden_vars: int,
    require_observer_pocket: bool = True,
) -> Dict[str, object]:
    """Empirical SL1 spot-check for a pocket over a small number of ticks.

    If require_observer_pocket=True, we enforce Definition 5.1 before running the check.
    """
    diag = pocket_diagnostics(world, pocket)
    if require_observer_pocket and not bool(diag["pocket_ok"]):
        return {"ok": False, "reason": "invalid_observer_pocket", "diag": diag, "entropies": [], "diffs": []}

    B = set(pocket.visible_panel)

    entropies: List[int] = []
    for t in range(int(ticks) + 1):
        x_t = world.get_state_dict(B)
        world.update(tick=t)
        x_t1 = world.get_state_dict(B)
        St = observer_entropy_one_step(world, pocket, x_t, x_t1, max_hidden_vars=max_hidden_vars)
        entropies.append(St)

    diffs = [entropies[i + 1] - entropies[i] for i in range(len(entropies) - 1)]
    ok = all(d >= 0 for d in diffs)
    return {"ok": bool(ok), "diag": diag, "entropies": entropies, "diffs": diffs}


def degeneracy_bound_check(world: WorldInstance, pocket: PocketSpec, *, require_observer_pocket: bool = True) -> Dict[str, object]:
    """Compute κ0(P) and check Assumption 5.1 (Definitions 5.4-5.5 / Assumption 5.1)."""
    diag = pocket_diagnostics(world, pocket)
    if require_observer_pocket and not bool(diag["pocket_ok"]):
        return {"ok": False, "reason": "invalid_observer_pocket", "diag": diag, "kappa0": 0, "hmin": [], "min_alpha": None}

    P = set(pocket.pocket_nodes)
    B = set(pocket.visible_panel)
    H = P - B

    # depth on condensation DAG (Definition 2.2)
    C, scc_map = condensation_dag(world.graph)
    depths_scc = dag_depths(C)
    depth_node = {v: depths_scc[scc_map[v]] for v in world.node_ids}

    if not H:
        return {"ok": True, "diag": diag, "kappa0": 0, "hmin": [], "min_alpha": None}

    # hidden minimal layer (Definition 5.5)
    min_depth = min(depth_node[h] for h in H)
    hmin = [h for h in H if depth_node[h] == min_depth]

    sigma = compute_sigma(world)

    # κ0(P) = max_{h in Hmin} max_{d in Succ(h)} σ(d)
    kappa0 = 0
    for h in hmin:
        succ = set(int(v) for v in world.graph.successors(h) if int(v) in P)
        for d in succ:
            kappa0 = max(kappa0, int(sigma[d]))

    min_alpha = min(len(world.alphabets[h]) for h in hmin)
    ok = (min_alpha >= kappa0)
    return {"ok": bool(ok), "diag": diag, "kappa0": int(kappa0), "hmin": sorted(hmin), "min_alpha": int(min_alpha)}
