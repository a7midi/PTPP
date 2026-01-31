from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .world import WorldInstance
from .observer import predecessor_hull


@dataclass(frozen=True)
class BellExperiment:
    """Minimal Bell/CHSH sector specification.

    All nodes are integer IDs in the world's graph.
    Settings are read at tick t; outcomes and herald are read at tick t+1 by default.
    """
    source: int
    alice_setting: int
    bob_setting: int
    alice_outcome: int
    bob_outcome: int
    herald: int

    # Optional sector expansion
    pred_depth: int = 2  # include predecessors up to this depth for all readout nodes
    max_vars: int = 20   # guard on exact enumeration

    # Interpretation for CHSH: outcomes and settings must be binary {0,1} indices.
    # outcome_sign maps outcome index -> ±1
    outcome_sign: Tuple[int, int] = (-1, +1)

    def sector_nodes(self, world: WorldInstance) -> Set[int]:
        seeds = {
            int(self.source),
            int(self.alice_setting),
            int(self.bob_setting),
            int(self.alice_outcome),
            int(self.bob_outcome),
            int(self.herald),
        }
        S = set(seeds)
        for _ in range(int(self.pred_depth)):
            S = predecessor_hull(world.graph, S)
        return set(int(x) for x in S)


@dataclass(frozen=True)
class BellAuditResult:
    total_histories: int
    accepted_histories: int
    acceptance_rate: float
    chsh: float
    no_signaling_residual: float


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def run_bell_audit(
    world: WorldInstance,
    exp: BellExperiment,
    *,
    global_constraint: Optional[Callable[[Dict[int, int], Dict[int, int]], bool]] = None,
) -> BellAuditResult:
    """Enumerate histories in a Bell sector and compute CHSH on heralded subensemble.

    This implements the paper's operational stance: we do not simulate a quantum state. We generate
    classical histories and condition on a herald predicate (Section 6 / 6.2).
    """
    if global_constraint is None:
        global_constraint = lambda x_t, x_t1: True

    sector = sorted(exp.sector_nodes(world))
    if len(sector) > exp.max_vars:
        raise ValueError(f"Bell sector too large: {len(sector)} > max_vars={exp.max_vars}")

    # Domain sizes
    dom_sizes = [len(world.alphabets[v]) for v in sector]
    if any(s < 2 for s in dom_sizes):
        # CHSH requires binary settings/outcomes. We'll still run but CHSH will be NaN.
        pass

    # Identify indices of key nodes within sector list
    pos = {v: i for i, v in enumerate(sector)}
    required = [exp.alice_setting, exp.bob_setting, exp.alice_outcome, exp.bob_outcome, exp.herald]
    for v in required:
        if v not in pos:
            raise ValueError("Bell sector does not include required node: " + str(v))

    # Precompute predecessor positions for fast evaluation of t+1 updates on sector nodes.
    # We evaluate one synchronous tick but only need outputs for outcomes and herald.
    def eval_node_at_t1(node: int, assignment_t: Tuple[int, ...]) -> int:
        rule = world.rules[node]
        preds = rule.pred_nodes
        pred_vals = []
        for p in preds:
            if p in pos:
                pred_vals.append(int(assignment_t[pos[p]]))
            else:
                # treat outside-sector predecessors as fixed at their current world state
                pred_vals.append(int(world.state[world.node_index[p]]))
        return int(rule.eval(pred_vals))

    total = 0
    accepted = 0

    # counts keyed by (x,y) for CHSH
    # E_xy = E[A*B | x,y]
    counts_xy = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}

    # For no-signaling residuals, we need empirical P(A|x,y) and P(B|x,y)
    # We'll store counts for A and B outcomes (binary).
    A_counts: Dict[Tuple[int, int], np.ndarray] = {(x, y): np.zeros(2, dtype=np.int64) for x in (0, 1) for y in (0, 1)}
    B_counts: Dict[Tuple[int, int], np.ndarray] = {(x, y): np.zeros(2, dtype=np.int64) for x in (0, 1) for y in (0, 1)}

    for assign in np.ndindex(*dom_sizes):
        total += 1
        x_t = {sector[i]: int(assign[i]) for i in range(len(sector))}

        # one-tick update for key readouts
        a_out = eval_node_at_t1(exp.alice_outcome, assign)
        b_out = eval_node_at_t1(exp.bob_outcome, assign)
        h_out = eval_node_at_t1(exp.herald, assign)

        x_t1 = {
            exp.alice_outcome: a_out,
            exp.bob_outcome: b_out,
            exp.herald: h_out,
        }

        if not global_constraint(x_t, x_t1):
            continue
        if h_out != 1:
            continue

        accepted += 1
        x = int(x_t[exp.alice_setting])
        y = int(x_t[exp.bob_setting])
        if x not in (0, 1) or y not in (0, 1) or a_out not in (0, 1) or b_out not in (0, 1):
            continue

        A = exp.outcome_sign[a_out]
        B = exp.outcome_sign[b_out]
        counts_xy[(x, y)].append(A * B)
        A_counts[(x, y)][a_out] += 1
        B_counts[(x, y)][b_out] += 1

    if total == 0:
        return BellAuditResult(0, 0, float("nan"), float("nan"), float("nan"))

    acceptance_rate = accepted / total

    # CHSH
    E = {}
    for x in (0, 1):
        for y in (0, 1):
            vals = counts_xy[(x, y)]
            E[(x, y)] = float(np.mean(vals)) if vals else 0.0

    chsh = E[(0, 0)] + E[(0, 1)] + E[(1, 0)] - E[(1, 1)]

    # No-signaling residual: max_y ||P(A|x,y)-P(A|x,1-y)||_1 and similar for B
    def norm_counts(c: np.ndarray) -> np.ndarray:
        s = float(c.sum())
        return (c / s) if s > 0 else np.array([0.5, 0.5], dtype=float)

    ns = 0.0
    for x in (0, 1):
        pA0 = norm_counts(A_counts[(x, 0)])
        pA1 = norm_counts(A_counts[(x, 1)])
        ns = max(ns, _l1(pA0, pA1))
    for y in (0, 1):
        pB0 = norm_counts(B_counts[(0, y)])
        pB1 = norm_counts(B_counts[(1, y)])
        ns = max(ns, _l1(pB0, pB1))

    return BellAuditResult(
        total_histories=total,
        accepted_histories=accepted,
        acceptance_rate=float(acceptance_rate),
        chsh=float(chsh),
        no_signaling_residual=float(ns),
    )
