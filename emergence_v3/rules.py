from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalRule:
    """Deterministic local update map λ_v as a lookup table.

    The rule table is indexed by predecessor-state indices (not values).
    Output is an index into the local alphabet A_v (post-quotient or pre-quotient).
    """

    node: int
    pred_nodes: List[int]
    table: np.ndarray  # dtype int, shape = (|A_pred1|, |A_pred2|, ..., )

    def eval(self, pred_state_indices: Sequence[int]) -> int:
        if len(self.pred_nodes) == 0:
            # scalar table
            return int(self.table.reshape(-1)[0])
        return int(self.table[tuple(int(x) for x in pred_state_indices)])

    @property
    def pred_sizes(self) -> List[int]:
        return list(self.table.shape)

    @property
    def out_size(self) -> int:
        return int(self.table.max()) + 1 if self.table.size else 0


def generate_random_lookup_rules(
    *,
    graph_nodes: Sequence[int],
    predecessors: Dict[int, List[int]],
    alphabet_sizes: Dict[int, int],
    rng: np.random.Generator,
    max_arity: int | None = None,
    arity_sampling: str = "random",
) -> Dict[int, LocalRule]:
    """Generate per-node random deterministic rules as uniform random lookup tables.

    IMPORTANT: For hubs (e.g., Barabasi-Albert), full truth tables are exponential in indegree.
    We cap the effective rule arity by selecting at most `max_arity` predecessor inputs.
    Unselected predecessors are implicitly non-essential (the rule does not depend on them).
    """
    rules: Dict[int, LocalRule] = {}
    for v in graph_nodes:
        pred_nodes = list(predecessors[v])

        # Cap arity to avoid exponential blow-up
        if max_arity is not None and len(pred_nodes) > int(max_arity):
            k = int(max_arity)
            if arity_sampling == "random":
                pred_nodes = sorted(int(x) for x in rng.choice(pred_nodes, size=k, replace=False).tolist())
            elif arity_sampling == "first":
                pred_nodes = pred_nodes[:k]
            else:
                raise ValueError(f"Unknown arity_sampling={arity_sampling!r} (use 'random' or 'first')")

        shape = tuple(int(alphabet_sizes[p]) for p in pred_nodes)
        out_size = int(alphabet_sizes[v])

        if len(shape) == 0:
            table = rng.integers(0, out_size, size=(1,), dtype=np.int16)
        else:
            table = rng.integers(0, out_size, size=shape, dtype=np.int16)

        rules[v] = LocalRule(node=v, pred_nodes=pred_nodes, table=table)
    return rules


def essential_predecessors(rule: LocalRule) -> List[int]:
    """Compute Ess(v) (Definition 4.1) from a truth table."""
    preds = rule.pred_nodes
    if len(preds) == 0:
        return []

    tab = rule.table
    ess: List[int] = []
    for axis, pred_node in enumerate(preds):
        # Move axis to front -> shape (size_axis, -1)
        moved = np.moveaxis(tab, axis, 0).reshape(tab.shape[axis], -1)
        # If any column has variation, the predecessor is essential
        col_min = moved.min(axis=0)
        col_max = moved.max(axis=0)
        if np.any(col_min != col_max):
            ess.append(pred_node)
    return ess


def single_coordinate_degeneracy(rule: LocalRule, *, output_size: int) -> int:
    """Compute σ(d) (Definition 5.4) for a node d given its rule truth table."""
    preds = rule.pred_nodes
    if len(preds) == 0:
        return 1

    tab = rule.table
    sigma = 1
    for axis in range(len(preds)):
        moved = np.moveaxis(tab, axis, 0).reshape(tab.shape[axis], -1)  # (size_axis, ncols)
        # For each column (fixed other coords), count max multiplicity per output value
        # output_size is typically small (e.g., 2), so a small loop is ok.
        for col in range(moved.shape[1]):
            vals = moved[:, col]
            # multiplicity of the most common output
            counts = np.bincount(vals.astype(np.int64), minlength=output_size)
            sigma = max(sigma, int(counts.max()))
    return sigma