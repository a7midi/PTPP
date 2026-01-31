from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

from .rules import LocalRule


@dataclass
class WorldInstance:
    """Finite deterministic world instance (Definition 2.1).

    State uses *indices* into each node's local alphabet (0..|A_v|-1). Values are recoverable via
    alphabets[node][idx] if needed.

    Updates are synchronous by default (one global tick applies all λ_v simultaneously).
    """

    graph: nx.MultiDiGraph
    alphabets: Dict[int, List[int]]  # node -> list of values (finite)
    rules: Dict[int, LocalRule]      # node -> local deterministic update
    state: np.ndarray               # shape (N,), dtype int, per-node alphabet index

    node_ids: List[int]
    node_index: Dict[int, int]
    predecessors: Dict[int, List[int]]
    predecessor_indices: Dict[int, np.ndarray]

    @classmethod
    def build(
        cls,
        graph: nx.MultiDiGraph,
        alphabets: Dict[int, List[int]],
        rules: Dict[int, LocalRule],
        *,
        initial_state: Optional[np.ndarray] = None,
    ) -> "WorldInstance":
        node_ids = list(graph.nodes())
        node_ids.sort()
        node_index = {v: i for i, v in enumerate(node_ids)}

        predecessors: Dict[int, List[int]] = {}
        predecessor_indices: Dict[int, np.ndarray] = {}

        for v in node_ids:
            # Graph predecessors (for validation only)
            graph_preds = sorted(set(int(p) for p in graph.predecessors(v)))

            # Rule predecessors (the actual inputs λ_v depends on)
            rule_preds = list(rules[v].pred_nodes)

            # Safety check: rule inputs must be real predecessors in the graph
            # Note: We convert to sets for subset check, but rule_preds may be a subset of graph_preds
            if not set(rule_preds).issubset(set(graph_preds)):
                raise ValueError(f"Rule pred_nodes for v={v} are not a subset of graph predecessors")

            predecessors[v] = rule_preds
            predecessor_indices[v] = np.array([node_index[p] for p in rule_preds], dtype=np.int32)

        N = len(node_ids)
        if initial_state is None:
            # default state 0 for all nodes
            initial_state = np.zeros((N,), dtype=np.int16)
        else:
            initial_state = np.asarray(initial_state, dtype=np.int16)
            if initial_state.shape != (N,):
                raise ValueError(f"initial_state must have shape ({N},)")

        return cls(
            graph=graph,
            alphabets=alphabets,
            rules=rules,
            state=initial_state,
            node_ids=node_ids,
            node_index=node_index,
            predecessors=predecessors,
            predecessor_indices=predecessor_indices,
        )

    @property
    def N(self) -> int:
        return len(self.node_ids)

    def copy(self) -> "WorldInstance":
        return WorldInstance.build(
            graph=self.graph.copy(),
            alphabets={k: list(v) for k, v in self.alphabets.items()},
            rules=self.rules,  # LocalRule is immutable
            initial_state=self.state.copy(),
        )

    def randomize_state(self, rng: np.random.Generator) -> None:
        for v in self.node_ids:
            i = self.node_index[v]
            self.state[i] = int(rng.integers(0, len(self.alphabets[v])))

    def update(self, tick: int | None = None) -> np.ndarray:
        """Apply λ_v to all nodes synchronously, producing a^(t+1)."""
        new_state = np.empty_like(self.state)
        # local variables for speed
        state = self.state
        for v in self.node_ids:
            rule = self.rules[v]
            idx = self.node_index[v]
            pred_idx = self.predecessor_indices[v]
            # Use rule.eval consistently with truncated pred_nodes
            pred_vals = state[pred_idx] if pred_idx.size > 0 else []
            new_state[idx] = int(rule.eval(pred_vals))
            
        self.state = new_state
        return new_state

    def get_state_dict(self, nodes: Sequence[int]) -> Dict[int, int]:
        """Get current state as a dict {node: alphabet_index}."""
        return {v: int(self.state[self.node_index[v]]) for v in nodes}

    def set_state_dict(self, state_map: Dict[int, int]) -> None:
        for v, val in state_map.items():
            self.state[self.node_index[v]] = int(val)