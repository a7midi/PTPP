from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

from .partitions import Partition, PartitionFamily, partitions_equal
from .rules import LocalRule
from .world import WorldInstance
from .rng import make_rng


def condensation_dag(G: nx.MultiDiGraph) -> tuple[nx.DiGraph, Dict[int, int]]:
    """Return (condensation DAG, node->scc_id)."""
    # networkx condensation works on DiGraph; convert ignoring multiplicity
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes())
    DG.add_edges_from((u, v) for u, v in G.edges())
    # Determinism guard:
    # nx.strongly_connected_components yields SCCs in an order that can depend on
    # internal traversal details. To make the SCC labeling reproducible across
    # Python processes (hash randomization) and platforms, we impose a stable order
    # by sorting SCCs by their minimum original node id.
    sccs = list(nx.strongly_connected_components(DG))
    sccs.sort(key=lambda comp: min(int(v) for v in comp) if comp else -1)
    mapping: Dict[int, int] = {}
    for i, comp in enumerate(sccs):
        for v in comp:
            mapping[int(v)] = i
    C = nx.condensation(DG, scc=sccs)
    return C, mapping


def dag_depths(D: nx.DiGraph) -> Dict[int, int]:
    """depth(x) = length of longest directed path ending at x."""
    if len(D) == 0:
        return {}
    depths: Dict[int, int] = {int(n): 0 for n in D.nodes()}
    for n in nx.topological_sort(D):
        dn = depths[int(n)]
        for succ in D.successors(n):
            succ = int(succ)
            depths[succ] = max(depths[succ], dn + 1)
    return depths


def find_vertex_diamonds(
    G: nx.MultiDiGraph,
    *,
    depth_by_node: Dict[int, int],
    max_diamonds: int,
    sampling: str,
    rng: np.random.Generator,
) -> List[Tuple[int, int, int, int]]:
    """Find diamonds (x,y,z,w) in the *vertex graph*, using a depth assignment.

    Condition (lifted from condensation DAG definition):
      x->y, x->z, y->w, z->w, y!=z, depth(y)==depth(z).

    For performance on large graphs, we cap the result via max_diamonds (configurable).
    """
    diamonds: List[Tuple[int, int, int, int]] = []
    nodes = list(G.nodes())
    nodes.sort()

    # Precompute successors as sets (unique) for intersection
    succ: Dict[int, set[int]] = {int(u): set(int(v) for v in G.successors(u)) for u in nodes}

    for x in nodes:
        outs = list(succ[x])
        if len(outs) < 2:
            continue
        # for each pair y,z among out-neighbors, look for common successors w
        # degree is small in our bundles (~3), so O(d^2) is cheap.
        for i in range(len(outs)):
            y = outs[i]
            for j in range(i + 1, len(outs)):
                z = outs[j]
                if y == z:
                    continue
                if depth_by_node.get(y, 0) != depth_by_node.get(z, 0):
                    continue
                common_w = succ.get(y, set()) & succ.get(z, set())
                for w in common_w:
                    diamonds.append((x, y, z, w))
                    if len(diamonds) >= max_diamonds and sampling == "first":
                        return diamonds
    if len(diamonds) <= max_diamonds:
        return diamonds
    if sampling == "random":
        idx = rng.choice(len(diamonds), size=max_diamonds, replace=False)
        return [diamonds[int(i)] for i in idx]
    return diamonds[:max_diamonds]


@dataclass
class SCDCResult:
    partitions: PartitionFamily
    iters: int
    num_diamonds_used: int


class SCDC_Engine:
    """Self-Consistent Dynamical Constraint (SCDC) iteration (Section 3)."""

    def __init__(
        self,
        world: WorldInstance,
        *,
        max_iters: int,
        star_internal_max_iters: int,
        diamond_internal_max_iters: int,
        diamond_enabled: bool,
        diamond_max_diamonds: int,
        diamond_sampling: str,
        diamond_max_assignments_per_diamond: int,
        diamond_seed: int,
    ):
        self.world = world
        self.max_iters = int(max_iters)
        self.star_internal_max_iters = int(star_internal_max_iters)
        self.diamond_internal_max_iters = int(diamond_internal_max_iters)
        self.diamond_enabled = bool(diamond_enabled)
        self.diamond_max_diamonds = int(diamond_max_diamonds)
        self.diamond_sampling = str(diamond_sampling)
        self.diamond_max_assignments_per_diamond = int(diamond_max_assignments_per_diamond)
        self._rng = make_rng(int(diamond_seed))

        # Geometry-derived depth labels used for diamond filtering
        C, scc_map = condensation_dag(world.graph)
        depths_scc = dag_depths(C)
        depth_by_node = {v: depths_scc[scc_map[v]] for v in world.node_ids}
        self.depth_by_node = depth_by_node

        self.diamonds: List[Tuple[int, int, int, int]] = []
        if self.diamond_enabled and self.diamond_max_diamonds > 0:
            self.diamonds = find_vertex_diamonds(
                world.graph,
                depth_by_node=depth_by_node,
                max_diamonds=self.diamond_max_diamonds,
                sampling=self.diamond_sampling,
                rng=self._rng,
            )

        # Cache for input tuple enumeration by table shape
        self._tuple_cache: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    def _tuples_for_shape(self, shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        if shape not in self._tuple_cache:
            if len(shape) == 0:
                self._tuple_cache[shape] = [tuple()]
            else:
                self._tuple_cache[shape] = list(np.ndindex(*shape))
        return self._tuple_cache[shape]

    def _enforce_star_on_node(self, v: int, parts: PartitionFamily) -> Partition:
        part_v = parts[v]
        rule = self.world.rules[v]
        preds = rule.pred_nodes
        if len(preds) == 0:
            return part_v

        # Build unions required for star equivariance
        unions: List[Tuple[int, int]] = []
        seen_key_to_rep_output: Dict[int, int] = {}

        # mixed radix multipliers for class keys
        radices = [parts[p].num_classes for p in preds]
        mults: List[int] = []
        m = 1
        for r in radices:
            mults.append(m)
            m *= int(r)

        for idx_tuple in self._tuples_for_shape(rule.table.shape):
            key = 0
            for axis, p in enumerate(preds):
                cls = int(parts[p].class_of[idx_tuple[axis]])
                key += cls * mults[axis]
            out = int(rule.table[idx_tuple])
            if key in seen_key_to_rep_output:
                rep_out = seen_key_to_rep_output[key]
                # if outputs land in different equivalence classes, merge them
                if int(part_v.class_of[rep_out]) != int(part_v.class_of[out]):
                    unions.append((rep_out, out))
            else:
                seen_key_to_rep_output[key] = out

        if not unions:
            return part_v
        return part_v.with_unions(unions)

    def star_closure(self, parts_in: PartitionFamily) -> PartitionFamily:
        parts = {v: parts_in[v] for v in parts_in.keys()}
        for _ in range(self.star_internal_max_iters):
            changed = False
            for v in self.world.node_ids:
                new_p = self._enforce_star_on_node(v, parts)
                if new_p != parts[v]:
                    parts[v] = new_p
                    changed = True
            if not changed:
                break
        return parts

    def _enforce_diamond_on_one(self, diamond: Tuple[int, int, int, int], parts: PartitionFamily) -> List[Tuple[int, int, int]]:
        """Return unions for the w-alphabet implied by this diamond.

        We implement a conservative schedule-independence constraint:
            w_y := λ_w( y := λ_y(...), z := z_t, others := t)
            w_z := λ_w( y := y_t, z := λ_z(...), others := t)
        Require w_y and w_z to be equivalent (merge outputs if needed).

        This is a pragmatic implementation consistent with the intent of 'diamond-coherence'
        as an order-independence condition on parallel updates.
        """
        x, y, z, w = diamond
        rule_y = self.world.rules[y]
        rule_z = self.world.rules[z]
        rule_w = self.world.rules[w]

        # variables needed for evaluating the above expressions at tick t
        vars_set = set(rule_y.pred_nodes) | set(rule_z.pred_nodes) | set(rule_w.pred_nodes) | {y, z}
        vars_list = sorted(vars_set)

        # Domain sizes from current (possibly coarsened) alphabets? Diamond coherence is defined
        # on induced quotient updates, but our closure operates by merging outputs; we conservatively
        # enumerate *symbol indices* in the current alphabet sizes.
        dom_sizes = {u: parts[u].size for u in vars_list}

        # Quick bailout: if any alphabet is size 0 (should not happen)
        if any(dom_sizes[u] <= 0 for u in vars_list):
            return []

        # Guard on brute-force enumeration size (configurable)
        total_assignments = 1
        for u in vars_list:
            total_assignments *= int(dom_sizes[u])
            if total_assignments > self.diamond_max_assignments_per_diamond:
                return []

        # Precompute predecessor lists for indexing
        pred_y = rule_y.pred_nodes
        pred_z = rule_z.pred_nodes
        pred_w = rule_w.pred_nodes

        # Map node -> position in assignment vector
        pos = {u: i for i, u in enumerate(vars_list)}

        # Enumerate all assignments (exact, small domains only). This is guarded by diamond sampling.
        # If domains are large, this can blow up; we therefore cap runtime by only sampling diamonds.
        unions: List[Tuple[int, int, int]] = []  # (w, out1, out2)
        for flat in np.ndindex(*(dom_sizes[u] for u in vars_list)):
            assign = flat  # tuple of indices per var

            # compute updated y and z values
            y_inputs = [assign[pos[p]] for p in pred_y]
            z_inputs = [assign[pos[p]] for p in pred_z]
            y1 = int(rule_y.eval(y_inputs))
            z1 = int(rule_z.eval(z_inputs))

            # w inputs use tick-t values except we replace y or z depending on branch
            w_inputs_base = [assign[pos[p]] for p in pred_w]

            # helper to replace one coordinate if present
            def replaced(inputs: List[int], node: int, new_val: int) -> List[int]:
                if node not in pred_w:
                    return inputs
                out = list(inputs)
                for i, p in enumerate(pred_w):
                    if p == node:
                        out[i] = new_val
                return out

            w_in_y = replaced(w_inputs_base, y, y1)
            w_in_z = replaced(w_inputs_base, z, z1)

            out_y = int(rule_w.eval(w_in_y))
            out_z = int(rule_w.eval(w_in_z))

            if int(parts[w].class_of[out_y]) != int(parts[w].class_of[out_z]):
                unions.append((w, out_y, out_z))

        return unions

    def diamond_closure(self, parts_in: PartitionFamily) -> PartitionFamily:
        if not self.diamond_enabled or not self.diamonds:
            return {v: parts_in[v] for v in parts_in.keys()}

        parts = {v: parts_in[v] for v in parts_in.keys()}
        for _ in range(self.diamond_internal_max_iters):
            changed = False
            unions_by_w: Dict[int, List[Tuple[int, int]]] = {}
            for d in self.diamonds:
                unions = self._enforce_diamond_on_one(d, parts)
                for w, a, b in unions:
                    unions_by_w.setdefault(w, []).append((a, b))

            for w, unions in unions_by_w.items():
                new_p = parts[w].with_unions(unions)
                if new_p != parts[w]:
                    parts[w] = new_p
                    changed = True

            if not changed:
                break
        return parts

    def run(self) -> SCDCResult:
        # Start from discrete partition (Definition 3.3 / Theorem 3.3)
        parts: PartitionFamily = {
            v: Partition.discrete(len(self.world.alphabets[v])) for v in self.world.node_ids
        }

        for it in range(self.max_iters):
            prev = parts
            parts = self.star_closure(parts)
            parts = self.diamond_closure(parts)
            if partitions_equal(parts, prev):
                return SCDCResult(partitions=parts, iters=it + 1, num_diamonds_used=len(self.diamonds))

        return SCDCResult(partitions=parts, iters=self.max_iters, num_diamonds_used=len(self.diamonds))


def build_quotient_world(world: WorldInstance, parts: PartitionFamily) -> WorldInstance:
    """Construct the quotient world instance induced by Λ⋆ partitions (Definition 3.2)."""
    # New alphabets are classes 0..C-1
    new_alphabets: Dict[int, List[int]] = {v: list(range(parts[v].num_classes)) for v in world.node_ids}

    # Build representative symbol index in the original alphabet for each class
    reps: Dict[int, Dict[int, int]] = {}
    for v in world.node_ids:
        rep: Dict[int, int] = {}
        for idx, cls in enumerate(parts[v].class_of.tolist()):
            if cls not in rep:
                rep[cls] = idx
        reps[v] = rep

    # Build new rules as lookup tables over predecessor *classes*.
    new_rules: Dict[int, LocalRule] = {}
    for v in world.node_ids:
        rule = world.rules[v]
        pred_nodes = rule.pred_nodes

        pred_class_sizes = [parts[p].num_classes for p in pred_nodes]
        out_classes = parts[v].num_classes

        if len(pred_nodes) == 0:
            out_idx = int(parts[v].class_of[int(rule.table.reshape(-1)[0])])
            table_q = np.array([out_idx], dtype=np.int16)
            new_rules[v] = LocalRule(node=v, pred_nodes=pred_nodes, table=table_q)
            continue

        table_q = np.empty(tuple(pred_class_sizes), dtype=np.int16)

        for cls_tuple in np.ndindex(*tuple(pred_class_sizes)):
            # pick representatives in original alphabets
            orig_inputs = []
            for axis, p in enumerate(pred_nodes):
                cls = int(cls_tuple[axis])
                orig_inputs.append(int(reps[p][cls]))
            out_orig = int(rule.eval(orig_inputs))
            out_cls = int(parts[v].class_of[out_orig])
            table_q[cls_tuple] = out_cls

        new_rules[v] = LocalRule(node=v, pred_nodes=pred_nodes, table=table_q)

    # Initial state: map each node's original symbol index to class id
    new_state = np.empty_like(world.state)
    for v in world.node_ids:
        i = world.node_index[v]
        new_state[i] = int(parts[v].class_of[int(world.state[i])])

    return WorldInstance.build(world.graph, new_alphabets, new_rules, initial_state=new_state)
