from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        # path compression
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(int(p))
        return int(self.parent[x])

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass(frozen=True)
class Partition:
    """Equivalence relation on a finite alphabet via class IDs for each symbol."""
    class_of: np.ndarray  # shape (K,), dtype int, class id per symbol index

    @staticmethod
    def discrete(size: int) -> "Partition":
        return Partition(class_of=np.arange(int(size), dtype=np.int16))

    @property
    def size(self) -> int:
        return int(self.class_of.shape[0])

    @property
    def num_classes(self) -> int:
        return int(np.unique(self.class_of).size)

    def canonicalize(self) -> "Partition":
        # remap class ids to 0..C-1 in order of first appearance
        mapping: Dict[int, int] = {}
        next_id = 0
        new = np.empty_like(self.class_of)
        for i, c in enumerate(self.class_of.tolist()):
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new[i] = mapping[c]
        return Partition(class_of=new)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partition):
            return False
        if self.class_of.shape != other.class_of.shape:
            return False
        return bool(np.array_equal(self.class_of, other.class_of))

    def with_unions(self, unions: Iterable[Tuple[int, int]]) -> "Partition":
        """Return the least coarsening obtained by unioning the given symbol-index pairs."""
        uf = UnionFind(self.size)
        # seed with existing equivalences
        rep: Dict[int, int] = {}
        for i, c in enumerate(self.class_of.tolist()):
            if c in rep:
                uf.union(i, rep[c])
            else:
                rep[c] = i
        # apply new unions
        for a, b in unions:
            uf.union(int(a), int(b))
        roots = np.array([uf.find(i) for i in range(self.size)], dtype=np.int32)
        return Partition(class_of=roots).canonicalize()


PartitionFamily = Dict[int, Partition]


def partitions_equal(a: PartitionFamily, b: PartitionFamily) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a.keys():
        if a[k] != b[k]:
            return False
    return True
