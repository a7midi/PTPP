from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np


SeedLike = Union[int, str]


def _stable_bytes(items: Iterable[SeedLike]) -> bytes:
    parts: list[bytes] = []
    for it in items:
        if isinstance(it, int):
            parts.append(str(it).encode("utf-8"))
        else:
            parts.append(it.encode("utf-8"))
        parts.append(b"|")
    return b"".join(parts)


def derive_seed(base_seed: int, *components: SeedLike, modulo: int = 2**32 - 1) -> int:
    """Derive a deterministic 32-bit-ish seed from a base seed and arbitrary components."""
    h = hashlib.blake2b(digest_size=8)
    h.update(_stable_bytes((base_seed, *components)))
    digest = int.from_bytes(h.digest(), "big", signed=False)
    return int(digest % modulo)


def make_rng(seed: int) -> np.random.Generator:
    """Create a NumPy Generator with a deterministic seed."""
    return np.random.default_rng(int(seed))
