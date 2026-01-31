from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

from .config import MeshDesign
from .sim import SimResult


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).copy()
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    # Gini = (n+1 - 2*sum_i cum_i / cum_n) / n
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def shannon_entropy(x: np.ndarray, eps: float = 1e-15) -> float:
    x = np.asarray(x, dtype=float)
    s = float(x.sum())
    if s <= 0:
        return 0.0
    p = x / s
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def hill_tail_index(x: np.ndarray, k: int = 5) -> Optional[float]:
    """Hill estimator for Pareto tail index alpha using top-k order statistics.

    Returns alpha (larger = thinner tail). None if insufficient positive samples.
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if x.size < k + 1:
        return None
    x = np.sort(x)
    top = x[-k:]
    xk1 = x[-k-1]
    if xk1 <= 0:
        return None
    return float(k / np.sum(np.log(top) - np.log(xk1)))


def active_counts(design: MeshDesign, sim: SimResult) -> Dict[str, Any]:
    thr = float(design.power_threshold)
    if sim.steady_P is not None:
        P = sim.steady_P
        active = (P > thr)
        per_stage = active.sum(axis=1).astype(int)
        total = int(active.sum())
    else:
        P = sim.pulse_P
        assert P is not None
        E = P.sum(axis=2)  # integrate time bins
        active = (E > thr)
        per_stage = active.sum(axis=1).astype(int)
        total = int(active.sum())
    return {"per_stage": per_stage.tolist(), "total": total}


def transport_regime(total_active: int, dead_lt: int = 50, shock_gt: int = 300) -> str:
    """Heuristic regime classifier on total active nodes.

    These thresholds are *design-time defaults* for L~50, W~20 (max nodes ~1000).
    You should re-tune based on your design and calibration.
    """
    if total_active < dead_lt:
        return "dead"
    if total_active > shock_gt:
        return "shockwave"
    return "localized"


def summarize(design: MeshDesign, sim: SimResult) -> Dict[str, Any]:
    y = np.asarray(sim.outputs_energy, dtype=float)
    counts = active_counts(design, sim)
    total = int(counts["total"])
    reg = transport_regime(total_active=total)
    return {
        "inject_channel": int(sim.inject_channel),
        "mode": str(sim.meta.get("mode")),
        "outputs": {
            "sum": float(y.sum()),
            "max": float(y.max()),
            "median": float(np.median(y)),
            "gini": gini(y),
            "entropy": shannon_entropy(y),
            "focus_gain": float(y.max() / (np.median(y) + 1e-15)),
            "hill_alpha_k5": hill_tail_index(y, k=5),
        },
        "active": counts,
        "regime": reg,
    }
