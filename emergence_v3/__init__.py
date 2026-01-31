"""Emergence v3: deterministic finite-state discrete physics engine + universality diagnostics.

Core components used by the PRL universality suite:

- **WorldInstance**: directed substrate + local alphabets + deterministic local rules.
- **SCDC engine**: deterministic quotient construction (schedule-independence / confluence test).
- **ObserverPocket**: small entropy / degeneracy audits (optional for the universality plots).
- **Geometry**: condensation-DAG cone volumes, curvature proxy, memory density,
  and the affine (Einstein–Memory) response with primary coupling **a\***.

See:
- `scripts/run_universality_sweep.py`
- `scripts/run_ba_rewire_stresstest.py`
- `paper/main.pdf`
"""

__all__ = [
    "config",
    "graph_generators",
    "rules",
    "world",
    "partitions",
    "scdc",
    "observer",
    "geometry",
    "nulls",
]
