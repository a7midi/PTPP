from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import networkx as nx

from .config import MeshDesign


@dataclass
class SimResult:
    inject_channel: int
    steady_P: Optional[np.ndarray]  # shape (L+1, W)
    pulse_P: Optional[np.ndarray]   # shape (L+1, W, T)
    outputs_energy: np.ndarray      # shape (W,)
    meta: Dict[str, Any]


def simulate_steady(design: MeshDesign, G: nx.DiGraph, inject_channel: int, power_in: float = 1.0) -> SimResult:
    """Stage-by-stage steady (single-pulse, no time bins) intensity propagation."""
    L, W = design.L, design.W
    P = np.zeros((L+1, W), dtype=np.float64)
    P[0, inject_channel] = float(power_in)

    # propagate
    for s in range(L):
        for c in range(W):
            p = P[s, c]
            if p == 0.0:
                continue
            u = (s, c)
            for v in G.successors(u):
                d = G.edges[u, v]
                w = float(d.get("w", 0.0))
                loss_dB = float(d.get("loss_dB", d.get("il_dB", 0.0)))
                loss_lin = 10.0 ** (-loss_dB / 10.0)
                (sv, cv) = v
                P[sv, cv] += p * w * loss_lin

    outputs = P[L, :].copy()
    return SimResult(
        inject_channel=inject_channel,
        steady_P=P,
        pulse_P=None,
        outputs_energy=outputs,
        meta={"mode": "steady", "power_in": float(power_in)}
    )


def simulate_pulsed(design: MeshDesign, G: nx.DiGraph, inject_channel: int,
                    power_in: float = 1.0, jitter_ps: float = 0.0, T: Optional[int] = None) -> SimResult:
    """Pulsed propagation with edge delays mapped to discrete time bins.

    The model is incoherent: we propagate *power* waveforms.
    Each edge u->v shifts the waveform by delay(u,v)/dt bins.

    Notes:
    - This is a coarse discrete-time approximation.
    - In a real PIC, dispersion and bandwidth matter; treat this as a design-time estimator.
    """
    L, W = design.L, design.W
    if T is None:
        T = int(design.max_time_bins)
    dt_ps = float(design.dt_ps)

    P = np.zeros((L+1, W, T), dtype=np.float64)

    # injection as a single-bin pulse (delta)
    t0 = int(round(float(jitter_ps) / dt_ps))
    t0 = max(0, min(T-1, t0))
    P[0, inject_channel, t0] = float(power_in)

    for s in range(L):
        for c in range(W):
            u = (s, c)
            x = P[s, c, :]
            if not np.any(x):
                continue
            for v in G.successors(u):
                d = G.edges[u, v]
                w = float(d.get("w", 0.0))
                delay_ps = float(d.get("delay_ps", 0.0))
                shift = int(round(delay_ps / dt_ps))
                loss_dB = float(d.get("loss_dB", d.get("il_dB", 0.0)))
                loss_lin = 10.0 ** (-loss_dB / 10.0)
                y = x * (w * loss_lin)
                if shift != 0:
                    # shift right with zero padding
                    if shift > 0:
                        y2 = np.zeros_like(y)
                        if shift < T:
                            y2[shift:] = y[:T-shift]
                        y = y2
                    else:
                        # negative shift: advance
                        sh = -shift
                        y2 = np.zeros_like(y)
                        if sh < T:
                            y2[:T-sh] = y[sh:]
                        y = y2
                (sv, cv) = v
                P[sv, cv, :] += y

    outputs_energy = P[L, :, :].sum(axis=1)
    return SimResult(
        inject_channel=inject_channel,
        steady_P=None,
        pulse_P=P,
        outputs_energy=outputs_energy,
        meta={"mode": "pulsed", "power_in": float(power_in), "jitter_ps": float(jitter_ps), "dt_ps": dt_ps, "T": T}
    )
