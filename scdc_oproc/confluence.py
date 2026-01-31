from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any
import numpy as np
import networkx as nx


@dataclass
class Diamond:
    x: Tuple[int, int]
    y: Tuple[int, int]
    z: Tuple[int, int]
    w: Tuple[int, int]


def list_diamonds(G: nx.DiGraph, max_count: int = 200000) -> List[Diamond]:
    """Enumerate directed diamonds x->y, x->z, y->w, z->w.

    This is bounded to avoid pathological cases; v0 meshes are small.
    """
    diamonds: List[Diamond] = []
    for x in G.nodes():
        succ = list(G.successors(x))
        if len(succ) < 2:
            continue
        for i in range(len(succ)):
            for j in range(i+1, len(succ)):
                y = succ[i]; z = succ[j]
                if y == z:
                    continue
                wy = set(G.successors(y))
                wz = set(G.successors(z))
                for w in wy.intersection(wz):
                    diamonds.append(Diamond(x=x, y=y, z=z, w=w))
                    if len(diamonds) >= max_count:
                        return diamonds
    return diamonds


def diamond_mismatch_report(G: nx.DiGraph, diamonds: List[Diamond]) -> Dict[str, Any]:
    """Compute 2-step path mismatch for each diamond, using edge delay/loss.

    For diamond x->y->w and x->z->w:
      delay mismatch = |(delay(x,y)+delay(y,w)) - (delay(x,z)+delay(z,w))|
      loss mismatch  = |(loss(x,y)+loss(y,w)) - (loss(x,z)+loss(z,w))|
      weight mismatch= |(w(x,y)*w(y,w)) - (w(x,z)*w(z,w))|
    """
    d_delays = []
    d_losses = []
    d_weights = []
    for dm in diamonds:
        def edge(u, v):
            return G.edges[u, v]
        a1 = edge(dm.x, dm.y); a2 = edge(dm.y, dm.w)
        b1 = edge(dm.x, dm.z); b2 = edge(dm.z, dm.w)

        da = float(a1.get("delay_ps", 0.0) + a2.get("delay_ps", 0.0))
        db = float(b1.get("delay_ps", 0.0) + b2.get("delay_ps", 0.0))
        la = float(a1.get("loss_dB", 0.0) + a2.get("loss_dB", 0.0))
        lb = float(b1.get("loss_dB", 0.0) + b2.get("loss_dB", 0.0))
        wa = float(a1.get("w", 0.0) * a2.get("w", 0.0))
        wb = float(b1.get("w", 0.0) * b2.get("w", 0.0))

        d_delays.append(abs(da - db))
        d_losses.append(abs(la - lb))
        d_weights.append(abs(wa - wb))

    def summ(x: List[float]) -> Dict[str, float]:
        if not x:
            return {"count": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        arr = np.asarray(x, dtype=float)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(arr.max()),
        }

    return {
        "diamonds": int(len(diamonds)),
        "delay_ps": summ(d_delays),
        "loss_dB": summ(d_losses),
        "weight": summ(d_weights),
    }
