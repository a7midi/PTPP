from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .config import MeshDesign
from .layout import Component
from .sim import SimResult


def plot_layout(design: MeshDesign, comps: List[Component], out_path: Path) -> None:
    xs = [c.x_um for c in comps if c.kind == "coupler_2x2"]
    ys = [c.y_um for c in comps if c.kind == "coupler_2x2"]
    plt.figure()
    plt.scatter(xs, ys, s=10)
    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.title(f"Coupler placement L={design.L} W={design.W} pf={design.pf} knot={design.knot_on}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_active_profile(per_stage: List[int], out_path: Path, title: str) -> None:
    plt.figure()
    plt.plot(per_stage)
    plt.xlabel("stage")
    plt.ylabel("# active channels")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_outputs(outputs_energy: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.bar(np.arange(outputs_energy.size), outputs_energy)
    plt.xlabel("output channel")
    plt.ylabel("integrated power (arb.)")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_impulse_heatmap(sim: SimResult, out_path: Path, title: str, vmax: Optional[float] = None) -> None:
    if sim.pulse_P is None:
        return
    # show outputs at final stage: W x T heatmap
    mat = sim.pulse_P[-1, :, :]
    plt.figure()
    plt.imshow(mat, aspect='auto', origin='lower', vmax=vmax)
    plt.xlabel("time bin")
    plt.ylabel("output channel")
    plt.title(title)
    plt.colorbar(label="power")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
