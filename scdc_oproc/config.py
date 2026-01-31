from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any


@dataclass(frozen=True)
class MeshDesign:
    """Hardware-constrained SCDC optical mesh design (v0).

    This is intentionally *photonic realistic*:
    - W parallel waveguide channels (rails)
    - L stages (layers) of nearest-neighbor 2x2 couplers (planar, low crossings)
    - Coupler occupancy is controlled by pf (background) and pknot (inside knot)
    - Fan-in/out are bounded by construction (2x2 only).

    The design is a *directed layered graph* in time/propagation direction.
    """
    L: int = 50
    W: int = 20

    # "Forward density": probability a coupler site is populated in bulk
    pf: float = 0.12

    # Knot parameters
    knot_on: bool = True
    knot_layers: Tuple[int, int] = (10, 18)   # [start, end) stage range
    knot_channels: Tuple[int, int] = (6, 14)  # [start, end) channel range
    pknot: float = 0.90

    # Coupler model (intensity-mixing, incoherent mode)
    split_eta: float = 0.50            # nominal cross-coupling (0.5 = 50/50)
    split_sigma: float = 0.02          # per-coupler variation (Gaussian std)
    il_dB: float = 0.15                # insertion loss per coupler (dB), typical placeholder

    # Waveguide model (placeholders; use PDK values in real tapeout)
    stage_pitch_um: float = 200.0      # x pitch per stage for placement
    channel_pitch_um: float = 20.0     # y pitch per channel for placement
    wg_loss_dB_per_cm: float = 2.0     # propagation loss (dB/cm), placeholder
    wg_group_index: float = 4.0        # group index for delay estimates, placeholder

    # Simulation / measurement
    power_threshold: float = 1e-4      # threshold for 'active node' counting
    max_time_bins: int = 256           # for pulsed sim (optional)
    dt_ps: float = 5.0                 # discrete time bin size (ps) for pulsed sim

    # Reproducibility
    seed: int = 1
    tag: str = "v0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SweepSpec:
    pf_list: List[float]
    knot_on_list: List[bool]
    seeds: List[int]
