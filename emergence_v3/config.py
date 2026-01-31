from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


@dataclass(frozen=True)
class AlphabetConfig:
    kind: str  # "fixed" only (v3) - future: per-node distributions
    size: int


@dataclass(frozen=True)
class RuleConfig:
    kind: str  # "random_lookup" (uniform random truth table)
    max_arity: int
    arity_sampling: str  # "random" | "first"


@dataclass(frozen=True)
class DiamondConfig:
    enabled: bool
    max_diamonds: int
    sampling: str  # "first" | "random"
    max_assignments_per_diamond: int  # guard on brute-force enumeration within one diamond


@dataclass(frozen=True)
class SCDCConfig:
    max_iters: int
    star_internal_max_iters: int
    diamond_internal_max_iters: int
    diamond: DiamondConfig


@dataclass(frozen=True)
class ObserverPocketConfig:
    seed_size: int
    max_nodes: int
    visible_fraction: float
    ticks: int
    max_hidden_vars: int  # hard guard for exact counting

    # NEW: optional cap on the number of seed-resampling attempts when constructing a pocket.
    # If None, the implementation will default to attempts = number of nodes in the world.
    max_attempts: int | None = None



@dataclass(frozen=True)
class GeometryConfig:
    r_scales: List[int]
    fig1_scale_r: int

    # g* estimator strategy
    # - "plateau_ratio": preregistered plateau search on g_R ratio curve (recommended)
    # - "median_last_k": fallback, not a plateau claim
    # - "largest_R": single-scale, weakest
    gstar_strategy: str
    gstar_last_k: int

    # Domain enforcement (Assumption 5.2 denominator condition, operationalized)
    domain_eta: float = 0.05
    domain_min_blocks: int = 20
    rho_min: float = 0.0  # keep domain definition rho>0 but allow excluding tiny positives

    # Plateau criterion
    plateau_window_k: int = 3
    plateau_rel_tol: float = 0.15
    plateau_min_domain_blocks: int = 20

    # Numerical guard for the plateau relative variation criterion when median≈0.
    # Used as denom floor in (max-min)/max(|median|, plateau_median_eps).
    plateau_median_eps: float = 1e-12

    # Classification threshold for "small vs large" Δa*.
    # ClassI if |Δa*| <= delta_a_star_threshold, else ClassII (provided a* plateau exists).
    delta_a_star_threshold: float = 0.1


@dataclass(frozen=True)
class UniversalityConfig:
    seeds_per_family: int
    family_seed_stride: int
    output_dir: str


@dataclass(frozen=True)
class EngineConfig:
    random_seed_base: int
    allow_self_loops: bool
    alphabet: AlphabetConfig
    rules: RuleConfig
    scdc: SCDCConfig
    observer: ObserverPocketConfig
    geometry: GeometryConfig
    universality: UniversalityConfig


def _require(d: Mapping[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    return d[key]


def _get(d: Mapping[str, Any], key: str, default: Any) -> Any:
    return d[key] if key in d else default


def load_engine_config(path: str | Path) -> EngineConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")

    alphabet = AlphabetConfig(**_require(data, "alphabet"))
    rules = RuleConfig(**_require(data, "rules"))
    diamond = DiamondConfig(**_require(_require(data, "scdc"), "diamond"))
    scdc_dict = dict(_require(data, "scdc"))
    scdc_dict.pop("diamond")
    scdc = SCDCConfig(diamond=diamond, **scdc_dict)

    observer = ObserverPocketConfig(**_require(data, "observer"))

    # Geometry: backward compatible defaults for newly introduced PRL-ready fields
    gdict = dict(_require(data, "geometry"))
    geometry = GeometryConfig(
        r_scales=list(gdict["r_scales"]),
        fig1_scale_r=int(gdict["fig1_scale_r"]),
        gstar_strategy=str(gdict.get("gstar_strategy", "plateau_ratio")),
        gstar_last_k=int(gdict.get("gstar_last_k", 3)),
        domain_eta=float(gdict.get("domain_eta", 0.05)),
        domain_min_blocks=int(gdict.get("domain_min_blocks", 20)),
        rho_min=float(gdict.get("rho_min", 0.0)),
        # Backward-compatible: accept plateau_last_k as an alias for plateau_window_k.
        plateau_window_k=int(gdict.get("plateau_window_k", gdict.get("plateau_last_k", 3))),
        plateau_rel_tol=float(gdict.get("plateau_rel_tol", 0.15)),
        # Backward-compatible: accept plateau_min_blocks as an alias.
        plateau_min_domain_blocks=int(gdict.get("plateau_min_domain_blocks", gdict.get("plateau_min_blocks", 20))),
        plateau_median_eps=float(gdict.get("plateau_median_eps", 1e-12)),
        delta_a_star_threshold=float(gdict.get("delta_a_star_threshold", 0.1)),
    )

    universality = UniversalityConfig(**_require(data, "universality"))

    return EngineConfig(
        random_seed_base=int(_require(data, "random_seed_base")),
        allow_self_loops=bool(_require(data, "allow_self_loops")),
        alphabet=alphabet,
        rules=rules,
        scdc=scdc,
        observer=observer,
        geometry=geometry,
        universality=universality,
    )


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} must contain a mapping at the top level")
    return data
