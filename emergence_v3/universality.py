from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .config import EngineConfig
from .graph_generators import generate_graph
from .rng import derive_seed, make_rng
from .rules import generate_random_lookup_rules
from .world import WorldInstance
from .scdc import SCDC_Engine, build_quotient_world
from .observer import make_observer_pocket, sl1_check, degeneracy_bound_check
from .geometry import compute_geometry_diagnostics, GeometryDiagnostics


@dataclass(frozen=True)
class RunOutputs:
    family: str
    seed: int
    graph_meta: Dict[str, Any]
    scdc_iters: int
    num_diamonds_used: int
    sl1_ok: bool
    sl1_entropies: list[int]
    sl1_diffs: list[int]
    degeneracy_ok: bool
    kappa0: int
    min_alpha: Optional[int]
    # Primary universality coordinates (new hypothesis)
    a_star: float
    a_star_ls: float
    delta_a_star: float

    # Secondary ratio diagnostic (kept for reference; not used for classification)
    g_star: float

    # Classification based on (a*, Δa*)
    phase: str  # "Phase0" | "ClassI" | "ClassII"

    # Plateau information (for a_R plateau)
    plateau_ok: bool
    plateau_Rs: list[int]
    plateau_rel_var: float

    # Optional ratio plateau status (g_R plateau)
    g_plateau_ok: bool
    g_plateau_Rs: list[int]
    g_plateau_rel_var: float
    domain_ok_any: bool
    domain_ok_fraction: float


def run_one_world(
    *,
    family: str,
    family_params: Dict[str, Any],
    N: int,
    mean_degree: float,
    engine_cfg: EngineConfig,
    seed_index: int,
    family_index: int,
) -> tuple[RunOutputs, GeometryDiagnostics]:
    base = engine_cfg.random_seed_base
    stride = engine_cfg.universality.family_seed_stride
    seed = derive_seed(base, family, family_index, seed_index, stride)

    rng = make_rng(seed)

    G, meta = generate_graph(
        family,
        N=int(N),
        mean_out_degree=float(mean_degree),
        params=family_params,
        allow_self_loops=engine_cfg.allow_self_loops,
        rng=rng,
    )

    # Alphabets: fixed size per config
    A_size = int(engine_cfg.alphabet.size)
    alphabets = {int(v): list(range(A_size)) for v in G.nodes()}

    # Predecessor lists (unique)
    predecessors = {int(v): sorted(set(int(p) for p in G.predecessors(v))) for v in G.nodes()}

    # Rules: random lookup (capped arity for hubs)
    rules = generate_random_lookup_rules(
        graph_nodes=sorted(int(v) for v in G.nodes()),
        predecessors=predecessors,
        alphabet_sizes={int(v): A_size for v in G.nodes()},
        rng=rng,
        max_arity=engine_cfg.rules.max_arity,
        arity_sampling=engine_cfg.rules.arity_sampling,
    )

    world = WorldInstance.build(G, alphabets, rules)
    world.randomize_state(rng)

    # SCDC equilibration
    scdc_engine = SCDC_Engine(
        world,
        max_iters=engine_cfg.scdc.max_iters,
        star_internal_max_iters=engine_cfg.scdc.star_internal_max_iters,
        diamond_internal_max_iters=engine_cfg.scdc.diamond_internal_max_iters,
        diamond_enabled=engine_cfg.scdc.diamond.enabled,
        diamond_max_diamonds=engine_cfg.scdc.diamond.max_diamonds,
        diamond_sampling=engine_cfg.scdc.diamond.sampling,
        diamond_max_assignments_per_diamond=engine_cfg.scdc.diamond.max_assignments_per_diamond,
        diamond_seed=derive_seed(seed, "diamond"),
    )
    scdc_res = scdc_engine.run()

    # Quotient world
    qworld = build_quotient_world(world, scdc_res.partitions)

    # Observer pocket + SL1
    pocket = make_observer_pocket(
        qworld,
        rng=rng,
        seed_size=engine_cfg.observer.seed_size,
        max_nodes=engine_cfg.observer.max_nodes,
        visible_fraction=engine_cfg.observer.visible_fraction,
        max_hidden_vars=engine_cfg.observer.max_hidden_vars,
        max_attempts=engine_cfg.observer.max_attempts,
    )
    deg = degeneracy_bound_check(qworld, pocket)
    sl1 = sl1_check(
        qworld.copy(),
        pocket,
        ticks=engine_cfg.observer.ticks,
        max_hidden_vars=engine_cfg.observer.max_hidden_vars,
    )

    # Geometry diagnostics (multi-coupling RG flow)
    geom = compute_geometry_diagnostics(
        qworld.graph,
        r_scales=engine_cfg.geometry.r_scales,
        gstar_strategy=engine_cfg.geometry.gstar_strategy,
        gstar_last_k=engine_cfg.geometry.gstar_last_k,
        domain_eta=engine_cfg.geometry.domain_eta,
        domain_min_blocks=engine_cfg.geometry.domain_min_blocks,
        rho_min=engine_cfg.geometry.rho_min,
        plateau_window_k=engine_cfg.geometry.plateau_window_k,
        plateau_rel_tol=engine_cfg.geometry.plateau_rel_tol,
        plateau_min_domain_blocks=engine_cfg.geometry.plateau_min_domain_blocks,
    )

    # Phase classification (new):
    #   Phase0  : no a_R plateau (domain fails or no stable window)
    #   ClassI  : plateau + |Δa*| <= threshold
    #   ClassII : plateau + |Δa*| > threshold
    domain_ok_any = any(bool(x) for x in geom.domain_ok_by_R.values()) if geom.domain_ok_by_R else False
    domain_ok_fraction = (
        float(np.mean([1.0 if geom.domain_ok_by_R.get(R, False) else 0.0 for R in geom.domain_ok_by_R]))
        if geom.domain_ok_by_R
        else 0.0
    )

    delta_thr = float(getattr(engine_cfg.geometry, "delta_a_star_threshold", 0.1))

    if (not np.isfinite(geom.a_star)) or (not bool(geom.plateau_ok)):
        phase = "Phase0"
    else:
        if np.isfinite(geom.delta_a_star) and abs(float(geom.delta_a_star)) <= delta_thr:
            phase = "ClassI"
        else:
            phase = "ClassII"

    out = RunOutputs(
        family=family,
        seed=int(seed),
        graph_meta=meta,
        scdc_iters=int(scdc_res.iters),
        num_diamonds_used=int(scdc_res.num_diamonds_used),
        sl1_ok=bool(sl1["ok"]),
        sl1_entropies=list(sl1["entropies"]),
        sl1_diffs=list(sl1["diffs"]),
        degeneracy_ok=bool(deg["ok"]),
        kappa0=int(deg["kappa0"]),
        min_alpha=None if deg.get("min_alpha") is None else int(deg["min_alpha"]),
        a_star=float(geom.a_star),
        a_star_ls=float(geom.a_star_ls),
        delta_a_star=float(geom.delta_a_star),
        g_star=float(geom.g_star),
        phase=phase,
        plateau_ok=bool(geom.plateau_ok),
        plateau_Rs=list(geom.plateau_Rs),
        plateau_rel_var=float(getattr(geom, "plateau_rel_var", float("nan"))),
        g_plateau_ok=bool(getattr(geom, "g_plateau_ok", False)),
        g_plateau_Rs=list(getattr(geom, "g_plateau_Rs", [])),
        g_plateau_rel_var=float(getattr(geom, "g_plateau_rel_var", float("nan"))),
        domain_ok_any=bool(domain_ok_any),
        domain_ok_fraction=float(domain_ok_fraction),
    )
    return out, geom
