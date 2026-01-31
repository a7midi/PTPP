from __future__ import annotations

"""emergence_v3.geometry

Geometry diagnostics for Emergence v3.

Core claim (letter upgrade): *geometry is a multi-coupling RG flow on heterogeneous substrates*.

Accordingly we compute, at each blocking scale R on the condensation DAG:
  - domain-guarded block observables (rho_R, kappa_R)
  - ratio coupling:      g_R = median(kappa_R / rho_R)   (secondary diagnostic)
  - affine couplings:    kappa_R \approx a_R rho_R + b_R  (primary diagnostic)

Primary universality order parameter:
  a*  = plateau value of robust slope a_R (Theil–Sen) on a preregistered plateau window.

Secondary heterogeneity axis (preferred option):
  Δa* = a*_LS − a*_rob, where a*_LS is the plateau median least-squares slope on the *same* window.

Plateau selection is designed to be review-proof:
  - only consider scales R where domain_ok[R] is true AND enough domain blocks exist
  - scan contiguous windows coarse-to-fine in R (largest R first) and return the first passing window
  - criterion: (max−min)/|median| with stable handling when median≈0

Blocking / domain semantics are intentionally unchanged from the existing suite:
  - condensation DAG computed deterministically (stable SCC labeling)
  - depth slices partitioned deterministically into chunk blocks of size ~R
  - domain is blocks with rho_R > 0 (optionally rho_R > rho_min)

No heavyweight dependencies are introduced: SciPy is optional; if absent, robust fits fall back to
least-squares.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import math

import numpy as np
import networkx as nx

try:
    from scipy.stats import theilslopes  # robust slope estimate
except Exception:  # pragma: no cover
    theilslopes = None


@dataclass
class GeometryDiagnostics:
    # -------------------------
    # Primary (new hypothesis)
    # -------------------------

    # Plateau value of robust slope a_R (Theil–Sen) on the declared domain.
    a_star: float = float("nan")

    # Plateau value of least-squares slope on the same window.
    a_star_ls: float = float("nan")

    # Secondary heterogeneity marker: Δa* = a*_LS − a*_rob (same window).
    delta_a_star: float = float("nan")

    # Slope plateau selection output (this replaces g*-centric plateau_ok).
    plateau_ok: bool = False
    plateau_Rs: List[int] = field(default_factory=list)
    plateau_rel_var: float = float("nan")
    astar_source: str = "nan"
    reason_a: str = ""

    # -------------------------
    # Secondary (ratio diagnostic)
    # -------------------------

    # Optional ratio plateau estimator g⋆ (kept for backwards-compatible diagnostics)
    g_star: float = float("nan")
    g_plateau_ok: bool = False
    g_plateau_Rs: List[int] = field(default_factory=list)
    g_plateau_rel_var: float = float("nan")
    gstar_source: str = "nan"
    reason_g: str = ""

    # Ratio curve: g_R = median(kappa_R / rho_R) on domain blocks.
    gR_by_R: Dict[int, float] = field(default_factory=dict)

    # Affine couplings per scale (domain-only fit):
    # robust (Theil–Sen) slope/intercept
    aR_by_R: Dict[int, float] = field(default_factory=dict)
    bR_by_R: Dict[int, float] = field(default_factory=dict)
    # least-squares slope/intercept
    aR_ls_by_R: Dict[int, float] = field(default_factory=dict)
    bR_ls_by_R: Dict[int, float] = field(default_factory=dict)

    # Per-scale heterogeneity diagnostic: Δa_R = a_R_LS − a_R_rob
    delta_aR_by_R: Dict[int, float] = field(default_factory=dict)

    # Points used for scatter plots at each R (DOMAIN points only: rho>0 and kappa defined)
    # maps R -> (rho_array, kappa_array)
    points_by_R: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # Optional: all used blocks including rho<=0 (for debugging/diagnostics)
    points_all_by_R: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # Per-R block summaries on the DOMAIN (rho>0)
    rho_med_by_R: Dict[int, float] = field(default_factory=dict)
    kappa_med_by_R: Dict[int, float] = field(default_factory=dict)

    # Domain / quality metrics per R
    n_blocks_used_by_R: Dict[int, int] = field(default_factory=dict)
    n_domain_blocks_by_R: Dict[int, int] = field(default_factory=dict)
    frac_pos_blocks_by_R: Dict[int, float] = field(default_factory=dict)
    mean_rho_pos_by_R: Dict[int, float] = field(default_factory=dict)
    domain_ok_by_R: Dict[int, bool] = field(default_factory=dict)

    # Structural stats (helps explain domain failures)
    num_scc: int = 0
    max_depth: int = 0
    depth_hist: Dict[int, int] = field(default_factory=dict)

    # -------------------------
    # Backwards-compatible aliases
    # -------------------------

    @property
    def slopes_by_R(self) -> Dict[int, float]:
        """Alias for aR_by_R (robust slope), kept for older scripts."""
        return self.aR_by_R

    @property
    def intercepts_by_R(self) -> Dict[int, float]:
        """Alias for bR_by_R (robust intercept), kept for older scripts."""
        return self.bR_by_R

    @property
    def slopes_ls_by_R(self) -> Dict[int, float]:
        """Alias for aR_ls_by_R (LS slope), kept for convenience."""
        return self.aR_ls_by_R

    @property
    def intercepts_ls_by_R(self) -> Dict[int, float]:
        """Alias for bR_ls_by_R (LS intercept), kept for convenience."""
        return self.bR_ls_by_R


# -------------------------
# Graph preprocessing
# -------------------------


def _as_digraph_simple(G: nx.MultiDiGraph | nx.DiGraph) -> nx.DiGraph:
    """Convert to a simple DiGraph view (collapse parallel edges)."""
    if isinstance(G, nx.MultiDiGraph):
        DG = nx.DiGraph()
        DG.add_nodes_from(G.nodes())
        DG.add_edges_from((int(u), int(v)) for u, v, _k in G.edges(keys=True))
        return DG
    if isinstance(G, nx.DiGraph):
        return nx.DiGraph(G)  # copy
    return nx.DiGraph(G)


def condensation_dag_stable(G: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[int, int]]:
    """Deterministic condensation DAG and node->SCC map.

    NetworkX's condensation labels SCC nodes in discovery order; to stabilize, we:
      - compute SCCs
      - sort SCCs by min original node id
      - build the condensed DAG explicitly
    """
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        C = nx.DiGraph()
        return C, {}

    sccs_sorted = sorted((set(int(x) for x in comp) for comp in sccs), key=lambda c: min(c))
    scc_map: Dict[int, int] = {}
    for i, comp in enumerate(sccs_sorted):
        for v in comp:
            scc_map[int(v)] = int(i)

    C = nx.DiGraph()
    C.add_nodes_from(range(len(sccs_sorted)))
    for u, v in G.edges():
        su = scc_map[int(u)]
        sv = scc_map[int(v)]
        if su != sv:
            C.add_edge(int(su), int(sv))
    return C, scc_map


def dag_depths_longest_path_ending(C: nx.DiGraph) -> Dict[int, int]:
    """depth(x) = length of a longest directed path ending at x (Definition 2.2)."""
    if C.number_of_nodes() == 0:
        return {}
    topo = list(nx.topological_sort(C))
    depth: Dict[int, int] = {int(n): 0 for n in topo}
    for n in topo:
        n = int(n)
        best = 0
        for p in C.predecessors(n):
            p = int(p)
            best = max(best, depth[p] + 1)
        depth[n] = best
    return depth


def _partition_depth_slice_into_blocks(nodes_at_depth: List[int], R: int) -> List[List[int]]:
    """Deterministic chunking into blocks of size ~R.

    Rule: sort SCC ids and chunk sequentially into groups of size R; include last remainder block.
    """
    if not nodes_at_depth:
        return []
    nodes_sorted = sorted(int(x) for x in nodes_at_depth)
    blocks: List[List[int]] = []
    for i in range(0, len(nodes_sorted), int(R)):
        blocks.append(nodes_sorted[i : i + int(R)])
    return blocks


# -------------------------
# Fits + plateau logic
# -------------------------


def _robust_line_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (slope, intercept) for y ≈ slope*x + intercept."""
    if x.size < 2:
        return float("nan"), float("nan")

    # Theil–Sen if available; otherwise fallback to least squares
    if theilslopes is not None:
        res = theilslopes(y, x, 0.95)
        return float(res[0]), float(res[1])

    try:
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)
    except Exception:
        return float("nan"), float("nan")


def _ls_line_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (slope, intercept) for y ≈ slope*x + intercept (least squares)."""
    if x.size < 2:
        return float("nan"), float("nan")
    # Guard against degenerate x
    if float(np.std(x)) == 0.0:
        return float("nan"), float("nan")
    try:
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)
    except Exception:
        return float("nan"), float("nan")


def _median_ratio(kappa: np.ndarray, rho: np.ndarray) -> float:
    if kappa.size == 0 or rho.size == 0:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = kappa / rho
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(np.median(r))


def _relative_variation(vals: Sequence[float], *, median_eps: float) -> float:
    """(max-min)/|median| with stable handling when median≈0 via denom floor."""
    if not vals:
        return float("inf")
    med = float(np.median(np.array(vals, dtype=float)))
    span = float(max(vals) - min(vals))
    denom = max(abs(med), float(median_eps))
    return float(span / denom)


def _select_plateau_window(
    *,
    Rs: List[int],
    values_by_R: Optional[Dict[int, float]] = None,
    # Backwards-compatible alias used by some older scripts
    gR: Optional[Dict[int, float]] = None,
    domain_ok: Dict[int, bool],
    n_domain: Dict[int, int],
    window_k: int,
    min_blocks: int,
    rel_tol: float,
    median_eps: float = 1e-12,
) -> tuple[bool, List[int], float]:
    """Review-proof plateau finder.

    - Eligible scales must satisfy:
        * finite value
        * domain_ok[R] == True
        * n_domain[R] >= min_blocks
    - Searches contiguous windows of fixed length window_k.
    - Scans windows from the coarsest end (largest R) downward and returns the first passing window.
    """

    if values_by_R is None:
        values_by_R = gR or {}

    if not Rs:
        return False, [], float("inf")

    Rs_sorted = sorted(int(r) for r in Rs)
    k = max(1, int(window_k))
    if len(Rs_sorted) < k:
        return False, [], float("inf")

    def eligible(R: int) -> Optional[float]:
        v = float(values_by_R.get(int(R), float("nan")))
        if not math.isfinite(v):
            return None
        if not bool(domain_ok.get(int(R), False)):
            return None
        if int(n_domain.get(int(R), 0)) < int(min_blocks):
            return None
        return float(v)

    # Scan from largest R downward (coarse-to-fine RG scan)
    for start in range(len(Rs_sorted) - k, -1, -1):
        window = Rs_sorted[start : start + k]
        vals: List[float] = []
        ok = True
        for R in window:
            v = eligible(int(R))
            if v is None:
                ok = False
                break
            vals.append(float(v))
        if not ok:
            continue

        rel_var = _relative_variation(vals, median_eps=float(median_eps))
        if rel_var <= float(rel_tol):
            return True, [int(x) for x in window], float(rel_var)

    return False, [], float("inf")


# -------------------------
# Public API
# -------------------------


def compute_geometry_diagnostics(
    G: nx.MultiDiGraph | nx.DiGraph,
    *,
    r_scales: Sequence[int],
    # Secondary ratio estimator (kept for backwards-compatible diagnostics)
    gstar_strategy: str = "plateau_ratio",
    gstar_last_k: int = 3,
    # Domain thresholds
    domain_eta: float = 0.05,
    domain_min_blocks: int = 20,
    rho_min: float = 0.0,
    # Plateau search parameters (used for the PRIMARY a_R plateau)
    plateau_window_k: int = 3,
    plateau_rel_tol: float = 0.15,
    plateau_min_domain_blocks: int = 20,
    # Handling median≈0 in the relative variation criterion
    plateau_median_eps: float = 1e-12,
) -> GeometryDiagnostics:
    """Compute per-scale geometry diagnostics on the condensation DAG.

    Returns GeometryDiagnostics containing:
      - per-R domain stats and scatter points
      - per-R couplings (g_R, a_R, b_R, and LS analogs)
      - primary plateau (a*) and secondary heterogeneity marker (Δa*)
      - optional ratio plateau summary (g⋆)

    Notes:
      - The *primary* plateau is always attempted on a_R (robust slope).
      - g⋆ is computed only according to gstar_strategy and is treated as secondary.
    """

    diag = GeometryDiagnostics()

    DG = _as_digraph_simple(G)
    C, _scc_map = condensation_dag_stable(DG)
    depths = dag_depths_longest_path_ending(C)

    diag.num_scc = int(C.number_of_nodes())
    if diag.num_scc == 0:
        diag.reason_a = "empty condensation DAG"
        diag.reason_g = "empty condensation DAG"
        return diag

    diag.max_depth = int(max(depths.values())) if depths else 0
    for d in depths.values():
        diag.depth_hist[int(d)] = diag.depth_hist.get(int(d), 0) + 1

    # Per-SCC observables on C (r=1 version)
    outdeg = {int(v): int(C.out_degree(v)) for v in C.nodes()}
    rho_mem = {int(v): float(outdeg[int(v)] - 1) for v in C.nodes()}

    # Truncated future cone volume at r=1: V1(v) = 1 + |succ(v)|
    V1 = {int(v): float(1 + len(set(int(x) for x in C.successors(v)))) for v in C.nodes()}

    # Edge curvature proxy: kappa(i->j) = V1(j) - V1(i)
    edge_kappa: Dict[Tuple[int, int], float] = {}
    for i, j in C.edges():
        i = int(i)
        j = int(j)
        edge_kappa[(i, j)] = float(V1[j] - V1[i])

    # Depth slices
    by_depth: Dict[int, List[int]] = {}
    for v, d in depths.items():
        by_depth.setdefault(int(d), []).append(int(v))

    Rs = sorted(int(r) for r in r_scales)

    for R in Rs:
        rho_all: List[float] = []
        kappa_all: List[float] = []
        rho_dom: List[float] = []
        kappa_dom: List[float] = []

        n_blocks_used = 0
        n_domain = 0

        for _d, nodes_d in by_depth.items():
            blocks = _partition_depth_slice_into_blocks(nodes_d, R=int(R))
            for block in blocks:
                if not block:
                    continue

                # rho_R(block) = mean rho_mem over SCC nodes in block
                rhoR = float(np.mean([rho_mem[int(v)] for v in block]))

                # kappa_R(block) = mean kappa(i->j) over outgoing edges from SCC nodes in block
                kappas: List[float] = []
                for i in block:
                    i = int(i)
                    for j in C.successors(i):
                        j = int(j)
                        kappas.append(edge_kappa[(i, j)])

                if not kappas:
                    # Undefined curvature for blocks with no outgoing edges
                    continue

                kappaR = float(np.mean(kappas))

                n_blocks_used += 1
                rho_all.append(rhoR)
                kappa_all.append(kappaR)

                if rhoR > float(rho_min) and rhoR > 0.0:
                    n_domain += 1
                    rho_dom.append(rhoR)
                    kappa_dom.append(kappaR)

        diag.n_blocks_used_by_R[R] = int(n_blocks_used)
        diag.n_domain_blocks_by_R[R] = int(n_domain)
        diag.frac_pos_blocks_by_R[R] = float(n_domain / n_blocks_used) if n_blocks_used > 0 else 0.0
        diag.mean_rho_pos_by_R[R] = float(np.mean(rho_dom)) if n_domain > 0 else float("nan")

        # Domain OK: enough positive blocks AND mean denominator positive enough
        diag.domain_ok_by_R[R] = bool(
            (n_domain >= int(domain_min_blocks))
            and (math.isfinite(diag.mean_rho_pos_by_R[R]))
            and (diag.mean_rho_pos_by_R[R] >= float(domain_eta))
        )

        # Save points
        diag.points_all_by_R[R] = (np.array(rho_all, dtype=float), np.array(kappa_all, dtype=float))
        diag.points_by_R[R] = (np.array(rho_dom, dtype=float), np.array(kappa_dom, dtype=float))

        # Per-R summaries on the domain
        if n_domain > 0:
            rho_arr = diag.points_by_R[R][0]
            kappa_arr = diag.points_by_R[R][1]
            diag.rho_med_by_R[R] = float(np.median(rho_arr))
            diag.kappa_med_by_R[R] = float(np.median(kappa_arr))

            # g_R ratio coupling (secondary diagnostic)
            diag.gR_by_R[R] = _median_ratio(kappa_arr, rho_arr)

            # a_R, b_R robust
            a, b = _robust_line_fit(rho_arr, kappa_arr)
            diag.aR_by_R[R] = float(a)
            diag.bR_by_R[R] = float(b)

            # a_R_LS, b_R_LS
            als, bls = _ls_line_fit(rho_arr, kappa_arr)
            diag.aR_ls_by_R[R] = float(als)
            diag.bR_ls_by_R[R] = float(bls)

            # Δa_R
            if math.isfinite(als) and math.isfinite(a):
                diag.delta_aR_by_R[R] = float(als - a)
            else:
                diag.delta_aR_by_R[R] = float("nan")
        else:
            diag.rho_med_by_R[R] = float("nan")
            diag.kappa_med_by_R[R] = float("nan")
            diag.gR_by_R[R] = float("nan")
            diag.aR_by_R[R] = float("nan")
            diag.bR_by_R[R] = float("nan")
            diag.aR_ls_by_R[R] = float("nan")
            diag.bR_ls_by_R[R] = float("nan")
            diag.delta_aR_by_R[R] = float("nan")

    # -------------------------
    # Primary plateau: a_R (robust slope)
    # -------------------------

    ok_a, win_a, rel_a = _select_plateau_window(
        Rs=Rs,
        values_by_R=diag.aR_by_R,
        domain_ok=diag.domain_ok_by_R,
        n_domain=diag.n_domain_blocks_by_R,
        window_k=int(plateau_window_k),
        min_blocks=int(plateau_min_domain_blocks),
        rel_tol=float(plateau_rel_tol),
        median_eps=float(plateau_median_eps),
    )

    diag.plateau_ok = bool(ok_a)
    diag.plateau_Rs = list(win_a)
    diag.plateau_rel_var = float(rel_a)

    if ok_a and win_a:
        a_vals = [float(diag.aR_by_R[R]) for R in win_a]
        als_vals = [float(diag.aR_ls_by_R.get(R, float("nan"))) for R in win_a]
        if all(math.isfinite(v) for v in a_vals):
            diag.a_star = float(np.median(np.array(a_vals, dtype=float)))
            diag.astar_source = f"plateau_slope:{win_a}"
        if all(math.isfinite(v) for v in als_vals):
            diag.a_star_ls = float(np.median(np.array(als_vals, dtype=float)))

        if math.isfinite(diag.a_star) and math.isfinite(diag.a_star_ls):
            diag.delta_a_star = float(diag.a_star_ls - diag.a_star)
        diag.reason_a = ""
    else:
        diag.a_star = float("nan")
        diag.a_star_ls = float("nan")
        diag.delta_a_star = float("nan")
        diag.astar_source = "nan"
        diag.reason_a = (
            "no a_R plateau window found on the domain; inspect domain_ok_by_R / n_domain_blocks_by_R, "
            "or adjust r_scales / plateau_* settings"
        )

    # -------------------------
    # Secondary ratio summary: g⋆
    # -------------------------

    gstar_strategy_l = str(gstar_strategy).lower().strip()

    valid_gRs = [R for R in Rs if math.isfinite(diag.gR_by_R.get(R, float("nan")))]

    if gstar_strategy_l in {"plateau_ratio", "plateau"}:
        ok_g, win_g, rel_g = _select_plateau_window(
            Rs=Rs,
            values_by_R=diag.gR_by_R,
            domain_ok=diag.domain_ok_by_R,
            n_domain=diag.n_domain_blocks_by_R,
            window_k=int(plateau_window_k),
            min_blocks=int(plateau_min_domain_blocks),
            rel_tol=float(plateau_rel_tol),
            median_eps=float(plateau_median_eps),
        )
        diag.g_plateau_ok = bool(ok_g)
        diag.g_plateau_Rs = list(win_g)
        diag.g_plateau_rel_var = float(rel_g)

        if ok_g and win_g:
            g_vals = [float(diag.gR_by_R[R]) for R in win_g]
            diag.g_star = float(np.median(np.array(g_vals, dtype=float)))
            diag.gstar_source = f"plateau_ratio:{win_g}"
            diag.reason_g = ""
        else:
            diag.g_star = float("nan")
            diag.gstar_source = "nan"
            diag.reason_g = "no g_R plateau window found (this is expected in Class II / affine-dominated cases)"

    elif gstar_strategy_l in {"median_last_k", "last_k"}:
        k = max(1, int(gstar_last_k))
        tail = sorted(valid_gRs)[-k:] if len(valid_gRs) >= k else []
        if len(tail) < k:
            diag.reason_g = f"insufficient valid g_R values for last_k={k}"
        else:
            vals = [float(diag.gR_by_R[R]) for R in tail]
            diag.g_star = float(np.median(np.array(vals, dtype=float)))
            diag.gstar_source = f"median_last_k:{tail}"
            diag.reason_g = ""

    elif gstar_strategy_l in {"largest_r", "largest_r_scale", "largest_r".replace("_", "").lower()}:
        # strongest single-scale diagnostic: use g_R at the largest scale with a defined domain.
        if valid_gRs:
            Rmax = max(valid_gRs)
            diag.g_star = float(diag.gR_by_R[Rmax])
            diag.gstar_source = f"largest_R:{Rmax}"
            diag.reason_g = ""
        else:
            diag.reason_g = "no valid g_R values"

    elif gstar_strategy_l in {"none", "off", "disable"}:
        diag.g_star = float("nan")
        diag.gstar_source = "disabled"
        diag.reason_g = "disabled"

    else:
        # Unknown strategy: keep g⋆ as NaN, but do not fail the whole diagnostic.
        diag.g_star = float("nan")
        diag.gstar_source = "nan"
        diag.reason_g = f"unknown gstar_strategy={gstar_strategy!r}"

    return diag
