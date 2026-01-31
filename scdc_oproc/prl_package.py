"""Assemble a lightweight PRL submission bundle for SCDC Optical Processor v0.

This creates a zip file that is suitable as a *submission supplement* alongside a
manuscript PDF. It intentionally does not try to compile LaTeX.

The bundle includes:
  - figures + key_numbers.json (from `scdc_oproc.prl_close`)
  - aggregated CSV backing the plots
  - an example run dir (optional) for auditability
  - the exact code snapshot (optional)

Usage:
  python -m scdc_oproc.prl_package \
    --fig_dir results/prl_figures \
    --out_zip submission/scdc_oproc_v0_supplement.zip \
    --example_run results/prl_submission_data/pf0.120_knot1_seed1 \
    --include_code
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def _zip_add_dir(z: ZipFile, src: Path, arc_prefix: str) -> None:
    for root, _, files in os.walk(src):
        root_p = Path(root)
        for f in files:
            p = root_p / f
            rel = p.relative_to(src)
            z.write(p, str(Path(arc_prefix) / rel))


def _zip_add_file(z: ZipFile, src: Path, arc_path: str) -> None:
    z.write(src, arc_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Package scdc_oproc v0 results + figures into a single zip")
    ap.add_argument("--fig_dir", type=str, required=True, help="Directory produced by scdc_oproc.prl_close")
    ap.add_argument("--out_zip", type=str, required=True, help="Output zip file path")
    ap.add_argument("--example_run", type=str, default=None, help="Optional run folder to include in the zip")
    ap.add_argument("--include_code", action="store_true", help="Include a code snapshot in the zip")
    ap.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Root of scdc_oproc repository. Defaults to parent of this file.",
    )
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    if not fig_dir.exists():
        raise FileNotFoundError(f"fig_dir does not exist: {fig_dir}")

    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    example_run = Path(args.example_run) if args.example_run else None
    if example_run is not None and not example_run.exists():
        raise FileNotFoundError(f"example_run does not exist: {example_run}")

    project_root = Path(args.project_root) if args.project_root else Path(__file__).resolve().parents[1]

    # Minimal expected artifacts from prl_close
    expected = [
        fig_dir / "agg_sim_summary.csv",
        fig_dir / "key_numbers.json",
        fig_dir / "fig1_phase_transition.png",
        fig_dir / "fig2_lensing.png",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "fig_dir is missing required artifacts. Run scdc_oproc.prl_close first. "
            f"Missing: {[str(p) for p in missing]}"
        )

    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as z:
        # Figures + stats
        _zip_add_dir(z, fig_dir, "figures")

        # Example run (audit trail)
        if example_run is not None:
            _zip_add_dir(z, example_run, "example_run")

        # Code snapshot
        if args.include_code:
            # Exclude big result folders if present
            tmp = project_root.parent / "_scdc_oproc_code_snapshot"
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(project_root, tmp)
            # Best-effort cleanup
            for junk in ["results", "__pycache__", ".venv", ".git"]:
                j = tmp / junk
                if j.exists():
                    shutil.rmtree(j, ignore_errors=True)
            _zip_add_dir(z, tmp, "code")
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"Wrote: {out_zip}")


if __name__ == "__main__":
    main()
