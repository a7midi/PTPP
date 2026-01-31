from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import subprocess
import sys
import json

from .config import SweepSpec


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a small design matrix sweep for SCDC Optical Processor v0")
    ap.add_argument("--pf_list", type=float, nargs="+", default=[0.10, 0.12, 0.14])
    ap.add_argument("--knot_on_list", type=str, nargs="+", default=["off", "on"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out_root", type=str, default="results/oproc_v0_sweep")
    ap.add_argument("--mode", choices=["steady", "pulsed", "both"], default="steady")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    spec = {
        "pf_list": list(map(float, args.pf_list)),
        "knot_on_list": args.knot_on_list,
        "seeds": list(map(int, args.seeds)),
        "mode": args.mode,
    }
    (out_root / "sweep_spec.json").write_text(json.dumps(spec, indent=2))

    for pf in args.pf_list:
        for kon in args.knot_on_list:
            knot_on = (kon.lower() == "on")
            for seed in args.seeds:
                out_dir = out_root / f"pf{pf:.3f}_knot{int(knot_on)}_seed{seed}"
                cmd = [
                    sys.executable, "-m", "scdc_oproc.build_v0",
                    "--L", "50", "--W", "20",
                    "--pf", str(pf),
                    "--pknot", "0.90",
                    "--knot_layers", "10,18",
                    "--knot_channels", "6,14",
                    "--seed", str(seed),
                    "--out_dir", str(out_dir),
                    "--mode", args.mode,
                ]
                if knot_on:
                    cmd.append("--knot_on")
                print("Running:", " ".join(cmd))
                subprocess.check_call(cmd)

    print(f"Sweep complete. Results under: {out_root}")


if __name__ == "__main__":
    main()
