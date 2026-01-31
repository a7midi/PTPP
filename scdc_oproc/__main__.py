from __future__ import annotations

import sys

HELP = """scdc_oproc: SCDC Optical Processor (v0)

Common commands:
  python -m scdc_oproc.build_v0 --pf 0.12 --knot_on --out_dir results/v0_pf012_knot
  python -m scdc_oproc.sweep_v0 --pf_list 0.10 0.12 0.14 --knot_on_list off on --seeds 1 2 3

"""

def main() -> None:
    print(HELP)

if __name__ == "__main__":
    main()
