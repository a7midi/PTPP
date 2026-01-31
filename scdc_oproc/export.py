from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from .config import MeshDesign
from .layout import Component, RouteEdge


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_components_csv(path: Path, comps: List[Component]) -> None:
    rows = []
    for c in comps:
        r = {"name": c.name, "kind": c.kind, "x_um": c.x_um, "y_um": c.y_um}
        for k, v in c.props.items():
            r[f"prop_{k}"] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_routes_csv(path: Path, routes: List[RouteEdge]) -> None:
    rows = []
    for r in routes:
        rows.append({
            "u": str(r.u),
            "v": str(r.v),
            "length_um": r.length_um,
            "delay_ps": r.delay_ps,
            "loss_dB": r.loss_dB,
            "kind": r.kind,
            "crosses": r.crosses,
        })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_netlist_json(path: Path, design: MeshDesign, comps: List[Component], routes: List[RouteEdge]) -> None:
    obj: Dict[str, Any] = {
        "design": design.to_dict(),
        "components": [asdict(c) for c in comps],
        "routes": [asdict(r) for r in routes],
    }
    write_json(path, obj)


def write_gdsfactory_stub(path: Path, netlist_json: Path) -> None:
    """Write a stub script that can convert the netlist to GDS in an environment with gdsfactory.

    We *do not* depend on gdsfactory here; this is a handoff artifact.
    """
    code = f'''# Auto-generated stub. Requires: pip install gdsfactory
import json
from pathlib import Path
import gdsfactory as gf

net = json.loads(Path(r"{netlist_json}").read_text())
design = net["design"]
comps = net["components"]

c = gf.Component("scdc_oproc_v0")

# NOTE:
# - This stub uses generic rectangles as placeholders.
# - Replace with your foundry PDK cells (MMI2x2, crossings, grating couplers, etc.)
for comp in comps:
    x = comp["x_um"]
    y = comp["y_um"]
    kind = comp["kind"]
    r = gf.components.rectangle(size=(10, 5), layer=(1,0))
    ref = c.add_ref(r)
    ref.move((x, y))
    ref.name = comp["name"]

# TODO: route waveguides using gf.routing based on net["routes"]
c.write_gds("scdc_oproc_v0.gds")
print("Wrote scdc_oproc_v0.gds")
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
