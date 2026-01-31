from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, Any, List
import pkgutil


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path, include_suffixes: List[str] = [".py"]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(root.rglob('*')):
        if p.is_file() and (p.suffix in include_suffixes):
            rel = str(p.relative_to(root))
            out[rel] = sha256_file(p)
    return out


def environment_stamp() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def write_meta(path: Path, extra: Dict[str, Any], project_root: Path) -> None:
    meta = {
        "env": environment_stamp(),
        "code_sha256": sha256_tree(project_root / "scdc_oproc", include_suffixes=[".py"]),
        "extra": extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
