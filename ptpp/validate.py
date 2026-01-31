from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .hashing import build_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_against_manifest(
    root: Path,
    manifest_path: Path,
    *,
    exclude_dirs: Iterable[str] = ("results",),
) -> tuple[bool, str]:
    """Validate file SHA256 hashes vs a recorded manifest.

    By default we exclude `results/` because those are generated artifacts.
    """
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = build_manifest(root, exclude_dirs=exclude_dirs)

    rec = {f["path"]: f["sha256"] for f in recorded["files"]}
    cur = {f["path"]: f["sha256"] for f in current["files"]}

    missing = sorted(set(rec) - set(cur))
    extra = sorted(set(cur) - set(rec))
    changed = sorted(p for p in rec.keys() & cur.keys() if rec[p] != cur[p])

    ok = (not missing) and (not extra) and (not changed)
    msg = json.dumps({"missing": missing, "extra": extra, "changed": changed}, indent=2)
    return ok, msg


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate repo file hashes vs manifest.")
    ap.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON")
    ap.add_argument("--exclude", type=str, nargs="*", default=["results"], help="Directories to exclude")
    args = ap.parse_args()

    root = _repo_root()
    ok, msg = validate_against_manifest(root, Path(args.manifest), exclude_dirs=tuple(args.exclude))
    print(msg)
    if not ok:
        raise SystemExit(1)
    print("[ptpp] manifest validation OK")


if __name__ == "__main__":
    main()
