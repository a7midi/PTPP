from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class FileHash:
    path: str
    sha256: str
    size_bytes: int


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_files(root: Path, *, exclude_dirs: Iterable[str] = ()) -> List[Path]:
    exclude = set(exclude_dirs)
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in exclude for part in rel.parts):
            continue
        out.append(p)
    return sorted(out)


def build_manifest(root: Path, *, exclude_dirs: Iterable[str] = ()) -> Dict[str, object]:
    files: List[FileHash] = []
    for p in iter_files(root, exclude_dirs=exclude_dirs):
        rel = p.relative_to(root).as_posix()
        files.append(FileHash(path=rel, sha256=sha256_file(p), size_bytes=p.stat().st_size))
    manifest = {
        "root": root.name,
        "file_count": len(files),
        "files": [f.__dict__ for f in files],
    }
    return manifest


def write_manifest(manifest: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
