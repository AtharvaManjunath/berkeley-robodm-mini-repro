from __future__ import annotations

import shutil
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def recursive_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def file_sizes(path: Path) -> list[int]:
    if path.is_file():
        return [path.stat().st_size]
    return [p.stat().st_size for p in path.rglob("*") if p.is_file()]
