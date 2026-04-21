from __future__ import annotations

from pathlib import Path


def resolve_path(raw_path: str | Path | None, base_dir: str | Path) -> Path | None:
    """Resolve path relative to base_dir unless raw_path is absolute."""
    if raw_path is None:
        return None

    raw = str(raw_path).strip()
    if raw == "":
        return None

    path = Path(raw)
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()
