"""Configuration loading with relative inheritance for reproducible experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalar values are replaced."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one JSON config and recursively resolve its optional ``_base_`` key.

    Base paths are interpreted relative to the file that declares them.  A
    string selects one base; a list applies several bases from left to right.
    The returned dictionary never contains ``_base_`` and is therefore the
    complete configuration that should be written as ``resolved_config.json``.
    """
    return _load_config(Path(path).expanduser().resolve(), stack=())


def _load_config(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"cyclic configuration inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        current = json.load(stream)
    if not isinstance(current, dict):
        raise ValueError(f"configuration root must be a JSON object: {path}")

    base_entries = current.pop("_base_", [])
    if isinstance(base_entries, str):
        base_entries = [base_entries]
    if not isinstance(base_entries, list) or not all(
        isinstance(item, str) and item for item in base_entries
    ):
        raise ValueError(f"_base_ must be a path string or a list of path strings: {path}")

    resolved: dict[str, Any] = {}
    for entry in base_entries:
        base_path = (path.parent/entry).resolve()
        resolved = deep_merge(resolved, _load_config(base_path, (*stack, path)))
    return deep_merge(resolved, current)

