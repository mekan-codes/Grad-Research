from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with optional ``inherits`` support.

    PyYAML is used when installed. A small fallback parser handles the simple
    nested mappings in this repository so debug scripts still run in minimal
    environments.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = _read_yaml_mapping(config_path)
    parent = config.pop("inherits", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
            if not parent_path.exists():
                parent_path = config_path.parents[1] / parent
        base = load_config(parent_path)
        return deep_update(base, config)
    return config


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
        return loaded
    except ModuleNotFoundError:
        return _parse_simple_yaml(text, path)


def _parse_simple_yaml(text: str, path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line {line_no} in {path}: {raw_line}")

        key, raw_value = line.strip().split(":", 1)
        value_text = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value_text == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value_text)
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

