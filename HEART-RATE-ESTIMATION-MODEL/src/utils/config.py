from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config with optional ``inherits`` support.

    PyYAML is used when installed. A small fallback parser handles the simple
    nested mappings in this repository so debug scripts still run in minimal
    environments.
    """

    config_path = resolve_project_path(path, prefer_existing=True)
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


def resolve_project_path(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    prefer_existing: bool = False,
) -> Path:
    """Resolve relative project paths consistently from scripts or the repo root."""

    raw = Path(path)
    if raw.is_absolute():
        return raw

    base = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    candidates = [Path.cwd() / raw, base / raw, PROJECT_ROOT / raw, REPO_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if prefer_existing:
        return base / raw
    return base / raw


def normalize_config_paths(config: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Any]:
    """Return a copy with file-system path fields made absolute.

    Config files in this repo are written with project-relative paths. This
    keeps them usable when a script is launched from a parent folder or IDE.
    """

    updated = deepcopy(config)
    base = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    _normalize_mapping_paths(updated.setdefault("data", {}), ("data_root", "prepared_dir", "data_dir"), base)
    _normalize_mapping_paths(updated.setdefault("tinyppg", {}), ("model_dir", "checkpoint_path"), base)
    _normalize_mapping_paths(updated.setdefault("project", {}), ("output_dir",), base)
    _normalize_mapping_paths(updated.setdefault("paths", {}), ("output_dir", "checkpoint_dir", "log_dir"), base)
    _normalize_mapping_paths(
        updated.setdefault("training", {}),
        ("checkpoint_dir", "hr_checkpoint_dir", "framework_checkpoint_dir"),
        base,
    )
    return updated


def _normalize_mapping_paths(mapping: dict[str, Any], keys: tuple[str, ...], base_dir: Path) -> None:
    for key in keys:
        value = mapping.get(key)
        if value is None or value == "":
            continue
        mapping[key] = str(resolve_project_path(value, base_dir=base_dir))


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    """Write a config dict as YAML when PyYAML is available, JSON otherwise."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(config, sort_keys=False)
    except ModuleNotFoundError:
        import json

        text = json.dumps(config, indent=2)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def apply_cli_overrides(
    config: dict[str, Any],
    data_root: str | None = None,
    output_dir: str | None = None,
    resume: str | bool | None = None,
) -> dict[str, Any]:
    """Apply common launcher overrides without mutating the caller's dict."""

    updated = deepcopy(config)
    updated.setdefault("data", {})
    updated.setdefault("project", {})
    updated.setdefault("paths", {})
    updated.setdefault("training", {})

    if data_root:
        updated["data"]["data_root"] = data_root
    if output_dir:
        updated["project"]["output_dir"] = output_dir
        updated["paths"]["output_dir"] = output_dir
        updated["paths"]["checkpoint_dir"] = str(Path(output_dir) / "checkpoints")
        updated["paths"]["log_dir"] = str(Path(output_dir) / "logs")
        updated["training"]["hr_checkpoint_dir"] = str(Path(output_dir) / "checkpoints" / "baseline")
        updated["training"]["framework_checkpoint_dir"] = str(Path(output_dir) / "checkpoints" / "full_framework")
    if resume is not None:
        updated["resume"] = resume
        updated["training"]["resume"] = resume

    normalize_model_aliases(updated)
    return updated


def normalize_model_aliases(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.setdefault("model", {})
    model_type = str(model_cfg.get("type", model_cfg.get("name", "hr_estimator")))
    aliases = {
        "simple_cnn": "hr_estimator",
        "hr_estimator": "hr_estimator",
        "robust_cnn_gru": "robust_hr_estimator",
        "robust_hr_estimator": "robust_hr_estimator",
    }
    model_cfg["name"] = aliases.get(model_type, model_type)
    return config


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
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].strip()
        if not items:
            return []
        return [_parse_scalar(item.strip()) for item in items.split(",")]
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
