from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterator

import torch
from torch import nn


@dataclass
class LoadedTinyPPG:
    model: nn.Module
    model_path: Path
    checkpoint_path: Path | None
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)


def load_tinyppg(
    model_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
    require_checkpoint: bool = True,
) -> LoadedTinyPPG:
    """Load TinyPPG from the external folder and freeze it for inference."""

    model_file = find_tinyppg_model_file(model_dir)
    if model_file is None:
        raise FileNotFoundError(
            "Could not find TinyPPG model.py. Checked the configured path, "
            "../Tiny-PPG-master, and checkpoints/tiny_ppg."
        )

    checkpoint = find_tinyppg_checkpoint(checkpoint_path, model_file.parent)
    if checkpoint is None and require_checkpoint:
        raise FileNotFoundError(
            "Could not find a TinyPPG checkpoint. Set tinyppg.checkpoint_path "
            "or place model_parameter-*.pkl under Tiny-PPG-master/Save_Model."
        )

    module = _load_module_from_file(model_file)
    if not hasattr(module, "Model"):
        raise AttributeError(f"TinyPPG module does not define Model: {model_file}")

    model = module.Model()
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    if checkpoint is not None:
        state = _torch_load_state(checkpoint, device)
        state_dict = extract_state_dict(state)
        load_result = model.load_state_dict(state_dict, strict=strict)
        missing_keys = list(getattr(load_result, "missing_keys", []))
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))

    model.to(device)
    freeze_module(model)
    return LoadedTinyPPG(
        model=model,
        model_path=model_file,
        checkpoint_path=checkpoint,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def load_tinyppg_model(
    model_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
    require_checkpoint: bool = True,
) -> nn.Module:
    """Convenience wrapper returning only the frozen TinyPPG module."""

    return load_tinyppg(
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        device=device,
        strict=strict,
        require_checkpoint=require_checkpoint,
    ).model


def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module


def has_trainable_parameters(module: nn.Module) -> bool:
    return any(param.requires_grad for param in module.parameters())


def assert_no_trainable_tinyppg_parameters(module: nn.Module) -> None:
    trainable = [name for name, param in module.named_parameters() if param.requires_grad]
    if trainable:
        preview = ", ".join(trainable[:5])
        raise AssertionError(f"TinyPPG has trainable parameters: {preview}")


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def find_tinyppg_model_file(model_dir: str | Path | None = None) -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    candidates: list[Path] = []
    if model_dir is not None:
        base = _resolve_against_roots(model_dir, project_root, repo_root)
        candidates.extend([base / "model.py", base])
    candidates.extend(
        [
            repo_root / "Tiny-PPG-master" / "model.py",
            project_root / "checkpoints" / "tiny_ppg" / "model.py",
            project_root / "Tiny-PPG-master" / "model.py",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "model.py":
            return candidate.resolve()
    return None


def find_tinyppg_checkpoint(
    checkpoint_path: str | Path | None = None,
    model_dir: Path | None = None,
) -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    if checkpoint_path is not None:
        path = _resolve_against_roots(checkpoint_path, project_root, repo_root)
        return path.resolve() if path.exists() else None

    roots = [
        model_dir / "Save_Model" if model_dir is not None else None,
        model_dir,
        repo_root / "Tiny-PPG-master" / "Save_Model",
        project_root / "checkpoints" / "tiny_ppg",
    ]
    patterns = ("model_parameter*.pkl", "*.pth", "*.pt", "*.ckpt", "*.pkl")
    for root in roots:
        if root is None or not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0].resolve()
    return None


def extract_state_dict(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
    if not isinstance(state, dict):
        raise RuntimeError("TinyPPG checkpoint does not contain a state dict")
    return {
        key.replace("module.", "", 1): value
        for key, value in state.items()
        if hasattr(value, "shape")
    }


def _torch_load_state(path: Path, device: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def _load_module_from_file(model_file: Path) -> Any:
    module_name = f"_tinyppg_model_{abs(hash(str(model_file)))}"
    with _temporary_syspath(model_file.parent):
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load TinyPPG module spec: {model_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@contextmanager
def _temporary_syspath(path: Path) -> Iterator[None]:
    value = str(path)
    inserted = False
    if value not in sys.path:
        sys.path.insert(0, value)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def _resolve_against_roots(path: str | Path, project_root: Path, repo_root: Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    for root in (project_root, repo_root, Path.cwd()):
        candidate = root / raw
        if candidate.exists():
            return candidate
    return project_root / raw

