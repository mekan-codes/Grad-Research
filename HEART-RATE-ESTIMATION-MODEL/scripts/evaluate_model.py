from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.evaluate import evaluate_checkpoint
from src.utils.config import load_config, normalize_config_paths, normalize_model_aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved HR checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return 1
    config = normalize_model_aliases(normalize_config_paths(load_config(args.config))) if args.config else None
    metrics = evaluate_checkpoint(checkpoint, config=config)
    print(f"MAE: {_format_metric(metrics.get('mae'))} bpm")
    print(f"RMSE: {_format_metric(metrics.get('rmse'))} bpm")
    return 0


def _format_metric(value) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
