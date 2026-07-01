from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_framework import train_full_framework
from src.utils.config import load_config, normalize_config_paths, normalize_model_aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyPPG cropper plus HR estimator.")
    parser.add_argument("--config", default="configs/debug_cpu.yaml", help="Path to YAML config.")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, help="Resume from latest or a checkpoint path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = normalize_model_aliases(normalize_config_paths(load_config(args.config)))
    result = train_full_framework(config, resume=args.resume)
    print(f"Best MAE: {result['best_mae']:.3f}")
    print(f"Saved checkpoint: {result['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
