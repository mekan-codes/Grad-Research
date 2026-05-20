from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_hr import train_hr_estimator
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HR estimator only.")
    parser.add_argument("--config", default="configs/debug_cpu.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    result = train_hr_estimator(config)
    print(f"Best MAE: {result['best_mae']:.3f}")
    print(f"Saved checkpoint: {result['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

