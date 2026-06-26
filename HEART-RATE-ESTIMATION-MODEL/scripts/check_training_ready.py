from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import apply_cli_overrides, load_config, normalize_config_paths
from src.utils.environment import print_environment_report, validate_training_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether this workspace is ready for full PPG training.")
    parser.add_argument("--config", default="configs/train_workspace.yaml", help="Training config path.")
    parser.add_argument("--data-root", default=None, help="Override PPG-DaLiA dataset root.")
    parser.add_argument("--output-dir", default=None, help="Override run output directory.")
    parser.add_argument("--allow-missing-data", action="store_true", help="Do not fail if dataset root is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = normalize_config_paths(
        apply_cli_overrides(
            load_config(args.config),
            data_root=args.data_root,
            output_dir=args.output_dir,
        )
    )
    report = validate_training_environment(
        config,
        data_root=config.get("data", {}).get("data_root"),
        output_dir=config.get("paths", {}).get("output_dir") or config.get("project", {}).get("output_dir"),
        require_dataset=not args.allow_missing_data and str(config.get("data", {}).get("dataset", "")).lower() != "synthetic",
        require_tinyppg=True,
        require_plotting=False,
    )
    print_environment_report(report)
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
