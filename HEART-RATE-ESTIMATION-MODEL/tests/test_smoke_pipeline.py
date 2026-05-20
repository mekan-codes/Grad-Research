from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


def test_smoke_pipeline_runs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint = root.parent / "Tiny-PPG-master" / "Save_Model" / "model_parameter-2023-5-31-1.pkl"
    if not checkpoint.exists():
        pytest.skip("Local TinyPPG checkpoint is not available")

    output_dir = tmp_path / "smoke_run"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_full_training_pipeline.py",
            "--config",
            "configs/smoke_test.yaml",
            "--smoke-only",
            "--output-dir",
            str(output_dir),
            "--skip-calibration",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "metrics" / "comparison.json").exists()

