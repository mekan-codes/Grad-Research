from __future__ import annotations

from pathlib import Path


def test_plotting_scripts_are_headless_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    for script_name in ("run_artifact_crop_demo.py", "diagnose_tinyppg_output.py"):
        text = (root / "scripts" / script_name).read_text(encoding="utf-8")
        assert "matplotlib.use(\"Agg\", force=True)" in text
        assert "plt.show" not in text
        assert "--no-plot" in text

