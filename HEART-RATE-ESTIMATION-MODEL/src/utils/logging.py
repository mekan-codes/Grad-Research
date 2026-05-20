from __future__ import annotations

import logging
from pathlib import Path
import sys


def get_logger(name: str = "ppg_hr", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers and log_file is None:
        return logger
    if logger.handlers and log_file is not None:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == Path(log_file):
                return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)
    return logger


def configure_run_logger(log_dir: str | Path, name: str = "ppg_hr") -> logging.Logger:
    return get_logger(name=name, log_file=Path(log_dir) / "run.log")
