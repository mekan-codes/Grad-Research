from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_loader import load_ppg_csv
from src.data.preprocessing import PreprocessConfig, config_from_mapping, preprocess_ppg_window
from src.data.windowing import median_label_for_range, sliding_window_ranges


@dataclass
class SubjectRecord:
    subject_id: str
    path: Path
    ppg: np.ndarray
    hr: np.ndarray
    fs: float


@dataclass
class WindowIndex:
    subject_index: int
    start: int
    stop: int
    hr_label: float


class PPGDaLiAWindowDataset(Dataset):
    """Windowed PPG-DaLiA/PPG FieldStudy style CSV dataset.

    Expected prepared CSV columns are ``time``, ``ppg``, and ``hr``. Optional
    accelerometer columns can remain present; they are preserved in metadata for
    future fusion work but are not used by the current HR model.
    """

    def __init__(
        self,
        data_dir: str | Path,
        sampling_rate_hz: float,
        window_sec: float = 30.0,
        step_sec: float | None = None,
        preprocess: dict[str, Any] | None = None,
        max_windows: int | None = None,
        subjects: list[str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.window_sec = float(window_sec)
        self.step_sec = step_sec
        self.preprocess_config = config_from_mapping(preprocess or {}, self.sampling_rate_hz)

        paths = self._discover_paths(subjects)
        if not paths:
            raise FileNotFoundError(
                f"No prepared subject CSV files found in {self.data_dir}. "
                "Run scripts/prepare_ppg_dalia.py or set data.dataset=synthetic for debug."
            )

        self.subjects: list[SubjectRecord] = []
        self.windows: list[WindowIndex] = []
        for path in paths:
            loaded = load_ppg_csv(path, fallback_fs=self.sampling_rate_hz)
            if loaded.ground_truth_hr is None:
                continue
            subject_index = len(self.subjects)
            record = SubjectRecord(
                subject_id=path.stem.upper(),
                path=path,
                ppg=loaded.ppg.astype(float, copy=False),
                hr=loaded.ground_truth_hr.astype(float, copy=False),
                fs=float(loaded.fs),
            )
            self.subjects.append(record)
            ranges = sliding_window_ranges(
                n_samples=record.ppg.size,
                sampling_rate_hz=record.fs,
                window_sec=self.window_sec,
                step_sec=self.step_sec,
                include_partial=False,
            )
            for start, stop in ranges:
                label = median_label_for_range(record.hr, start, stop)
                if label is None or not np.isfinite(label):
                    continue
                self.windows.append(WindowIndex(subject_index, start, stop, label))
                if max_windows is not None and len(self.windows) >= int(max_windows):
                    return

        if not self.windows:
            raise ValueError(f"No labeled PPG windows could be built from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.windows[index]
        subject = self.subjects[item.subject_index]
        raw = subject.ppg[item.start : item.stop]
        preprocess_config = self.preprocess_config
        if abs(preprocess_config.sampling_rate_hz - subject.fs) > 1e-6:
            preprocess_config = PreprocessConfig(
                sampling_rate_hz=subject.fs,
                normalize=preprocess_config.normalize,
                detrend=preprocess_config.detrend,
                bandpass_enabled=preprocess_config.bandpass_enabled,
                bandpass_low_hz=preprocess_config.bandpass_low_hz,
                bandpass_high_hz=preprocess_config.bandpass_high_hz,
                bandpass_order=preprocess_config.bandpass_order,
            )
        ppg = preprocess_ppg_window(raw, preprocess_config)
        return {
            "ppg": torch.from_numpy(ppg),
            "hr_label": float(item.hr_label),
            "metadata": {
                "subject": subject.subject_id,
                "source": str(subject.path),
                "start": item.start,
                "stop": item.stop,
                "sampling_rate_hz": subject.fs,
                "window_sec": (item.stop - item.start) / subject.fs,
            },
        }

    def _discover_paths(self, subjects: list[str] | None) -> list[Path]:
        if not self.data_dir.exists():
            return []
        if subjects:
            wanted = _normalize_subjects(subjects)
            paths = [self.data_dir / f"{subject}.csv" for subject in wanted]
            missing = [subject for subject, path in zip(wanted, paths) if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Prepared CSV(s) not found in {self.data_dir}: {', '.join(missing)}"
                )
            return paths
        return sorted(self.data_dir.glob("*.csv"), key=lambda path: _subject_sort_key(path.stem))


class SyntheticPPGWindowDataset(Dataset):
    """Small deterministic dataset for CPU smoke tests and demos."""

    def __init__(
        self,
        num_samples: int = 32,
        sampling_rate_hz: float = 64.0,
        window_sec: float = 30.0,
        seed: int = 42,
    ) -> None:
        self.num_samples = int(num_samples)
        self.fs = float(sampling_rate_hz)
        self.window_samples = int(round(window_sec * sampling_rate_hz))
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.labels = rng.uniform(55.0, 125.0, size=self.num_samples).astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        hr = float(self.labels[index])
        rng = np.random.default_rng(self.seed + int(index) + 1)
        t = np.arange(self.window_samples, dtype=float) / self.fs
        freq = hr / 60.0
        waveform = (
            np.sin(2.0 * np.pi * freq * t)
            + 0.35 * np.sin(4.0 * np.pi * freq * t + 0.3)
            + 0.05 * np.sin(2.0 * np.pi * 0.2 * t)
        )
        noise = rng.normal(0.0, 0.08, size=t.shape)
        ppg = waveform + noise
        if index % 4 == 0:
            start = self.window_samples // 3
            stop = min(self.window_samples, start + self.window_samples // 8)
            ppg[start:stop] += rng.normal(0.0, 1.0, size=stop - start)
        ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        return {
            "ppg": torch.tensor(ppg, dtype=torch.float32),
            "hr_label": hr,
            "metadata": {
                "source": "synthetic",
                "index": index,
                "sampling_rate_hz": self.fs,
            },
        }


def _subject_sort_key(subject: str) -> tuple[int, str]:
    text = subject.upper()
    try:
        return int(text.lstrip("S")), text
    except ValueError:
        return 9999, text


def _normalize_subjects(subjects: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for subject in subjects:
        text = str(subject).strip().upper().removesuffix(".CSV")
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized
