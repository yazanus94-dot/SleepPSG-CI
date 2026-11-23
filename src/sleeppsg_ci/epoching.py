"""Epoch-level aggregation helpers shared by the modality feature extractors."""

from __future__ import annotations

import numpy as np

from .preprocessing import epoch_signal


def epoch_matrix(signal: np.ndarray, fs: float, epoch_seconds: float = 30.0) -> np.ndarray:
    """Return a (n_epochs, n_samples) matrix of fixed-duration epochs."""
    return epoch_signal(signal, fs, epoch_seconds=epoch_seconds)


def aggregate_epoch_features(
    values: np.ndarray,
) -> dict[str, float | int]:
    """Summarize one feature across epochs with robust summary statistics.

    NaN epochs are retained in the count but ignored in each statistic; if every
    epoch is NaN the resulting statistics are NaN (never silently zero-filled).
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        stats = {name: float("nan") for name in ("mean", "std", "median", "p10", "p90")}
    else:
        stats = {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=0)),
            "median": float(np.median(finite)),
            "p10": float(np.percentile(finite, 10)),
            "p90": float(np.percentile(finite, 90)),
        }
    return {
        "n_epochs": int(array.size),
        "n_valid_epochs": int(finite.size),
        **stats,
    }


def aggregate_modality_features(
    epoch_features: dict[str, np.ndarray] | None,
) -> dict[str, float | int]:
    """Flatten ``{feature: per_epoch_values}`` into night-level prefixed stats."""
    if not epoch_features:
        return {}
    output: dict[str, float | int] = {}
    for name, values in sorted(epoch_features.items()):
        for stat, value in aggregate_epoch_features(values).items():
            output[f"{name}__{stat}"] = value
    return output
