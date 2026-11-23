"""Shared helpers for aggregating per-segment and per-epoch feature dictionaries."""

from __future__ import annotations

import numpy as np


def weighted_feature_mean(segments: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    """Length-weighted mean of numeric features computed on contiguous segments.

    Keys present in at least one segment appear in the output; a key missing
    from a segment is skipped for that segment (not filled with zero).
    """
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    for length, features in segments:
        if length <= 0:
            continue
        for key, value in features.items():
            if not np.isfinite(value):
                continue
            totals[key] = totals.get(key, 0.0) + float(value) * length
            weights[key] = weights.get(key, 0.0) + length
    return {key: totals[key] / weights[key] for key in totals if weights[key] > 0}


def feature_summary(values: list[float]) -> dict[str, float]:
    """Robust documented summary of one feature across epochs/channels."""
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=0)),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
    }
