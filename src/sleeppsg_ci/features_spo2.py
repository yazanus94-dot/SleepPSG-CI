"""Descriptive SpO2 (pulse oximetry) features from a verified SpO2 channel.

These are distributional summaries of oxygen saturation, not event-based
indices: no attempt is made to score apneas/hypopneas without annotated events.
Non-finite gaps split the signal into contiguous segments that are summarized
separately, so no artificial temporal adjacency is created.
"""

from __future__ import annotations

import numpy as np

from .features_common import weighted_feature_mean
from .quality import contiguous_finite_segments

HYPOXEMIA_THRESHOLD = 90.0  # conventional descriptive SpO2 threshold


def extract_spo2_features(
    signal_array: np.ndarray,
    fs: float,
    unit: str | None = None,
    physical_limits: tuple[float | None, float | None] | None = None,
) -> dict[str, float]:
    """Return night-level SpO2 summaries in canonical percent.

    A fraction unit (e.g. ``1``/``ratio``) is converted to percent **only** when
    the EDF physical range or the observed values support a 0..1 interpretation;
    otherwise the unit is flagged and values are left untouched.
    """
    from .units import apply_unit_conversion, resolve_unit_conversion, unit_feature_flags

    array = np.asarray(signal_array, dtype=float).reshape(-1)
    low, high = physical_limits if physical_limits is not None else (None, None)
    handling = (
        resolve_unit_conversion(
            "SPO2", unit, physical_min=low, physical_max=high, sample_values=array
        )
        if unit is not None else None
    )
    array = apply_unit_conversion(array, handling)
    segments = contiguous_finite_segments(array, min_length=2)
    all_finite = np.concatenate(segments) if segments else np.array([], dtype=float)
    n_gaps = max(len(segments) - 1, 0)
    base: dict[str, float] = {
        "spo2_n_samples": float(array.size),
        "spo2_nonfinite_fraction": float((array.size - all_finite.size) / array.size) if array.size else 1.0,
        "spo2_n_gaps": float(n_gaps),
        "spo2_n_segments": float(len(segments)),
        "spo2_features_valid": 0.0,
    }
    base.update(unit_feature_flags("spo2", handling))
    for key in ("spo2_mean", "spo2_median", "spo2_min", "spo2_std", "spo2_below_90_fraction"):
        base[key] = float("nan")
    if all_finite.size < 2 or not np.isfinite(fs) or fs <= 0:
        return base

    summaries = []
    for segment in segments:
        summaries.append((segment.size, {
            "spo2_mean": float(np.mean(segment)),
            "spo2_median": float(np.median(segment)),
            "spo2_min": float(np.min(segment)),
            "spo2_std": float(np.std(segment, ddof=0)),
            "spo2_below_90_fraction": float(np.mean(segment < HYPOXEMIA_THRESHOLD)),
        }))
    base.update(weighted_feature_mean(summaries))
    base["spo2_features_valid"] = 1.0
    return base
