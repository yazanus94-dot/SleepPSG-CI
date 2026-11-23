"""Respiratory features from verified respiratory-effort or airflow channels."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .preprocessing import bandpass_filter, detrend_signal


def _segment_features(finite: np.ndarray, fs: float) -> dict[str, float]:
    """Respiratory estimates from one contiguous finite segment."""
    result: dict[str, float] = {
        "resp_rate_bpm": float("nan"),
        "resp_amplitude_cv": float("nan"),
        "resp_signal_range": float("nan"),
        "resp_regularity": float("nan"),
    }
    try:
        filtered = bandpass_filter(finite, fs, 0.05, 2.0, order=2)
    except ValueError:
        filtered = detrend_signal(finite)
    envelope = np.abs(signal.hilbert(filtered))
    peaks, _ = signal.find_peaks(
        envelope,
        distance=max(1, int(round(1.5 * fs))),
        prominence=float(np.std(envelope)),
    )
    result["resp_signal_range"] = float(np.max(envelope) - np.min(envelope))
    if peaks.size >= 2:
        intervals = np.diff(peaks) / fs
        intervals = intervals[(intervals > 1.0) & (intervals < 30.0)]
        if intervals.size:
            result["resp_rate_bpm"] = float(60.0 / np.mean(intervals))
            result["resp_regularity"] = float(
                1.0 / (1.0 + np.std(intervals) / max(np.mean(intervals), 1e-9))
            )
    amplitudes = envelope[peaks]
    if amplitudes.size >= 2 and np.mean(amplitudes) > 0:
        result["resp_amplitude_cv"] = float(np.std(amplitudes) / np.mean(amplitudes))
    return result


def extract_resp_features(signal_array: np.ndarray, fs: float, unit: str | None = None) -> dict[str, float]:
    """Estimate respiratory features within contiguous finite segments.

    Respiratory effort/airflow units are source-specific: no conversion is
    invented. Rate, variability and regularity are unit-invariant;
    ``resp_signal_range`` is unit-dependent and must not be compared across
    sites with different sensor units. The original unit is recorded as a flag.
    """
    from .features_common import weighted_feature_mean
    from .quality import contiguous_finite_segments
    from .units import resolve_unit_conversion, unit_feature_flags

    array = np.asarray(signal_array, dtype=float).reshape(-1)
    handling = resolve_unit_conversion("RESP", unit) if unit is not None else None
    segments = contiguous_finite_segments(array, min_length=64)
    n_finite = int(np.count_nonzero(np.isfinite(array)))
    base: dict[str, float] = {
        "resp_n_samples": float(array.size),
        "resp_nonfinite_fraction": float((array.size - n_finite) / array.size) if array.size else 1.0,
        "resp_n_gaps": float(max(len(segments) - 1, 0)),
        "resp_n_segments": float(len(segments)),
        "resp_features_valid": 0.0,
    }
    base.update(unit_feature_flags("resp", handling))
    for key in ("resp_rate_bpm", "resp_amplitude_cv", "resp_signal_range", "resp_regularity"):
        base[key] = float("nan")
    if n_finite < 64 or not np.isfinite(fs) or fs <= 0:
        return base

    per_segment = [(segment.size, _segment_features(segment, fs)) for segment in segments]
    base.update(weighted_feature_mean(per_segment))
    base["resp_features_valid"] = 1.0
    return base
