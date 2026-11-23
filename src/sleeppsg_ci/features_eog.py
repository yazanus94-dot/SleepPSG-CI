"""EOG amplitude, blink-like event and spectral features."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .features_eeg import welch_psd
from .preprocessing import detrend_signal


def _segment_features(finite: np.ndarray, fs: float) -> dict[str, float]:
    centered = detrend_signal(finite) if finite.size > 2 else finite - np.mean(finite)
    result: dict[str, float] = {
        "eog_std": float(np.std(centered)),
        "eog_peak_to_peak": float(np.max(centered) - np.min(centered)),
        "eog_slow_fast_ratio": float("nan"),
        "eog_blink_like_events": 0.0,
        "eog_total_power": float("nan"),
    }
    try:
        freqs, psd = welch_psd(centered, fs)
        total = float(np.sum(psd))
        slow = (freqs >= 0.1) & (freqs < 4.0)
        fast = (freqs >= 4.0) & (freqs < 16.0)
        slow_power = float(np.sum(psd[slow]))
        fast_power = float(np.sum(psd[fast]))
        result["eog_slow_fast_ratio"] = slow_power / fast_power if fast_power > 0 else float("nan")
        result["eog_total_power"] = total
        envelope = np.abs(signal.hilbert(centered))
        peaks, _ = signal.find_peaks(
            envelope,
            height=float(np.mean(envelope) + 2.0 * np.std(envelope)),
            distance=max(1, int(round(0.2 * fs))),
        )
        result["eog_blink_like_events"] = float(peaks.size)
    except ValueError:
        pass
    return result


def extract_eog_features(signal_array: np.ndarray, fs: float, unit: str | None = None) -> dict[str, float]:
    """Return EOG features in canonical microvolts from finite segments."""
    from .features_common import weighted_feature_mean
    from .quality import contiguous_finite_segments
    from .units import apply_unit_conversion, resolve_unit_conversion, unit_feature_flags

    array = np.asarray(signal_array, dtype=float).reshape(-1)
    handling = resolve_unit_conversion("EOG", unit) if unit is not None else None
    array = apply_unit_conversion(array, handling)
    segments = contiguous_finite_segments(array, min_length=16)
    n_finite = int(np.count_nonzero(np.isfinite(array)))
    base: dict[str, float] = {
        "eog_n_samples": float(array.size),
        "eog_nonfinite_fraction": float((array.size - n_finite) / array.size) if array.size else 1.0,
        "eog_n_gaps": float(max(len(segments) - 1, 0)),
        "eog_n_segments": float(len(segments)),
        "eog_features_valid": 0.0,
    }
    base.update(unit_feature_flags("eog", handling))
    for key in ("eog_std", "eog_peak_to_peak", "eog_slow_fast_ratio",
                "eog_blink_like_events", "eog_total_power"):
        base[key] = float("nan")
    if n_finite < 16 or not np.isfinite(fs) or fs <= 0:
        return base

    base.update(weighted_feature_mean([(s.size, _segment_features(s, fs)) for s in segments]))
    base["eog_features_valid"] = 1.0
    return base
