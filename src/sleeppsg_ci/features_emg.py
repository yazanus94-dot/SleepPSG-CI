"""EMG amplitude, burst and spectral features from verified EMG channels."""

from __future__ import annotations

import numpy as np

from .features_eeg import welch_psd
from .preprocessing import detrend_signal


def _segment_features(finite: np.ndarray, fs: float) -> dict[str, float]:
    centered = detrend_signal(finite) if finite.size > 2 else finite - np.mean(finite)
    result: dict[str, float] = {
        "emg_rms": float(np.sqrt(np.mean(centered**2))),
        "emg_mean_abs": float(np.mean(np.abs(centered))),
        "emg_peak_to_peak": float(np.max(centered) - np.min(centered)),
        "emg_activity_fraction": float(
            np.mean(np.abs(centered) > float(np.mean(np.abs(centered)) + 2.0 * np.std(centered)))
        ),
        "emg_high_freq_power": float("nan"),
        "emg_total_power": float("nan"),
        "emg_spectral_centroid": float("nan"),
    }
    try:
        freqs, psd = welch_psd(centered, fs)
        total = float(np.sum(psd))
        high = freqs >= 30.0
        result["emg_high_freq_power"] = float(np.sum(psd[high])) if high.any() else 0.0
        result["emg_total_power"] = total
        result["emg_spectral_centroid"] = float(np.sum(freqs * psd) / total) if total > 0 else float("nan")
    except ValueError:
        pass
    return result


def extract_emg_features(signal_array: np.ndarray, fs: float) -> dict[str, float]:
    """Return EMG features aggregated over contiguous finite segments."""
    from .features_common import weighted_feature_mean
    from .quality import contiguous_finite_segments

    array = np.asarray(signal_array, dtype=float).reshape(-1)
    segments = contiguous_finite_segments(array, min_length=16)
    n_finite = int(np.count_nonzero(np.isfinite(array)))
    base: dict[str, float] = {
        "emg_n_samples": float(array.size),
        "emg_nonfinite_fraction": float((array.size - n_finite) / array.size) if array.size else 1.0,
        "emg_n_gaps": float(max(len(segments) - 1, 0)),
        "emg_n_segments": float(len(segments)),
        "emg_features_valid": 0.0,
    }
    for key in ("emg_rms", "emg_mean_abs", "emg_peak_to_peak", "emg_activity_fraction",
                "emg_high_freq_power", "emg_total_power", "emg_spectral_centroid"):
        base[key] = float("nan")
    if n_finite < 16 or not np.isfinite(fs) or fs <= 0:
        return base

    base.update(weighted_feature_mean([(s.size, _segment_features(s, fs)) for s in segments]))
    base["emg_features_valid"] = 1.0
    return base
