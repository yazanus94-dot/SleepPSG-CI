"""EMG amplitude, burst and spectral features from verified EMG channels."""

from __future__ import annotations

import numpy as np

from .features_eeg import welch_psd
from .preprocessing import detrend_signal


def extract_emg_features(signal_array: np.ndarray, fs: float) -> dict[str, float]:
    """Return EMG features with amplitude- and activity-based summaries."""
    array = np.asarray(signal_array, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    base: dict[str, float] = {
        "emg_n_samples": float(array.size),
        "emg_nonfinite_fraction": float((array.size - finite.size) / array.size) if array.size else 1.0,
        "emg_features_valid": 0.0,
    }
    nan_keys = ("emg_rms", "emg_mean_abs", "emg_peak_to_peak", "emg_activity_fraction",
                "emg_high_freq_power", "emg_total_power", "emg_spectral_centroid")
    base.update({key: float("nan") for key in nan_keys})
    if finite.size < 16 or not np.isfinite(fs) or fs <= 0:
        return base

    centered = detrend_signal(finite) if finite.size > 2 else finite - np.mean(finite)
    base["emg_rms"] = float(np.sqrt(np.mean(centered**2)))
    base["emg_mean_abs"] = float(np.mean(np.abs(centered)))
    base["emg_peak_to_peak"] = float(np.max(centered) - np.min(centered))
    threshold = float(np.mean(np.abs(centered)) + 2.0 * np.std(centered))
    base["emg_activity_fraction"] = float(np.mean(np.abs(centered) > threshold))
    try:
        freqs, psd = welch_psd(centered, fs)
        total = float(np.sum(psd))
        high = freqs >= 30.0
        base["emg_high_freq_power"] = float(np.sum(psd[high])) if high.any() else 0.0
        base["emg_total_power"] = total
        base["emg_spectral_centroid"] = float(np.sum(freqs * psd) / total) if total > 0 else float("nan")
    except ValueError:
        pass
    base["emg_features_valid"] = 1.0
    return base
