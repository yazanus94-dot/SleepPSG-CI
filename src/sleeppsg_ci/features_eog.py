"""EOG amplitude, blink-like event and spectral features."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .features_eeg import welch_psd
from .preprocessing import detrend_signal


def extract_eog_features(signal_array: np.ndarray, fs: float) -> dict[str, float]:
    """Return EOG features from a verified EOG channel."""
    array = np.asarray(signal_array, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    base: dict[str, float] = {
        "eog_n_samples": float(array.size),
        "eog_nonfinite_fraction": float((array.size - finite.size) / array.size) if array.size else 1.0,
        "eog_features_valid": 0.0,
    }
    nan_keys = ("eog_std", "eog_peak_to_peak", "eog_slow_fast_ratio",
                "eog_blink_like_events", "eog_total_power")
    base.update({key: float("nan") for key in nan_keys})
    if finite.size < 16 or not np.isfinite(fs) or fs <= 0:
        return base

    centered = detrend_signal(finite) if finite.size > 2 else finite - np.mean(finite)
    base["eog_std"] = float(np.std(centered))
    base["eog_peak_to_peak"] = float(np.max(centered) - np.min(centered))
    try:
        freqs, psd = welch_psd(centered, fs)
        total = float(np.sum(psd))
        slow = (freqs >= 0.1) & (freqs < 4.0)
        fast = (freqs >= 4.0) & (freqs < 16.0)
        slow_power = float(np.sum(psd[slow]))
        fast_power = float(np.sum(psd[fast]))
        base["eog_slow_fast_ratio"] = slow_power / fast_power if fast_power > 0 else float("nan")
        base["eog_total_power"] = total
        envelope = np.abs(signal.hilbert(centered))
        peaks, _ = signal.find_peaks(
            envelope,
            height=float(np.mean(envelope) + 2.0 * np.std(envelope)),
            distance=max(1, int(round(0.2 * fs))),
        )
        base["eog_blink_like_events"] = float(peaks.size)
    except ValueError:
        pass
    base["eog_features_valid"] = 1.0
    return base
