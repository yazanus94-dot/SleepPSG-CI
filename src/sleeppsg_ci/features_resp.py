"""Respiratory features from verified respiratory-effort or airflow channels."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .preprocessing import bandpass_filter, detrend_signal


def extract_resp_features(signal_array: np.ndarray, fs: float) -> dict[str, float]:
    """Estimate respiratory rate and amplitude variability from one channel."""
    array = np.asarray(signal_array, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    base: dict[str, float] = {
        "resp_n_samples": float(array.size),
        "resp_nonfinite_fraction": float((array.size - finite.size) / array.size) if array.size else 1.0,
        "resp_features_valid": 0.0,
    }
    nan_keys = ("resp_rate_bpm", "resp_amplitude_cv", "resp_signal_range", "resp_regularity")
    base.update({key: float("nan") for key in nan_keys})
    if finite.size < 64 or not np.isfinite(fs) or fs <= 0:
        return base

    try:
        filtered = bandpass_filter(finite, fs, 0.05, 2.0, order=2)
    except ValueError:
        filtered = detrend_signal(finite)
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)
    peaks, _ = signal.find_peaks(
        envelope,
        distance=max(1, int(round(1.5 * fs))),
        prominence=float(np.std(envelope)),
    )
    base["resp_signal_range"] = float(np.max(envelope) - np.min(envelope))

    if peaks.size >= 2:
        intervals = np.diff(peaks) / fs
        intervals = intervals[(intervals > 1.0) & (intervals < 30.0)]
        if intervals.size:
            base["resp_rate_bpm"] = float(60.0 / np.mean(intervals))
            base["resp_regularity"] = float(1.0 / (1.0 + np.std(intervals) / max(np.mean(intervals), 1e-9)))
    amplitudes = envelope[peaks]
    if amplitudes.size >= 2 and np.mean(amplitudes) > 0:
        base["resp_amplitude_cv"] = float(np.std(amplitudes) / np.mean(amplitudes))
    base["resp_features_valid"] = 1.0
    return base
