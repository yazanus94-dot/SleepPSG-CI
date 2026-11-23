"""ECG-derived heart-rate and HRV features (ECG only, never PPG/PRV)."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .preprocessing import bandpass_filter

MIN_RR_S = 0.30  # reject detections implying >200 bpm
MAX_RR_S = 2.00  # reject detections implying <30 bpm


def detect_r_peaks(signal_array: np.ndarray, fs: float) -> np.ndarray:
    """Pan-Tompkins-style R-peak detection on a verified ECG channel."""
    array = np.asarray(signal_array, dtype=float).reshape(-1)
    filtered = bandpass_filter(array, fs, 5.0, 15.0, order=2)
    differentiated = np.diff(filtered, prepend=filtered[0])
    squared = differentiated**2
    window = max(1, int(round(0.15 * fs)))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")
    threshold = float(np.mean(integrated) + 0.5 * np.std(integrated))
    distance = max(1, int(round(MIN_RR_S * fs)))
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=distance)
    return peaks


def rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Return plausible RR intervals in seconds, rejecting implausible beats."""
    if peaks.size < 2:
        return np.array([], dtype=float)
    rr = np.diff(peaks) / float(fs)
    return rr[(rr >= MIN_RR_S) & (rr <= MAX_RR_S)]


def extract_ecg_features(signal_array: np.ndarray, fs: float) -> dict[str, float]:
    """Return ECG HRV features with explicit beat-detection quality flags."""
    array = np.asarray(signal_array, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    base: dict[str, float] = {
        "ecg_n_samples": float(array.size),
        "ecg_nonfinite_fraction": float((array.size - finite.size) / array.size) if array.size else 1.0,
        "ecg_n_beats": 0.0,
        "ecg_features_valid": 0.0,
    }
    nan_keys = ("ecg_mean_hr", "ecg_median_hr", "ecg_mean_rr", "ecg_sdnn",
                "ecg_rmssd", "ecg_pnn50", "ecg_beat_quality")
    base.update({key: float("nan") for key in nan_keys})

    if finite.size < 64 or not np.isfinite(fs) or fs <= 0:
        return base
    try:
        peaks = detect_r_peaks(finite, fs)
        rr = rr_intervals(peaks, fs)
    except (ValueError, RuntimeError):
        return base
    if rr.size < 2:
        return base

    diffs = np.diff(rr)
    heart_rates = 60.0 / rr
    candidate_beats = max(peaks.size - 1, 1)
    base.update({
        "ecg_n_beats": float(peaks.size),
        "ecg_mean_hr": float(np.mean(heart_rates)),
        "ecg_median_hr": float(np.median(heart_rates)),
        "ecg_mean_rr": float(np.mean(rr)),
        "ecg_sdnn": float(np.std(rr, ddof=0)),
        "ecg_rmssd": float(np.sqrt(np.mean(diffs**2))),
        "ecg_pnn50": float(np.count_nonzero(np.abs(diffs) > 0.05) / diffs.size),
        "ecg_beat_quality": float(rr.size / candidate_beats),
        "ecg_features_valid": 1.0,
    })
    return base
