"""EEG spectral features computed on verified EEG channels.

Bands follow common sleep-EEG practice: delta 0.5-4, theta 4-8, alpha 8-13,
sigma 12-16 and beta 13-30 Hz. Alpha/sigma/beta bands intentionally overlap in
the literature; both absolute and relative powers are reported so the choice is
transparent. Power is estimated with Welch's method on the documented sampling
rate; no resampling is assumed.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

BANDS_HZ: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (13.0, 30.0),
}


def welch_psd(signal_array: np.ndarray, fs: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return Welch PSD and frequencies for a 1-D signal."""
    array = np.asarray(signal_array, dtype=float).reshape(-1)
    if array.size < 16:
        raise ValueError("EEG signal too short for spectral estimation (need >= 16 samples).")
    segment = nperseg if nperseg is not None else min(512, array.size)
    segment = max(8, min(segment, array.size))
    freqs, psd = signal.welch(array, fs=fs, nperseg=segment, window="hann", detrend="constant")
    return freqs, psd


def band_powers(freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
    """Integrate PSD over each canonical band."""
    powers: dict[str, float] = {}
    for name, (low, high) in BANDS_HZ.items():
        mask = (freqs >= low) & (freqs < high)
        powers[f"{name}_power"] = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0
    return powers


def spectral_entropy(psd: np.ndarray) -> float:
    """Normalized Shannon entropy of the PSD (0 = single line, 1 = flat)."""
    total = float(np.sum(psd))
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    proportions = psd / total
    proportions = proportions[proportions > 0]
    entropy = float(-np.sum(proportions * np.log2(proportions)))
    max_entropy = float(np.log2(psd.size)) if psd.size > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else float("nan")


def spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, fraction: float = 0.95) -> float:
    """Frequency below which ``fraction`` of total spectral power lies."""
    total = float(np.sum(psd))
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    cumulative = np.cumsum(psd) / total
    index = int(np.searchsorted(cumulative, fraction))
    return float(freqs[min(index, freqs.size - 1)])


def extract_eeg_features(signal_array: np.ndarray, fs: float) -> dict[str, float]:
    """Return night-level EEG spectral features; degrade gracefully on bad input."""
    features: dict[str, float] = {}
    array = np.asarray(signal_array, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    features["eeg_n_samples"] = float(array.size)
    features["eeg_nonfinite_fraction"] = float((array.size - finite.size) / array.size) if array.size else 1.0

    if finite.size < 16 or not np.isfinite(fs) or fs <= 0:
        for name in (*(f"eeg_{b}_power" for b in BANDS_HZ), *(f"eeg_{b}_relative" for b in BANDS_HZ)):
            features[name] = float("nan")
        features.update({
            "eeg_spectral_entropy": float("nan"),
            "eeg_spectral_edge_95": float("nan"),
            "eeg_theta_beta_ratio": float("nan"),
            "eeg_alpha_delta_ratio": float("nan"),
            "eeg_features_valid": 0.0,
        })
        return features

    freqs, psd = welch_psd(finite, fs)
    absolute = band_powers(freqs, psd)
    for name, value in absolute.items():
        features[f"eeg_{name}"] = value
    total = sum(absolute.values())
    for name in BANDS_HZ:
        key = f"{name}_power"
        features[f"eeg_{name}_relative"] = absolute[key] / total if total > 0 else float("nan")

    features["eeg_spectral_entropy"] = spectral_entropy(psd)
    features["eeg_spectral_edge_95"] = spectral_edge_frequency(freqs, psd)
    beta = absolute["beta_power"]
    delta = absolute["delta_power"]
    features["eeg_theta_beta_ratio"] = absolute["theta_power"] / beta if beta > 0 else float("nan")
    features["eeg_alpha_delta_ratio"] = absolute["alpha_power"] / delta if delta > 0 else float("nan")
    features["eeg_features_valid"] = 1.0
    return features
