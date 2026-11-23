"""Deterministic, documented signal preprocessing primitives.

Filtering uses Butterworth second-order sections (``sos``) with zero-phase
``sosfiltfilt``. Edge handling is the SciPy default odd padding with a pad
length of ``3 * max(len(sos), 6)`` samples. Signalling is performed in single
channels and no record is silently discarded: every step is recorded in the
returned :class:`PreprocessResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .quality import require_usable_signal


def validate_fs(fs: float) -> float:
    """Validate a sampling frequency."""
    value = float(fs)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Sampling frequency must be finite and positive, got {fs}.")
    return value


def detrend_signal(signal_array: np.ndarray) -> np.ndarray:
    """Remove a linear trend (DC + slope) from the signal."""
    return signal.detrend(require_usable_signal(signal_array), type="linear", overwrite_data=False)


def baseline_correct(signal_array: np.ndarray) -> np.ndarray:
    """Subtract the signal mean (baseline correction)."""
    array = require_usable_signal(signal_array)
    return array - float(np.mean(array))


def bandpass_filter(
    signal_array: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter implemented as SOS.

    Raises if the requested band violates Nyquist or the signal is too short
    for the default filtfilt padding (no silent truncation).
    """
    rate = validate_fs(fs)
    array = require_usable_signal(signal_array)
    if not (0 < low_hz < high_hz < rate / 2):
        raise ValueError(
            f"Invalid band ({low_hz}, {high_hz}) Hz for fs={rate} Hz; require 0 < low < high < fs/2."
        )
    if order < 1:
        raise ValueError("Filter order must be >= 1.")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=rate, output="sos")
    padlen = 3 * (2 * sos.shape[0])
    if array.size <= padlen:
        raise ValueError(
            f"Signal too short for zero-phase filtering: {array.size} samples, need > {padlen}."
        )
    return signal.sosfiltfilt(sos, array)


def notch_filter(signal_array: np.ndarray, fs: float, freq_hz: float, quality: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch filter for mains-frequency interference."""
    rate = validate_fs(fs)
    array = require_usable_signal(signal_array)
    if not (0 < freq_hz < rate / 2):
        raise ValueError(f"Notch frequency must be in (0, fs/2); got {freq_hz} for fs={rate}.")
    if quality <= 0:
        raise ValueError("Notch quality factor must be positive.")
    b, a = signal.iirnotch(w0=freq_hz, Q=quality, fs=rate)
    # SciPy's filtfilt default padding is 3 * max(len(a), len(b)).
    padlen = 3 * max(len(a), len(b))
    if array.size <= padlen:
        raise ValueError(f"Signal too short for notch filtering: {array.size} samples.")
    return signal.filtfilt(b, a, array)


def resample_signal(signal_array: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample with an anti-aliasing FIR via ``scipy.signal.resample_poly``."""
    source = validate_fs(orig_fs)
    target = validate_fs(target_fs)
    array = require_usable_signal(signal_array)
    from math import gcd

    ratio = target / source
    divisor = gcd(int(round(target * 1000)), int(round(source * 1000))) or 1
    up = int(round(target * 1000)) // divisor
    down = int(round(source * 1000)) // divisor
    if abs(ratio - 1.0) < 1e-9:
        return array.copy()
    return signal.resample_poly(array, up, down)


def epoch_signal(signal_array: np.ndarray, fs: float, epoch_seconds: float = 30.0) -> np.ndarray:
    """Split a signal into fixed-duration, non-overlapping epochs.

    Incomplete trailing epochs are dropped and reported by the caller. A 30-second
    epoch matches the AASM scoring convention used by the challenge annotations.
    """
    rate = validate_fs(fs)
    array = require_usable_signal(signal_array, min_samples=1)
    if epoch_seconds <= 0:
        raise ValueError("epoch_seconds must be positive.")
    samples = int(round(rate * epoch_seconds))
    if samples < 1:
        raise ValueError("Epoch length is shorter than one sample.")
    n_epochs = array.size // samples
    if n_epochs < 1:
        raise ValueError(
            f"Signal shorter than one {epoch_seconds}s epoch ({array.size} samples at {rate} Hz)."
        )
    return array[: n_epochs * samples].reshape(n_epochs, samples)


@dataclass
class PreprocessConfig:
    """Declarative preprocessing recipe (all steps optional and documented)."""

    detrend: bool = True
    baseline: bool = False
    bandpass: tuple[float, float] | None = None
    notch_hz: float | None = None
    notch_quality: float = 30.0
    target_fs: float | None = None
    epoch_seconds: float | None = None
    filter_order: int = 4
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class PreprocessResult:
    """Processed signal plus the exact transformations applied."""

    signal: np.ndarray
    fs: float
    steps: list[str]
    n_dropped_samples: int = 0


def preprocess_signal(
    signal_array: np.ndarray,
    fs: float,
    config: PreprocessConfig,
) -> PreprocessResult:
    """Apply a configured preprocessing chain deterministically."""
    rate = validate_fs(fs)
    array = require_usable_signal(signal_array, min_samples=1)
    steps: list[str] = []

    if config.detrend:
        array = detrend_signal(array)
        steps.append("linear_detrend")
    if config.baseline:
        array = baseline_correct(array)
        steps.append("baseline_mean_subtraction")
    if config.bandpass is not None:
        low, high = config.bandpass
        array = bandpass_filter(array, rate, low, high, order=config.filter_order)
        steps.append(f"sos_butterworth_bandpass_{low}-{high}Hz_order{config.filter_order}")
    if config.notch_hz is not None:
        array = notch_filter(array, rate, config.notch_hz, config.notch_quality)
        steps.append(f"iir_notch_{config.notch_hz}Hz_Q{config.notch_quality}")
    if config.target_fs is not None:
        array = resample_signal(array, rate, config.target_fs)
        steps.append(f"resample_{rate}->{config.target_fs}Hz_polyphase")
        rate = validate_fs(config.target_fs)

    dropped = 0
    if config.epoch_seconds is not None:
        epochs = epoch_signal(array, rate, config.epoch_seconds)
        dropped = int(array.size - epochs.size)
        array = epochs.reshape(-1)
        steps.append(f"epoch_{config.epoch_seconds}s_drop_{dropped}_tail_samples")

    return PreprocessResult(signal=array, fs=rate, steps=steps, n_dropped_samples=dropped)
