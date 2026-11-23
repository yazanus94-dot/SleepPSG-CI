"""Signal-quality checks: non-finite samples, flatlines, clipping and duration."""

from __future__ import annotations

import numpy as np


def as_1d_float(signal: np.ndarray) -> np.ndarray:
    """Return a 1-D float array or raise a clear error."""
    array = np.asarray(signal, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D signal, got shape {array.shape}.")
    return array


def check_finite(signal: np.ndarray) -> dict[str, float | int]:
    """Count non-finite samples without discarding them silently."""
    array = as_1d_float(signal)
    bad = int(np.count_nonzero(~np.isfinite(array)))
    return {
        "n_samples": int(array.size),
        "n_nonfinite": bad,
        "fraction_nonfinite": round(bad / array.size, 6) if array.size else 0.0,
    }


def detect_flatline(signal: np.ndarray, tol: float = 1e-9) -> dict[str, bool | float]:
    """Detect constant or near-constant recordings."""
    array = as_1d_float(signal)
    finite = array[np.isfinite(array)]
    if finite.size < 2:
        return {"is_flatline": True, "range": 0.0}
    dynamic_range = float(np.max(finite) - np.min(finite))
    return {"is_flatline": bool(dynamic_range <= tol), "range": dynamic_range}


def detect_clipping(
    signal: np.ndarray,
    physical_min: float | None = None,
    physical_max: float | None = None,
    tol: float = 1e-6,
    flag_fraction: float = 0.01,
) -> dict[str, bool | float]:
    """Detect saturation at EDF physical limits.

    When no physical limits are supplied, clipping is assessed against the
    observed extrema and therefore cannot be distinguished from a genuine
    constant signal; this is reported explicitly via ``limits_available``.
    """
    array = as_1d_float(signal)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"is_clipped": False, "fraction_at_limit": 0.0, "limits_available": physical_min is not None}

    low = physical_min if physical_min is not None else float(np.min(finite))
    high = physical_max if physical_max is not None else float(np.max(finite))
    at_low = np.count_nonzero(np.abs(finite - low) <= tol)
    at_high = np.count_nonzero(np.abs(finite - high) <= tol)
    fraction = (at_low + at_high) / finite.size
    return {
        "is_clipped": bool(fraction >= flag_fraction and physical_min is not None),
        "fraction_at_limit": round(float(fraction), 6),
        "limits_available": physical_min is not None and physical_max is not None,
    }


def signal_quality_report(
    signal: np.ndarray,
    fs: float | None = None,
    physical_min: float | None = None,
    physical_max: float | None = None,
) -> dict:
    """Aggregate per-channel quality summary used by the channel audit."""
    array = as_1d_float(signal)
    finite_check = check_finite(array)
    flatline = detect_flatline(array)
    clipping = detect_clipping(array, physical_min, physical_max)
    finite = array[np.isfinite(array)]
    std = float(np.std(finite)) if finite.size else 0.0
    duration = float(array.size / fs) if fs and fs > 0 else None

    reasons: list[str] = []
    if finite_check["n_nonfinite"]:
        reasons.append("non_finite_samples")
    if flatline["is_flatline"]:
        reasons.append("flatline")
    if clipping["is_clipped"]:
        reasons.append("clipping")
    if fs is not None and duration is not None and duration < 30.0:
        reasons.append("short_recording")

    return {
        **finite_check,
        **flatline,
        **clipping,
        "duration_seconds": duration,
        "std_amplitude": round(std, 6),
        "quality_ok": not reasons,
        "reasons": reasons,
    }


def require_usable_signal(signal: np.ndarray, min_samples: int = 2) -> np.ndarray:
    """Validate that a signal can be processed; raise on empty/short input."""
    array = as_1d_float(signal)
    if array.size < min_samples:
        raise ValueError(f"Signal too short: {array.size} samples, need >= {min_samples}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Signal contains NaN or infinite samples; handle quality flags first.")
    return array
