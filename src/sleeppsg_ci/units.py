"""Explicit, tested signal-unit harmonization.

EDF headers expose a physical dimension per channel. This module applies only
explicit conversion rules and records what was done. It never touches patient
identifiers, source identifiers or outcome variables.

Canonical units
---------------
EEG / EOG / EMG : microvolts (uV)
ECG             : millivolts (mV) — chosen canonical voltage unit for ECG
SpO2            : percent (%)
RESP / AIRFLOW  : no conversion; source-specific units are preserved and only
                  unit-invariant features (rate, variability, regularity) are
                  interpreted. ``resp_signal_range`` is unit-dependent and must
                  not be compared across sites with different sensor units.

Unknown or incompatible units are flagged, never silently assumed equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CANONICAL_UNITS: dict[str, str] = {
    "EEG": "uV",
    "EOG": "uV",
    "EMG": "uV",
    "ECG": "mV",
    "SPO2": "%",
    "RESP": "source-specific (no conversion)",
    "AIRFLOW": "source-specific (no conversion)",
}
VOLTAGE_MODALITIES = frozenset({"EEG", "EOG", "EMG", "ECG"})
UNCONVERTED_MODALITIES = frozenset({"RESP", "AIRFLOW"})
MODALITY_PREFIXES: dict[str, str] = {
    "EEG": "eeg", "ECG": "ecg", "RESP": "resp", "AIRFLOW": "airflow",
    "EMG": "emg", "EOG": "eog", "SPO2": "spo2",
}

# Structural bookkeeping is never a physiological amplitude predictor.
UNIT_STRUCTURAL_TOKENS = (
    "unit_", "missing_", "n_channels_", "n_verified", "signal_duration_seconds",
    "_n_samples", "_nonfinite_fraction", "_n_gaps", "_n_segments",
    "_features_valid", "_n_epochs", "_n_excluded_epochs", "_n_beats", "_beat_quality",
)
# Features whose numerical scale does not depend on the signal's physical unit.
# Each inclusion is scientifically justified; everything else is treated as
# amplitude-dependent and gated by unit provenance.
UNIT_INVARIANT_TOKENS = (
    "_relative", "_ratio", "spectral_entropy", "spectral_edge_95",
    "activity_fraction", "spectral_centroid", "slow_fast_ratio", "blink_like_events",
    "rate_bpm", "amplitude_cv", "regularity", "_mean_hr", "_median_hr", "_mean_rr",
    "sdnn", "rmssd", "pnn50", "below_90_fraction",
)

# Conversion factor from each recognised unit to microvolts.
_VOLTAGE_TO_UV: dict[str, float] = {
    "uv": 1.0, "µv": 1.0, "microv": 1.0, "microvolt": 1.0, "microvolts": 1.0,
    "mv": 1_000.0, "milliv": 1_000.0, "millivolt": 1_000.0, "millivolts": 1_000.0,
    "v": 1_000_000.0, "volt": 1_000_000.0, "volts": 1_000_000.0,
    "nv": 0.001, "nanov": 0.001, "nanovolt": 0.001, "nanovolts": 0.001,
}
_SPO2_PERCENT = frozenset({"%", "percent", "percentage", "pct"})
_SPO2_FRACTION = frozenset({"1", "ratio", "fraction", "frac", "normalized", "normalised"})


def normalize_unit_token(unit: str | None) -> str:
    """Lowercase/strip a physical-dimension token for matching."""
    return str(unit or "").strip().casefold()


@dataclass(frozen=True)
class UnitHandling:
    """Outcome of resolving one channel's source unit to a canonical unit."""

    modality: str
    original_unit: str
    canonical_unit: str
    scale_factor: float
    status: str  # native | converted | unconverted | unknown | flagged
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "original_unit": self.original_unit,
            "canonical_unit": self.canonical_unit,
            "scale_factor": self.scale_factor,
            "status": self.status,
            "detail": self.detail,
        }


def _unknown(modality: str, unit: str, detail: str) -> UnitHandling:
    return UnitHandling(
        modality=modality,
        original_unit=unit,
        canonical_unit=CANONICAL_UNITS.get(modality, "unknown"),
        scale_factor=1.0,
        status="unknown",
        detail=detail,
    )


def _fraction_supported(
    physical_min: float | None,
    physical_max: float | None,
    sample_values: np.ndarray | None,
) -> bool:
    """Decide whether a 'fraction' SpO2 unit really carries 0..1 values."""
    upper = physical_max
    lower = physical_min
    if upper is None and sample_values is not None:
        finite = np.asarray(sample_values, dtype=float).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            upper = float(np.max(finite))
            lower = float(np.min(finite))
    if upper is None:
        return False
    if not np.isfinite(upper) or upper > 1.5:
        return False
    return lower is None or (np.isfinite(lower) and lower >= -0.01)


def resolve_unit_conversion(
    modality: str,
    unit: str | None,
    *,
    physical_min: float | None = None,
    physical_max: float | None = None,
    sample_values: np.ndarray | None = None,
) -> UnitHandling:
    """Resolve a source unit to a canonical unit using explicit rules only."""
    modality = str(modality).upper()
    token = normalize_unit_token(unit)
    original = str(unit or "").strip() or "<blank>"

    if modality in VOLTAGE_MODALITIES:
        canonical = CANONICAL_UNITS[modality]
        factor_to_uv = _VOLTAGE_TO_UV.get(token)
        if factor_to_uv is None:
            return _unknown(modality, original, "unrecognized voltage unit; no conversion applied")
        factor = factor_to_uv if canonical == "uV" else factor_to_uv / 1_000.0
        status = "native" if np.isclose(factor, 1.0) else "converted"
        detail = f"{original} -> {canonical} (factor {factor:g})"
        return UnitHandling(modality, original, canonical, float(factor), status, detail)

    if modality == "SPO2":
        if token in _SPO2_PERCENT:
            return UnitHandling(modality, original, "%", 1.0, "native", "already percent")
        if token in _SPO2_FRACTION:
            if _fraction_supported(physical_min, physical_max, sample_values):
                return UnitHandling(
                    modality, original, "%", 100.0, "converted",
                    "fraction 0..1 -> percent (confirmed by header/values)",
                )
            return UnitHandling(
                modality, original, "%", 1.0, "flagged",
                "fraction unit but header/values do not support 0..1; no conversion applied",
            )
        return _unknown(modality, original, "unrecognized SpO2 unit; no conversion applied")

    if modality in UNCONVERTED_MODALITIES:
        return UnitHandling(
            modality, original, CANONICAL_UNITS[modality], 1.0, "unconverted",
            "source-specific unit preserved; only unit-invariant features interpreted",
        )

    return _unknown(modality, original, "no unit rule for this modality")


def apply_unit_conversion(values, handling: UnitHandling | None) -> np.ndarray:
    """Apply a resolved conversion factor; ``None`` leaves values untouched."""
    array = np.asarray(values, dtype=float)
    if handling is None:
        return array
    return array * handling.scale_factor


def unit_feature_flags(prefix: str, handling: UnitHandling | None) -> dict[str, float]:
    """Numeric unit-provenance flags.

    Keys use a ``unit_`` prefix so they are metadata, not physiological
    predictors: a model must not learn site/equipment conventions from them.
    """
    if handling is None:
        return {}
    return {
        f"unit_{prefix}_scale": float(handling.scale_factor),
        f"unit_{prefix}_native": float(handling.status == "native"),
        f"unit_{prefix}_converted": float(handling.status == "converted"),
        f"unit_{prefix}_unknown": float(handling.status in {"unknown", "flagged"}),
        f"unit_{prefix}_unconverted": float(handling.status == "unconverted"),
    }


def feature_modality(name: str) -> str | None:
    """Return the modality a feature column belongs to, or ``None``."""
    lower = str(name).casefold()
    for modality, prefix in MODALITY_PREFIXES.items():
        if lower.startswith(f"{prefix}_"):
            return modality
    return None


def is_unit_invariant_feature(name: str) -> bool:
    """True for structural or explicitly unit-invariant features.

    Band-power *ratios*, relative powers, entropy/edge frequency, time-domain
    HRV, activity fractions, event counts and respiratory rate/CV/regularity do
    not depend on the signal's amplitude scale. ``below_90_fraction`` is
    invariant only on a confirmed percent scale (see
    :func:`is_feature_eligible`).
    """
    lower = str(name).casefold()
    if any(token in lower for token in UNIT_STRUCTURAL_TOKENS):
        return True
    return any(token in lower for token in UNIT_INVARIANT_TOKENS)


def is_amplitude_dependent_feature(name: str) -> bool:
    """True when a feature's value scales with the physical signal amplitude."""
    return not is_unit_invariant_feature(name)


def is_feature_eligible(
    name: str,
    *,
    modality: str | None = None,
    unit_unknown: bool = False,
    unit_harmonized: bool = False,
) -> bool:
    """Decide whether one feature may enter a cross-site model matrix.

    Rules
    -----
    * Structural bookkeeping and explicitly invariant features stay eligible.
    * A SpO2 90%-threshold fraction needs a confirmed percent scale.
    * RESP/AIRFLOW ``*_signal_range`` is unit-dependent and eligible only when
      those units were explicitly harmonized (the default pipeline never does).
    * Amplitude-dependent EEG/EOG/EMG/ECG/SpO2 features are excluded whenever
      any contributing channel had an unknown or incompatible unit.
    """
    resolved = modality or feature_modality(name)
    lower = str(name).casefold()

    if is_unit_invariant_feature(name):
        if resolved == "SPO2" and unit_unknown and "below_90_fraction" in lower:
            return False
        return True

    if resolved in UNCONVERTED_MODALITIES and "_signal_range" in lower:
        return unit_harmonized

    if unit_unknown and (resolved in VOLTAGE_MODALITIES or resolved == "SPO2"):
        return False
    return True
