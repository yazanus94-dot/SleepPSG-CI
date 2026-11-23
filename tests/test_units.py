"""Explicit signal-unit harmonization and cross-unit equivalence tests."""

from __future__ import annotations

import numpy as np
import pytest

from sleeppsg_ci.channel_audit import ChannelRule, audit_recording
from sleeppsg_ci.feature_table import extract_from_edf
from sleeppsg_ci.features_eeg import extract_eeg_features
from sleeppsg_ci.features_ecg import extract_ecg_features
from sleeppsg_ci.features_emg import extract_emg_features
from sleeppsg_ci.features_eog import extract_eog_features
from sleeppsg_ci.features_resp import extract_resp_features
from sleeppsg_ci.features_spo2 import extract_spo2_features
from sleeppsg_ci.modeling import select_feature_matrix
from sleeppsg_ci.units import (
    CANONICAL_UNITS,
    apply_unit_conversion,
    resolve_unit_conversion,
    unit_feature_flags,
)


def oscillatory(fs=256.0, seconds=30.0, scale=50.0):
    t = np.arange(0, seconds, 1 / fs)
    return scale * (np.sin(2 * np.pi * 2 * t) + 0.4 * np.sin(2 * np.pi * 10 * t))


def ecg_like(fs=256.0, seconds=10.0, scale=1.0):
    signal = np.zeros(int(fs * seconds), dtype=float)
    for beat in range(10):
        index = 40 + beat * int(fs * 0.9)
        signal[index - 2:index + 3] = scale * np.array([0, 1, 5, 1, 0])
    return signal


def test_canonical_units_documented():
    assert CANONICAL_UNITS["EEG"] == "uV"
    assert CANONICAL_UNITS["EOG"] == "uV"
    assert CANONICAL_UNITS["EMG"] == "uV"
    assert CANONICAL_UNITS["ECG"] == "mV"
    assert CANONICAL_UNITS["SPO2"] == "%"
    assert "no conversion" in CANONICAL_UNITS["RESP"]


@pytest.mark.parametrize("extractor,modality_key", [
    (extract_eeg_features, "eeg"),
    (extract_emg_features, "emg"),
    (extract_eog_features, "eog"),
])
def test_voltage_unit_equivalence_for_uv_mv_v(extractor, modality_key):
    base = oscillatory()
    uv = extractor(base, 256.0, unit="uV")
    mv = extractor(base / 1000.0, 256.0, unit="mV")
    volts = extractor(base / 1_000_000.0, 256.0, unit="V")

    key = f"{modality_key}_n_samples"
    assert uv[key] == mv[key] == volts[key]
    spectral = [k for k in uv if k.endswith("_power") or k == f"{modality_key}_rms"
                or k == f"{modality_key}_std"]
    assert spectral, "expected at least one amplitude-dependent feature"
    for feature in spectral:
        if np.isfinite(uv[feature]):
            assert uv[feature] == pytest.approx(mv[feature], rel=1e-5), feature
            assert uv[feature] == pytest.approx(volts[feature], rel=1e-5), feature
    assert uv[f"unit_{modality_key}_native"] == 1.0
    assert mv[f"unit_{modality_key}_converted"] == 1.0
    assert volts[f"unit_{modality_key}_converted"] == 1.0


def test_ecg_voltage_unit_equivalence():
    base = ecg_like(scale=1.0)
    uv = extract_ecg_features(base * 1000.0, 256.0, unit="uV")
    mv = extract_ecg_features(base, 256.0, unit="mV")
    volts = extract_ecg_features(base / 1000.0, 256.0, unit="V")
    assert mv["ecg_n_beats"] >= 2
    for feature in ("ecg_mean_hr", "ecg_mean_rr", "ecg_sdnn"):
        assert np.isfinite(mv[feature])
        assert mv[feature] == pytest.approx(uv[feature], rel=1e-6), feature
        assert mv[feature] == pytest.approx(volts[feature], rel=1e-6), feature
    assert uv["unit_ecg_converted"] == 1.0


def test_spo2_fraction_converted_only_when_supported():
    fraction = np.array([0.98, 0.95, 0.89, 0.97])
    converted = extract_spo2_features(fraction, 1.0, unit="1", physical_limits=(0.0, 1.0))
    assert converted["spo2_mean"] == pytest.approx(94.75)
    assert converted["spo2_below_90_fraction"] == pytest.approx(0.25)
    assert converted["unit_spo2_converted"] == 1.0

    # Values already in percent, mislabelled as fraction with a percent range.
    flagged = extract_spo2_features(
        np.array([98.0, 95.0, 89.0]), 1.0, unit="ratio", physical_limits=(0.0, 100.0)
    )
    assert flagged["unit_spo2_unknown"] == 1.0
    assert flagged["spo2_mean"] == pytest.approx(94.0)  # left untouched, not x100

    native = extract_spo2_features(np.array([98.0, 95.0]), 1.0, unit="%")
    assert native["unit_spo2_native"] == 1.0
    assert native["spo2_mean"] == pytest.approx(96.5)


def test_unrecognized_units_are_flagged_not_assumed():
    handling = resolve_unit_conversion("EEG", "furlongs")
    assert handling.status == "unknown"
    assert handling.scale_factor == 1.0
    features = extract_eeg_features(oscillatory(), 256.0, unit="furlongs")
    assert features["unit_eeg_unknown"] == 1.0
    assert np.isfinite(features["eeg_delta_power"])
    assert apply_unit_conversion(np.array([1.0]), handling)[0] == 1.0


def test_respiratory_units_are_not_converted_and_flags_recorded():
    handling = resolve_unit_conversion("RESP", "a.u.")
    assert handling.status == "unconverted"
    assert handling.scale_factor == 1.0
    assert handling.canonical_unit == CANONICAL_UNITS["RESP"]

    signal = np.sin(np.arange(0, 60.0, 1 / 64) * 2 * np.pi * 0.25) * 7.0
    features = extract_resp_features(signal, 64.0, unit="a.u.")
    assert features["unit_resp_unconverted"] == 1.0
    assert features["unit_resp_scale"] == 1.0
    # Unit-invariant features are scale independent.
    scaled = extract_resp_features(signal / 7.0, 64.0, unit="a.u.")
    assert features["resp_rate_bpm"] == pytest.approx(scaled["resp_rate_bpm"], rel=1e-6)
    assert features["resp_amplitude_cv"] == pytest.approx(scaled["resp_amplitude_cv"], rel=1e-6)


def test_unit_flags_are_numeric_and_not_predictor_columns():
    handling = resolve_unit_conversion("EEG", "mV")
    flags = unit_feature_flags("eeg", handling)
    assert all(isinstance(value, float) for value in flags.values())
    frame = {**flags, "Age": 60.0, "eeg_delta_power": 1.0, "patient_key": "S0001_p1"}
    import pandas as pd

    matrix = select_feature_matrix(pd.DataFrame([frame]), "combined")
    assert "unit_eeg_converted" not in matrix.columns
    assert "patient_key" not in matrix.columns
    assert "eeg_delta_power" in matrix.columns


def test_synthetic_edf_unit_provenance_and_equivalence(tmp_path):
    import edfio

    t = np.arange(0, 60.0, 1 / 128.0)
    base = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 2 * t)
    rules = (ChannelRule("EEG", "EEG", verified=True),)
    paths = {}
    for name, values, unit, rng in (
        ("uv.edf", base, "uV", (-100.0, 100.0)),
        ("mv.edf", base / 1000.0, "mV", (-0.1, 0.1)),
    ):
        path = tmp_path / name
        edfio.Edf([edfio.EdfSignal(values, 128.0, label="EEG", physical_dimension=unit,
                                   physical_range=rng)]).write(path)
        paths[name] = path

    uv_row = extract_from_edf(paths["uv.edf"], rules=rules, chunk_seconds=15.0)
    mv_row = extract_from_edf(paths["mv.edf"], rules=rules, chunk_seconds=15.0)
    assert uv_row["unit_original_eeg_eeg"] == "uV"
    assert mv_row["unit_original_eeg_eeg"] == "mV"
    assert mv_row["unit_status_eeg_eeg"] == "converted"
    assert mv_row["unit_scale_eeg_eeg"] == pytest.approx(1000.0)
    assert uv_row["eeg_delta_power"] == pytest.approx(mv_row["eeg_delta_power"], rel=1e-5)
    assert "patient_key" not in uv_row and "BDSPPatientID" not in uv_row

    audit = audit_recording(paths["mv.edf"], rules=rules, inspect_samples=False)
    handling = audit.to_dict()["unit_handling"]["EEG"]
    assert handling["original_unit"] == "mV"
    assert handling["canonical_unit"] == "uV"
    assert handling["status"] == "converted"
