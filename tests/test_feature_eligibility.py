"""Unit-aware predictor eligibility: unsafe amplitude features never reach models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleeppsg_ci.channel_audit import ChannelRule
from sleeppsg_ci.feature_table import extract_from_edf
from sleeppsg_ci.modeling import (
    build_baselines,
    feature_kind_for,
    fit_predict,
    make_model,
    select_feature_matrix,
)
from sleeppsg_ci.units import (
    is_amplitude_dependent_feature,
    is_feature_eligible,
    is_unit_invariant_feature,
)

VOLTAGE_MODALITIES = ("eeg", "ecg", "eog", "emg")


def voltage_frame(modality: str, unknown: int = 1):
    """Synthetic cohort with one amplitude and several invariant features."""
    n = 12
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "SiteID": ["S0001"] * n,
        "BDSPPatientID": [f"p{i}" for i in range(n)],
        "Age": rng.integers(50, 80, n),
        "BMI": rng.normal(27, 3, n),
        "Sex": rng.choice(["Male", "Female"], n),
        "Cognitive_Impairment": rng.integers(0, 2, n),
        "missing_" + modality: 0.0,
        f"unit_unknown_channels_{modality}": float(unknown),
        f"{modality}_n_samples": float(256 * 30),
        f"{modality}_ch_CH1_{modality}_delta_power": rng.normal(1, 0.2, n),
    })
    # Invariant features common to most modality extractors.
    frame[f"{modality}_theta_beta_ratio__median"] = rng.normal(1, 0.1, n)
    frame[f"{modality}_features_valid"] = 1.0
    if modality == "eeg":
        frame["eeg_delta_power"] = rng.normal(1, 0.2, n)
        frame["eeg_delta_relative"] = rng.uniform(0, 1, n)
        frame["eeg_spectral_entropy"] = rng.uniform(0, 1, n)
    elif modality == "emg":
        frame["emg_rms"] = rng.normal(2, 0.3, n)
        frame["emg_activity_fraction"] = rng.uniform(0, 1, n)
    elif modality == "eog":
        frame["eog_std"] = rng.normal(3, 0.4, n)
        frame["eog_slow_fast_ratio"] = rng.uniform(0, 2, n)
    else:
        frame["ecg_mean_hr"] = rng.normal(65, 5, n)
        frame["ecg_sdnn"] = rng.uniform(10, 60, n)
    return frame


@pytest.mark.parametrize("modality", VOLTAGE_MODALITIES)
def test_unknown_voltage_unit_drops_amplitude_keeps_invariant(modality):
    frame = voltage_frame(modality, unknown=1)
    matrix = select_feature_matrix(frame, "signal")
    columns = list(matrix.columns)

    assert frame[f"unit_unknown_channels_{modality}"].max() == 1.0
    assert f"{modality}_theta_beta_ratio__median" in columns
    assert f"{modality}_n_samples" in columns
    assert f"missing_{modality}" in columns
    assert f"{modality}_ch_CH1_{modality}_delta_power" not in columns
    if modality == "eeg":
        assert "eeg_delta_power" not in columns
        assert "eeg_delta_relative" in columns
        assert "eeg_spectral_entropy" in columns
    elif modality == "emg":
        assert "emg_rms" not in columns
        assert "emg_activity_fraction" in columns
    elif modality == "eog":
        assert "eog_std" not in columns
        assert "eog_slow_fast_ratio" in columns
    else:
        assert "ecg_mean_hr" in columns
        assert "ecg_sdnn" in columns


@pytest.mark.parametrize("modality", VOLTAGE_MODALITIES)
def test_known_unit_keeps_amplitude_features(modality):
    frame = voltage_frame(modality, unknown=0)
    columns = list(select_feature_matrix(frame, "signal").columns)
    assert f"{modality}_ch_CH1_{modality}_delta_power" in columns
    if modality == "eeg":
        assert "eeg_delta_power" in columns


def test_resp_airflow_signal_range_excluded_unless_harmonized():
    n = 10
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "SiteID": ["S0001"] * n,
        "BDSPPatientID": [f"p{i}" for i in range(n)],
        "Age": rng.integers(50, 80, n),
        "Cognitive_Impairment": rng.integers(0, 2, n),
        "unit_harmonized_resp": 0.0,
        "unit_harmonized_airflow": 0.0,
        "resp_signal_range": rng.normal(1, 0.1, n),
        "resp_rate_bpm": rng.normal(15, 2, n),
        "resp_amplitude_cv": rng.uniform(0, 1, n),
        "resp_regularity": rng.uniform(0, 1, n),
        "resp_ch_CH1_resp_signal_range": rng.normal(1, 0.1, n),
        "airflow_signal_range": rng.normal(1, 0.1, n),
        "airflow_rate_bpm": rng.normal(15, 2, n),
    })
    columns = list(select_feature_matrix(frame, "signal").columns)
    for excluded in ("resp_signal_range", "resp_ch_CH1_resp_signal_range", "airflow_signal_range"):
        assert excluded not in columns
    for kept in ("resp_rate_bpm", "resp_amplitude_cv", "resp_regularity", "airflow_rate_bpm"):
        assert kept in columns

    harmonized = frame.copy()
    harmonized["unit_harmonized_resp"] = 1.0
    harmonized["unit_harmonized_airflow"] = 1.0
    columns = list(select_feature_matrix(harmonized, "signal").columns)
    assert "resp_signal_range" in columns
    assert "airflow_signal_range" in columns


def test_spo2_unknown_unit_drops_nonstructural_features():
    n = 10
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "SiteID": ["S0001"] * n,
        "BDSPPatientID": [f"p{i}" for i in range(n)],
        "Age": rng.integers(50, 80, n),
        "Cognitive_Impairment": rng.integers(0, 2, n),
        "unit_unknown_channels_spo2": 1.0,
        "spo2_n_samples": float(256 * 30),
        "spo2_mean": rng.normal(97, 1, n),
        "spo2_below_90_fraction": rng.uniform(0, 0.1, n),
    })
    columns = list(select_feature_matrix(frame, "signal").columns)
    assert "spo2_n_samples" in columns
    assert "spo2_mean" not in columns
    assert "spo2_below_90_fraction" not in columns


def test_unit_provenance_columns_never_enter_matrix():
    n = 8
    frame = pd.DataFrame({
        "SiteID": ["S0001"] * n,
        "BDSPPatientID": [f"p{i}" for i in range(n)],
        "Age": [60] * n,
        "Cognitive_Impairment": [i % 2 for i in range(n)],
        "unit_original_eeg_EEG": ["furlongs"] * n,
        "unit_status_eeg_EEG": ["unknown"] * n,
        "unit_harmonized_eeg": [0.0] * n,
        "unit_unknown_channels_eeg": [1.0] * n,
        "eeg_delta_power": np.arange(n, dtype=float),
    })
    for kind in ("signal", "combined"):
        columns = list(select_feature_matrix(frame, kind).columns)
        assert all(not column.startswith("unit_") for column in columns)
        assert "eeg_delta_power" not in columns


def test_signal_only_and_combined_models_fit_with_eligible_features():
    frame = voltage_frame("eeg", unknown=1)
    frame["resp_rate_bpm"] = np.linspace(12, 18, len(frame))
    frame["unit_harmonized_resp"] = 0.0
    y = frame["Cognitive_Impairment"].to_numpy(dtype=int)

    for name in ("signal_lr", "signal_rf", "combined_lr", "combined_hgb"):
        model = build_baselines()[name]
        matrix = select_feature_matrix(frame, feature_kind_for(name))
        probabilities = fit_predict(model, matrix, y, matrix)
        assert probabilities.shape == (len(frame),)
        assert np.all(np.isfinite(probabilities))
        assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_mixed_provenance_is_conservative_and_imputer_handles_missing():
    frame = voltage_frame("eeg", unknown=0)
    frame.loc[0, "unit_unknown_channels_eeg"] = 1.0  # one bad channel disqualifies all
    frame.loc[1, "eeg_delta_relative"] = np.nan  # eligible feature with a gap
    matrix = select_feature_matrix(frame, "combined")
    assert "eeg_delta_power" not in matrix.columns
    assert "eeg_delta_relative" in matrix.columns
    assert np.isnan(matrix.loc[1, "eeg_delta_relative"])

    model = make_model("combined_lr")
    probabilities = fit_predict(
        model, matrix, frame["Cognitive_Impairment"].to_numpy(dtype=int), matrix
    )
    assert np.all(np.isfinite(probabilities))
    assert np.all(np.isfinite(model.named_steps["imputer"].statistics_))


def test_feature_table_records_harmonization_status(tmp_path):
    import edfio

    fs = 128.0
    t = np.arange(0, 60.0, 1 / fs)
    eeg = np.sin(2 * np.pi * 10 * t)
    resp = np.sin(2 * np.pi * 0.25 * t)
    path = tmp_path / "mixed.edf"
    edfio.Edf([
        edfio.EdfSignal(eeg, fs, label="EEG", physical_dimension="uV", physical_range=(-100, 100)),
        edfio.EdfSignal(resp, fs, label="RESP", physical_dimension="a.u.", physical_range=(-1, 1)),
    ]).write(path)
    rules = (ChannelRule("EEG", "EEG", verified=True), ChannelRule("RESP", "RESP", verified=True))

    row = extract_from_edf(path, rules=rules, chunk_seconds=15.0)
    assert row["unit_harmonized_eeg"] == 1.0
    assert row["unit_harmonized_resp"] == 0.0
    assert row["unit_unknown_channels_eeg"] == 0.0

    # EEG amplitude features stay eligible; respiratory range is not.
    frame = pd.DataFrame([row])
    columns = list(select_feature_matrix(frame, "signal").columns)
    assert any(column.startswith("eeg_") and "power" in column for column in columns)
    assert not any("signal_range" in column for column in columns)


def test_unknown_unit_in_edf_is_flagged_and_its_amplitude_is_gated(tmp_path):
    import edfio

    fs = 128.0
    t = np.arange(0, 60.0, 1 / fs)
    eeg = np.sin(2 * np.pi * 10 * t)
    path = tmp_path / "unknown.edf"
    edfio.Edf([
        edfio.EdfSignal(eeg, fs, label="EEG", physical_dimension="furlongs",
                        physical_range=(-100, 100)),
    ]).write(path)
    rules = (ChannelRule("EEG", "EEG", verified=True),)

    row = extract_from_edf(path, rules=rules, chunk_seconds=15.0)
    assert row["unit_original_eeg_eeg"] == "furlongs"
    assert row["unit_status_eeg_eeg"] == "unknown"
    assert row["unit_unknown_channels_eeg"] == 1.0
    assert row["unit_scale_eeg_eeg"] == 1.0  # no silent conversion

    columns = list(select_feature_matrix(pd.DataFrame([row]), "signal").columns)
    assert not any("power" in column for column in columns)
    assert any("relative" in column or "entropy" in column for column in columns)


def test_classification_helpers():
    assert is_amplitude_dependent_feature("eeg_delta_power")
    assert not is_amplitude_dependent_feature("eeg_delta_relative")
    assert is_unit_invariant_feature("resp_rate_bpm")
    assert not is_feature_eligible(
        "resp_signal_range", modality="RESP", unit_unknown=False, unit_harmonized=False
    )
    assert is_feature_eligible(
        "resp_signal_range", modality="RESP", unit_unknown=False, unit_harmonized=True
    )
