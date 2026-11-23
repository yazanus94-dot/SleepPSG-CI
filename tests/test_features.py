import numpy as np
import pytest

from sleeppsg_ci.features_eeg import band_powers, extract_eeg_features, welch_psd
from sleeppsg_ci.features_ecg import extract_ecg_features
from sleeppsg_ci.features_emg import extract_emg_features
from sleeppsg_ci.features_eog import extract_eog_features
from sleeppsg_ci.features_resp import extract_resp_features
from sleeppsg_ci.feature_table import extract_from_signals


def osc(freqs=(2, 10, 20), seconds=30.0, fs=256.0):
    t = np.arange(0, seconds, 1 / fs)
    return sum(np.sin(2 * np.pi * f * t) for f in freqs)


EXTRACTORS = (extract_eeg_features, extract_ecg_features, extract_resp_features,
              extract_emg_features, extract_eog_features)


def test_feature_shapes_finiteness_and_reproducibility():
    signal = osc()
    for extractor in EXTRACTORS:
        first = extractor(signal, 256.0)
        second = extractor(signal, 256.0)
        assert first == second
        assert isinstance(first, dict) and len(first) > 3
        for key, value in first.items():
            assert isinstance(value, float), (extractor.__name__, key)
            assert np.isfinite(value), (extractor.__name__, key)


def test_short_and_nan_inputs_do_not_crash():
    for extractor in EXTRACTORS:
        short = extractor(np.array([1.0, 2.0]), 256.0)
        assert short["eeg_features_valid" if extractor is extract_eeg_features else next(k for k in short if k.endswith("_features_valid"))] == 0.0
        noisy = extractor(np.array([1.0, np.nan, np.inf, 2.0, 3.0]), 256.0)
        assert isinstance(noisy, dict)


def test_eeg_bands_and_entropy():
    freqs, psd = welch_psd(osc(), 256.0)
    powers = band_powers(freqs, psd)
    assert set(powers) == {"delta_power", "theta_power", "alpha_power", "sigma_power", "beta_power"}
    features = extract_eeg_features(osc(), 256.0)
    assert features["eeg_features_valid"] == 1.0
    assert 0 <= features["eeg_spectral_entropy"] <= 1


def test_constant_signal_features_are_finite_or_nan_not_inf():
    for extractor in EXTRACTORS:
        result = extractor(np.full(1000, 5.0), 256.0)
        for value in result.values():
            assert not np.isinf(value)


def test_missing_modality_indicators_and_mapping():
    from sleeppsg_ci.channel_audit import ChannelRule

    rules = (
        ChannelRule("EEG", "EEG", verified=True),
        ChannelRule("ECG", "ECG", verified=True),
    )
    row = extract_from_signals(
        {"EEG": (osc(), 256.0), "UNVERIFIED": (osc(), 256.0)}, rules=rules
    )
    assert row["missing_eeg"] == 0.0
    assert row["missing_ecg"] == 1.0
    assert row["missing_eog"] == 1.0
    assert row["eeg_features_valid"] == 1.0
    assert row["n_verified_modalities"] == 1.0


def test_invalid_channel_modality_rejected():
    from sleeppsg_ci.channel_audit import ChannelMappingError, ChannelRule

    with pytest.raises(ChannelMappingError):
        ChannelRule("EEG", "NOT_A_MODALITY", verified=True)


def test_airflow_and_spo2_are_processed():
    from sleeppsg_ci.channel_audit import ChannelRule
    from sleeppsg_ci.features_spo2 import extract_spo2_features

    rules = (
        ChannelRule("AIR", "AIRFLOW", verified=True),
        ChannelRule("SPO2", "SPO2", verified=True),
    )
    row = extract_from_signals({
        "AIR": (osc(freqs=(0.25,), seconds=90.0), 64.0),
        "SPO2": (np.full(90 * 64, 96.0) + np.sin(np.arange(90 * 64) / 300), 64.0),
    }, rules=rules, epoch_seconds=30.0)
    assert row["missing_airflow"] == 0.0
    assert row["missing_spo2"] == 0.0
    assert row["n_verified_modalities"] == 2.0
    assert any(key.startswith("airflow_") for key in row)
    spo2 = extract_spo2_features(np.array([98.0, 97.0, 89.0, 95.0]), 1.0)
    assert spo2["spo2_min"] == 89.0
    assert 0 < spo2["spo2_below_90_fraction"] <= 0.25


def test_multiple_channels_are_retained_not_discarded():
    from sleeppsg_ci.channel_audit import ChannelRule

    rules = (
        ChannelRule("EEG1", "EEG", verified=True),
        ChannelRule("EEG2", "EEG", verified=True),
    )
    row = extract_from_signals({
        "EEG1": (osc(freqs=(2,), seconds=60.0), 256.0),
        "EEG2": (osc(freqs=(10,), seconds=60.0), 256.0),
    }, rules=rules, epoch_seconds=30.0)
    assert row["n_channels_eeg"] == 2.0
    assert any("ch_eeg1_" in key for key in row)
    assert any("ch_eeg2_" in key for key in row)
    assert row["missing_eeg"] == 0.0


def test_chunk_size_does_not_change_full_night_aggregates(tmp_path):
    import edfio
    from sleeppsg_ci.channel_audit import ChannelRule
    from sleeppsg_ci.feature_table import extract_from_edf

    fs = 128.0
    t = np.arange(0, 60.0, 1 / fs)
    data = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal = edfio.EdfSignal(data, fs, label="EEG", physical_dimension="uV",
                             physical_range=(-100, 100))
    path = tmp_path / "night.edf"
    edfio.Edf([signal]).write(path)
    rules = (ChannelRule("EEG", "EEG", verified=True),)

    small = extract_from_edf(path, rules=rules, chunk_seconds=10.0, epoch_seconds=30.0)
    large = extract_from_edf(path, rules=rules, chunk_seconds=30.0, epoch_seconds=30.0)
    assert small["eeg_n_epochs"] == large["eeg_n_epochs"] == 2.0
    assert small["signal_duration_seconds"] == pytest.approx(60.0)
    for key in ("eeg_delta_power", "eeg_eeg_features_valid" if "eeg_eeg_features_valid" in small else "eeg_features_valid"):
        assert small[key] == pytest.approx(large[key], rel=1e-9)


def test_nonfinite_gaps_are_not_compacted_for_ecg(monkeypatch):
    from sleeppsg_ci import features_ecg as ecg_mod

    fs = 256.0
    segment = np.zeros(256 * 3, dtype=float)
    for beat in range(3):
        index = 40 + beat * 128
        segment[index - 2:index + 3] = [0, 1, 5, 1, 0]
    gapped = np.concatenate([segment, np.full(256, np.nan), segment])

    calls: list[int] = []
    original = ecg_mod.detect_r_peaks

    def recording_detector(signal_array, rate):
        calls.append(int(signal_array.size))
        return original(signal_array, rate)

    monkeypatch.setattr(ecg_mod, "detect_r_peaks", recording_detector)
    features = ecg_mod.extract_ecg_features(gapped, fs)
    assert features["ecg_n_gaps"] == 1.0
    assert features["ecg_n_segments"] == 2.0
    assert len(calls) == 2  # R-peaks detected independently per segment
    assert all(np.isfinite(value) for value in features.values())


def test_nonfinite_gaps_are_not_compacted_for_resp(monkeypatch):
    from sleeppsg_ci import features_resp as resp_mod

    fs = 64.0
    segment = np.sin(np.arange(0, 60.0, 1 / fs) * 2 * np.pi * 0.25)
    gapped = np.concatenate([segment, np.full(64, np.nan), segment])

    calls: list[int] = []
    original = resp_mod._segment_features

    def recording_estimator(segment_array, rate):
        calls.append(int(segment_array.size))
        return original(segment_array, rate)

    monkeypatch.setattr(resp_mod, "_segment_features", recording_estimator)
    features = resp_mod.extract_resp_features(gapped, fs)
    assert features["resp_n_gaps"] == 1.0
    assert features["resp_n_segments"] == 2.0
    assert len(calls) == 2
