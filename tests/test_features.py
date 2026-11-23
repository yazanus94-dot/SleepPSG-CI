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
