import numpy as np
import pytest

from sleeppsg_ci.preprocessing import (
    PreprocessConfig,
    bandpass_filter,
    baseline_correct,
    detrend_signal,
    epoch_signal,
    notch_filter,
    preprocess_signal,
    resample_signal,
    validate_fs,
)


def test_short_signal_raises_without_scipy_padlen_error():
    """Signals shorter than sosfiltfilt's real padlen must fail with our message."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="too short"):
        bandpass_filter(rng.normal(size=13), 256.0, 5.0, 15.0, order=2)


def synthetic_signal(seconds=10.0, fs=256.0):
    t = np.arange(0, seconds, 1 / fs)
    return np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)


def test_sampling_rate_validation():
    validate_fs(256)
    with pytest.raises(ValueError):
        validate_fs(0)
    with pytest.raises(ValueError):
        validate_fs(float("nan"))


def test_invalid_band_and_short_signal():
    with pytest.raises(ValueError, match="Invalid band"):
        bandpass_filter(synthetic_signal(), 256, 100, 200)
    with pytest.raises(ValueError, match="too short"):
        bandpass_filter(np.arange(5.0), 256, 1, 40)


def test_detrend_baseline_notch_and_resample():
    signal = synthetic_signal()
    assert np.abs(np.mean(detrend_signal(signal))) < 1e-8
    assert np.abs(np.mean(baseline_correct(signal))) < 1e-8
    assert notch_filter(signal, 256, 50).shape == signal.shape
    assert resample_signal(signal, 256, 128).size == pytest.approx(signal.size / 2, rel=0.01)


def test_notch_validation():
    with pytest.raises(ValueError):
        notch_filter(synthetic_signal(), 256, 300)


def test_epoching_shapes_and_edge_cases():
    signal = synthetic_signal(seconds=65)
    epochs = epoch_signal(signal, 256, epoch_seconds=30)
    assert epochs.shape == (2, 30 * 256)
    with pytest.raises(ValueError):
        epoch_signal(signal[:100], 256, 30)
    with pytest.raises(ValueError):
        epoch_signal(signal, 256, 0)


def test_full_pipeline_is_deterministic():
    config = PreprocessConfig(
        detrend=True, baseline=True, bandpass=(1, 40), notch_hz=50,
        target_fs=128, epoch_seconds=30,
    )
    first = preprocess_signal(synthetic_signal(seconds=65), 256, config)
    second = preprocess_signal(synthetic_signal(seconds=65), 256, config)
    assert np.allclose(first.signal, second.signal)
    assert first.fs == 128
    assert "linear_detrend" in first.steps
    assert first.n_dropped_samples >= 0
