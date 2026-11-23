import numpy as np
import pytest

from sleeppsg_ci.quality import (
    check_finite,
    detect_clipping,
    detect_flatline,
    require_usable_signal,
    signal_quality_report,
)


def test_empty_and_short_signals():
    with pytest.raises(ValueError):
        check_finite(np.array([]).reshape(0, 1))
    with pytest.raises(ValueError, match="too short"):
        require_usable_signal(np.array([1.0]))


def test_constant_signal_is_flatline():
    report = detect_flatline(np.full(100, 3.0))
    assert report["is_flatline"] is True
    assert detect_flatline(np.linspace(0, 1, 100))["is_flatline"] is False


def test_nan_and_infinite_samples_counted():
    signal = np.array([1.0, 2.0, np.nan, np.inf, 4.0])
    report = check_finite(signal)
    assert report["n_nonfinite"] == 2
    assert report["fraction_nonfinite"] == pytest.approx(0.4)


def test_clipping_requires_physical_limits():
    signal = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
    assert detect_clipping(signal, 0.0, 1.0)["is_clipped"] is True
    assert detect_clipping(signal)["is_clipped"] is False


def test_short_recording_flagged():
    report = signal_quality_report(np.arange(100.0), fs=100.0)
    assert "short_recording" in report["reasons"]
    assert report["duration_seconds"] == pytest.approx(1.0)
