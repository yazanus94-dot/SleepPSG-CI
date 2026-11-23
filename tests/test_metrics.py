import numpy as np
import pytest

from sleeppsg_ci.metrics import (
    bootstrap_ci,
    calibration_slope_intercept,
    compute_auroc,
    compute_auroc_age,
    compute_auroc_weighted,
    compute_prevalence,
    confusion_counts,
    score_predictions,
    sensitivity,
    specificity,
)


def cohort():
    labels = np.array([1, 0, 1, 0, 1, 0])
    probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.4])
    ages = np.array([60, 61, 70, 71, 80, 81])
    return labels, probs, ages


def test_official_age_conditioned_auroc_perfect_and_inverted():
    labels, probs, ages = cohort()
    assert compute_auroc_age(labels, probs, ages) == pytest.approx(1.0)
    assert compute_auroc_age(labels, 1 - probs, ages) == pytest.approx(0.0)
    assert compute_auroc(labels, probs) == pytest.approx(1.0)


def test_single_class_is_not_scoreable():
    assert np.isnan(compute_auroc_age(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9]), np.array([60, 61, 62])))
    assert np.isnan(compute_auroc(np.array([0, 0]), np.array([0.2, 0.8])))
    result = score_predictions([1, 1], [0.2, 0.8], [60, 61])
    assert result["scoreable"] is False
    assert np.isnan(result["auroc_age"])


def test_unequal_lengths_rejected():
    with pytest.raises(ValueError):
        compute_auroc_age([1, 0], [0.9], [60, 61])


def test_confusion_and_rates():
    labels = np.array([1, 1, 0, 0])
    binary = np.array([1, 0, 1, 0])
    assert confusion_counts(labels, binary) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert sensitivity(labels, binary) == 0.5
    assert specificity(labels, binary) == 0.5


def test_calibration_and_weighted_auroc_finite():
    labels, probs, ages = cohort()
    slope, intercept = calibration_slope_intercept(labels, probs)
    assert np.isfinite(slope) and np.isfinite(intercept)
    assert np.isfinite(compute_auroc_weighted(labels, probs, ages))


def test_prevalence_allows_different_cohort_sizes():
    # Evaluation ages and training prevalence vectors need not be equal length.
    mapping = compute_prevalence([60, 61], [1, 0, 1], [60, 61, 62], gap=2)
    assert set(mapping) == {60.0, 61.0}
    assert all(0 <= value <= 1 for value in mapping.values())


def test_score_predictions_full_suite():
    labels, probs, ages = cohort()
    result = score_predictions(labels, probs, ages, prevalence_labels=labels, prevalence_ages=ages)
    assert result["n"] == 6
    assert result["auroc_age"] == pytest.approx(1.0)
    assert result["auprc"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(0.0583, abs=0.001)
    assert np.isfinite(result["reward"])
    assert result["confusion_matrix"]["tp"] >= 0


def test_invalid_labels_rejected():
    with pytest.raises(ValueError, match="binary"):
        score_predictions([1, 0, -1], [0.1, 0.2, 0.3], [60, 61, 62])


def test_bootstrap_deterministic():
    labels, probs, ages = cohort()
    first = bootstrap_ci(compute_auroc_age, labels, probs, ages, n_boot=30, seed=7)
    second = bootstrap_ci(compute_auroc_age, labels, probs, ages, n_boot=30, seed=7)
    assert first == second
    assert first["n_valid"] > 0
