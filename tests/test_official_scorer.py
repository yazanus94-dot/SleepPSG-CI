"""Differential tests against the official 2026 scorer implementation.

The ``_official_*`` functions below are copied verbatim from the public
``evaluate_model.py`` of https://github.com/physionetchallenges/python-example-2026
at commit c0bcd78ddb892290b9d218be9be19660f4d8bdf3 (BSD 3-Clause, Copyright (c)
2026 George B. Moody PhysioNet Challenges). They are kept inside the test suite
so CI can verify numerical equivalence without network access.

Documented difference: the official ``compute_auroc_age`` raises ZeroDivisionError
when no positive/negative pair lies within the age gap; this project deliberately
returns NaN and reports such a cohort as "not scoreable".
"""

from collections import defaultdict

import numpy as np
import pytest

from sleeppsg_ci.metrics import (
    compute_auroc_age,
    compute_auroc_weighted,
    compute_prevalence,
    compute_reward,
)

GAP = 2


def _official_compute_auroc_age(labels, predictions, ages, gap=0):
    m, n, o = len(labels), len(predictions), len(ages)
    assert m == n == o
    idx_pos = [i for i in range(m) if labels[i] == 1]
    idx_neg = [i for i in range(m) if labels[i] == 0]
    numer = 0
    denom = 0
    for i in range(len(idx_pos)):
        for j in range(len(idx_neg)):
            if abs(ages[idx_pos[i]] - ages[idx_neg[j]]) <= gap:
                if predictions[idx_pos[i]] > predictions[idx_neg[j]]:
                    numer += 1
                elif predictions[idx_pos[i]] == predictions[idx_neg[j]]:
                    numer += 0.5
                denom += 1
    return numer / denom


def _official_compute_auroc_weighted(labels, predictions, ages, gap=0):
    m, n, o = len(labels), len(predictions), len(ages)
    assert m == n == o
    finite_ages = ages[np.isfinite(ages)]
    range_ages = np.arange(np.min(finite_ages) - gap, np.max(finite_ages) + gap + 1)
    idx_pos = [i for i in range(m) if labels[i] == 1]
    idx_neg = [i for i in range(m) if labels[i] == 0]
    numer = np.zeros(len(range_ages))
    denom = np.zeros(len(range_ages))
    for k in range(len(range_ages)):
        for i in range(len(idx_pos)):
            for j in range(len(idx_neg)):
                if (abs(ages[idx_pos[i]] - range_ages[k]) <= gap
                        and abs(ages[idx_neg[j]] - range_ages[k]) <= gap):
                    if predictions[idx_pos[i]] > predictions[idx_neg[j]]:
                        numer[k] += 1
                    elif predictions[idx_pos[i]] == predictions[idx_neg[j]]:
                        numer[k] += 0.5
                    denom[k] += 1
    weights = np.zeros(len(range_ages))
    for k in range(len(range_ages)):
        weights[k] = sum(1 for i in range(len(finite_ages)) if abs(finite_ages[i] - range_ages[k]) <= gap)
    for k in range(len(range_ages)):
        if denom[k] == 0:
            weights[k] = 0
            numer[k] = 0
            denom[k] = 1
    weights = weights / np.sum(weights)
    return np.sum(weights * (numer / denom))


def _official_compute_prevalence(ages, prevalence_labels, prevalence_ages, gap=0):
    unique_ages = np.unique(ages[np.isfinite(ages)])
    age_to_labels = defaultdict(list)
    for age in unique_ages:
        for i in range(len(prevalence_labels)):
            if abs(age - prevalence_ages[i]) <= gap:
                age_to_labels[age].append(prevalence_labels[i])
    return {
        age: max(sum(age_to_labels[age]), 0.5) / len(age_to_labels[age])
        for age in age_to_labels
    }


def _official_compute_reward(labels, predictions, ages, age_to_prevalence):
    n = len(predictions)
    scores = np.zeros(n)
    num_scores = 0
    for i in range(n):
        if np.isfinite(ages[i]):
            p = age_to_prevalence[ages[i]]
            p = min(max(p, 0.5 / n), 1 - 0.5 / n)
            if labels[i] == 1 and predictions[i] == 1:
                scores[i] = 1 / p - 1
            elif labels[i] == 0 and predictions[i] == 1:
                scores[i] = -1
            if labels[i] == 1 and predictions[i] == 0:
                scores[i] = -1
            elif labels[i] == 0 and predictions[i] == 0:
                scores[i] = 1 / (1 - p) - 1
            num_scores += 1
    return np.sum(scores) / num_scores


def _fixture(seed=11):
    rng = np.random.default_rng(seed)
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    ages = np.array([60, 61, 62, 63, 64, 65, 80, 81], dtype=float)
    predictions = np.round(rng.uniform(0, 1, labels.size), 2)
    return labels, predictions, ages


def test_age_conditioned_auroc_matches_official_delta_2():
    labels, predictions, ages = _fixture()
    assert compute_auroc_age(labels, predictions, ages, gap=GAP) == pytest.approx(
        _official_compute_auroc_age(labels, predictions, ages, gap=GAP)
    )


def test_age_conditioned_auroc_ties_count_one_half():
    labels = np.array([1, 0, 1, 0])
    predictions = np.array([0.5, 0.5, 0.5, 0.5])
    ages = np.array([60, 61, 62, 63], dtype=float)
    assert compute_auroc_age(labels, predictions, ages, gap=GAP) == pytest.approx(0.5)
    assert compute_auroc_age(labels, predictions, ages, gap=GAP) == pytest.approx(
        _official_compute_auroc_age(labels, predictions, ages, gap=GAP)
    )


def test_no_comparable_age_pairs_difference_is_explicit():
    labels = np.array([1, 0])
    predictions = np.array([0.9, 0.1])
    ages = np.array([40, 80], dtype=float)
    with pytest.raises(ZeroDivisionError):
        _official_compute_auroc_age(labels, predictions, ages, gap=GAP)
    assert np.isnan(compute_auroc_age(labels, predictions, ages, gap=GAP))


def test_age_weighted_auroc_matches_official():
    labels, predictions, ages = _fixture()
    assert compute_auroc_weighted(labels, predictions, ages, gap=GAP) == pytest.approx(
        _official_compute_auroc_weighted(labels, predictions, ages, gap=GAP)
    )


def test_age_specific_prevalence_matches_official():
    labels, _, ages = _fixture()
    prevalence_labels = np.array([1, 0, 1, 0, 0, 1], dtype=int)
    prevalence_ages = np.array([59, 60, 62, 64, 79, 81], dtype=float)
    ours = compute_prevalence(ages, prevalence_labels, prevalence_ages, gap=GAP)
    official = _official_compute_prevalence(ages, prevalence_labels, prevalence_ages, gap=GAP)
    assert set(ours) == set(official)
    for key in official:
        assert ours[key] == pytest.approx(official[key])


def test_reward_matches_official():
    labels, probabilities, ages = _fixture()
    binary = (probabilities >= 0.5).astype(int)
    prevalence_labels = np.array([1, 0, 1, 0, 0, 1], dtype=int)
    prevalence_ages = np.array([59, 60, 62, 64, 79, 81], dtype=float)
    mapping = compute_prevalence(ages, prevalence_labels, prevalence_ages, gap=GAP)
    official_mapping = _official_compute_prevalence(
        ages, prevalence_labels, prevalence_ages, gap=GAP
    )
    assert compute_reward(labels, binary, ages, mapping) == pytest.approx(
        _official_compute_reward(labels, binary, ages, official_mapping)
    )
