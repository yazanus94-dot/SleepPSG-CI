"""Official and secondary evaluation metrics.

The age-conditioned AUROC, age-weighted AUROC, prevalence reward and confusion
metrics below are adapted from the official ``evaluate_model.py`` of the George
B. Moody PhysioNet Challenge 2026 example code at commit
``c0bcd78ddb892290b9d218be9be19660f4d8bdf3`` (inspected 2026-11-23).

BSD 3-Clause License
Copyright (c) 2026 George B. Moody PhysioNet Challenges
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

AGE_GAP_YEARS = 2


def _validate_vectors(*vectors: Sequence) -> list[np.ndarray]:
    arrays = [np.asarray(vector, dtype=float).reshape(-1) for vector in vectors]
    if not arrays or any(array.size != arrays[0].size for array in arrays):
        raise ValueError("All metric vectors must have equal length.")
    return arrays


def compute_prevalence(ages, prevalence_labels, prevalence_ages, gap: int = AGE_GAP_YEARS) -> dict:
    """Empirical positive-class prevalence at each age (official implementation)."""
    # Evaluation ages and the training prevalence cohort may have different sizes;
    # only the two prevalence vectors must be aligned with each other.
    ages_arr = np.asarray(ages, dtype=float).reshape(-1)
    labels = np.asarray(prevalence_labels, dtype=float).reshape(-1)
    ref_ages = np.asarray(prevalence_ages, dtype=float).reshape(-1)
    if labels.size != ref_ages.size:
        raise ValueError("Prevalence labels and ages must have equal length.")
    unique_ages = np.unique(ages_arr[np.isfinite(ages_arr)])
    mapping: dict[float, list[float]] = defaultdict(list)
    for age in unique_ages:
        for i in range(labels.size):
            if np.isfinite(ref_ages[i]) and abs(age - ref_ages[i]) <= gap:
                mapping[float(age)].append(float(labels[i]))
    return {age: max(sum(values), 0.5) / len(values) for age, values in mapping.items()}


def compute_reward(labels, binary_predictions, ages, age_to_prevalence: dict) -> float:
    """Prevalence-based reward metric (official implementation)."""
    labels_arr, preds, ages_arr = _validate_vectors(labels, binary_predictions, ages)
    n = labels_arr.size
    if n == 0 or not age_to_prevalence:
        return float("nan")
    scores = np.zeros(n)
    counted = 0
    for i in range(n):
        if np.isfinite(ages_arr[i]) and float(ages_arr[i]) in age_to_prevalence:
            p = age_to_prevalence[float(ages_arr[i])]
            p = min(max(p, 0.5 / n), 1 - 0.5 / n)
            if labels_arr[i] == 1 and preds[i] == 1:
                scores[i] = 1 / p - 1
            elif labels_arr[i] == 0 and preds[i] == 1:
                scores[i] = -1
            elif labels_arr[i] == 1 and preds[i] == 0:
                scores[i] = -1
            elif labels_arr[i] == 0 and preds[i] == 0:
                scores[i] = 1 / (1 - p) - 1
            counted += 1
    return float(np.sum(scores) / counted) if counted else float("nan")


def compute_auroc_age(labels, predictions, ages, gap: int = AGE_GAP_YEARS) -> float:
    """Primary metric: age-conditioned AUROC (official implementation).

    Returns NaN when no positive/negative pair is comparable within ``gap``
    years; such a fold must be reported as not scoreable, never as zero.
    """
    labels_arr, preds, ages_arr = _validate_vectors(labels, predictions, ages)
    pos = np.flatnonzero(labels_arr == 1)
    neg = np.flatnonzero(labels_arr == 0)
    numer = 0.0
    denom = 0
    for i in pos:
        for j in neg:
            if abs(ages_arr[i] - ages_arr[j]) <= gap:
                if preds[i] > preds[j]:
                    numer += 1
                elif preds[i] == preds[j]:
                    numer += 0.5
                denom += 1
    return numer / denom if denom else float("nan")


def compute_auroc_weighted(labels, predictions, ages, gap: int = AGE_GAP_YEARS) -> float:
    """Age-weighted mean AUROC across ages (official implementation)."""
    labels_arr, preds, ages_arr = _validate_vectors(labels, predictions, ages)
    finite_ages = ages_arr[np.isfinite(ages_arr)]
    if finite_ages.size == 0:
        return float("nan")
    range_ages = np.arange(np.min(finite_ages) - gap, np.max(finite_ages) + gap + 1)
    pos = np.flatnonzero(labels_arr == 1)
    neg = np.flatnonzero(labels_arr == 0)
    numer = np.zeros(range_ages.size)
    denom = np.zeros(range_ages.size)
    for k, age in enumerate(range_ages):
        for i in pos:
            for j in neg:
                if abs(ages_arr[i] - age) <= gap and abs(ages_arr[j] - age) <= gap:
                    if preds[i] > preds[j]:
                        numer[k] += 1
                    elif preds[i] == preds[j]:
                        numer[k] += 0.5
                    denom[k] += 1
    weights = np.array([np.count_nonzero(np.abs(finite_ages - age) <= gap) for age in range_ages], dtype=float)
    zero = denom == 0
    denom[zero] = 1
    numer[zero] = 0
    weights[zero] = 0
    total = np.sum(weights)
    return float(np.sum(weights * (numer / denom)) / total) if total else float("nan")


def compute_auroc(labels, predictions) -> float:
    """Standard ROC-AUC (undefined for a single class -> NaN)."""
    labels_arr, preds = _validate_vectors(labels, predictions)
    if len(np.unique(labels_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(labels_arr, preds))


def compute_auprc(labels, predictions) -> float:
    """Average precision / PR-AUC."""
    labels_arr, preds = _validate_vectors(labels, predictions)
    if len(np.unique(labels_arr)) < 2:
        return float("nan")
    return float(average_precision_score(labels_arr, preds, pos_label=1))


def confusion_counts(labels, binary_predictions) -> dict[str, int]:
    """Return TP, FP, FN, TN counts."""
    labels_arr, preds = _validate_vectors(labels, binary_predictions)
    return {
        "tp": int(np.sum((labels_arr == 1) & (preds == 1))),
        "fp": int(np.sum((labels_arr == 0) & (preds == 1))),
        "fn": int(np.sum((labels_arr == 1) & (preds == 0))),
        "tn": int(np.sum((labels_arr == 0) & (preds == 0))),
    }


def sensitivity(labels, binary_predictions) -> float:
    counts = confusion_counts(labels, binary_predictions)
    denominator = counts["tp"] + counts["fn"]
    return counts["tp"] / denominator if denominator else float("nan")


def specificity(labels, binary_predictions) -> float:
    counts = confusion_counts(labels, binary_predictions)
    denominator = counts["tn"] + counts["fp"]
    return counts["tn"] / denominator if denominator else float("nan")


def balanced_accuracy(labels, binary_predictions) -> float:
    sens = sensitivity(labels, binary_predictions)
    spec = specificity(labels, binary_predictions)
    return (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else float("nan")


def calibration_slope_intercept(labels, probabilities) -> tuple[float, float]:
    """Descriptive calibration slope/intercept from logit(probability) -> label.

    This is fitted on the evaluation cohort for reporting only; it is never used
    to tune a model or to change predictions.
    """
    labels_arr, probs = _validate_vectors(labels, probabilities)
    valid = np.isfinite(probs) & (probs > 0) & (probs < 1)
    if valid.sum() < 3 or len(np.unique(labels_arr[valid])) < 2:
        return float("nan"), float("nan")
    logits = np.log(probs[valid] / (1 - probs[valid]))
    design = np.column_stack([np.ones(logits.size), logits])
    intercept, slope = np.linalg.lstsq(design, labels_arr[valid], rcond=None)[0]
    return float(slope), float(intercept)


def bootstrap_ci(
    metric_fn: Callable[[Sequence, Sequence, Sequence], float],
    labels: Sequence,
    probabilities: Sequence,
    ages: Sequence,
    patient_keys: Sequence[str],
    n_boot: int = 200,
    seed: int = 42,
    gap: int = AGE_GAP_YEARS,
) -> dict[str, float | int]:
    """True patient-cluster bootstrap CI for a metric.

    Resampling units are composite patient identities, not rows: when a patient
    is drawn, all of their eligible sessions/records are carried together. This
    prevents repeated visits from being treated as independent patients.
    """
    labels_arr, probs, ages_arr = _validate_vectors(labels, probabilities, ages)
    keys = np.asarray(patient_keys, dtype=object).reshape(-1)
    if keys.size != labels_arr.size:
        raise ValueError("patient_keys must match the label vector length.")
    if any(str(key).strip() == "" for key in keys):
        raise ValueError("patient_keys must be non-blank composite identifiers.")

    groups: dict[str, np.ndarray] = {}
    for key in sorted({str(key) for key in keys}):
        groups[key] = np.flatnonzero(np.array([str(value) == key for value in keys]))
    group_names = list(groups)
    group_indices = [groups[name] for name in group_names]

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(group_names), size=len(group_names))
        index = np.concatenate([group_indices[i] for i in chosen])
        try:
            value = metric_fn(labels_arr[index], probs[index], ages_arr[index], gap=gap)
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return {
            "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "n_valid": 0, "n_patients": len(group_names), "resampling_unit": "composite_patient",
        }
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "n_valid": len(values),
        "n_patients": len(group_names),
        "resampling_unit": "composite_patient",
    }


def score_predictions(
    labels,
    probabilities,
    ages,
    binary_predictions=None,
    prevalence_labels=None,
    prevalence_ages=None,
    gap: int = AGE_GAP_YEARS,
) -> dict:
    """Compute the primary and secondary metric suite for one cohort."""
    labels_arr, probs, ages_arr = _validate_vectors(labels, probabilities, ages)
    if np.any((labels_arr != 0) & (labels_arr != 1)):
        raise ValueError("Labels must be binary 0/1; indeterminate records must be excluded.")
    binary = np.asarray(binary_predictions, dtype=float).reshape(-1) if binary_predictions is not None else (probs >= 0.5).astype(float)
    if binary.size != labels_arr.size:
        raise ValueError("Binary predictions must match label length.")

    slope, intercept = calibration_slope_intercept(labels_arr, probs)
    primary = compute_auroc_age(labels_arr, probs, ages_arr, gap=gap)
    prevalence_map: dict | None = None
    if prevalence_labels is not None and prevalence_ages is not None:
        prevalence_map = compute_prevalence(ages_arr, prevalence_labels, prevalence_ages, gap=gap)

    return {
        "n": int(labels_arr.size),
        "n_positive": int(np.sum(labels_arr == 1)),
        "n_negative": int(np.sum(labels_arr == 0)),
        # A fold is scoreable only when the PRIMARY metric is defined: two
        # classes alone are insufficient if no comparable age pair exists.
        "scoreable": bool(len(np.unique(labels_arr)) >= 2 and np.isfinite(primary)),
        "auroc_age": primary,
        "auroc_age_weighted": compute_auroc_weighted(labels_arr, probs, ages_arr, gap=gap),
        "auroc": compute_auroc(labels_arr, probs),
        "auprc": compute_auprc(labels_arr, probs),
        "brier": float(brier_score_loss(labels_arr, probs)) if len(np.unique(labels_arr)) >= 2 else float("nan"),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "sensitivity": sensitivity(labels_arr, binary),
        "specificity": specificity(labels_arr, binary),
        "balanced_accuracy": balanced_accuracy(labels_arr, binary),
        "confusion_matrix": confusion_counts(labels_arr, binary),
        "reward": (compute_reward(labels_arr, binary, ages_arr, prevalence_map)
                   if prevalence_map is not None else float("nan")),
        "age_gap_years": gap,
    }
