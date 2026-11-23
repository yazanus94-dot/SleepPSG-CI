"""Leakage-resistant split utilities for site-held-out evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from .identity import LeakageError, assert_patient_disjoint as _assert_keys_disjoint, patient_keys

__all__ = [
    "LeakageError",
    "assert_patient_disjoint",
    "assert_patient_frames_disjoint",
    "fold_summary",
    "is_scoreable",
    "leave_one_site_out",
]


def leave_one_site_out(sites: Sequence[str]) -> list[tuple[tuple[str, ...], str]]:
    """Return deterministic ``(training_sites, held_out_site)`` folds.

    At least two distinct sources are required; otherwise cross-source
    generalization is not identifiable and the function refuses to fabricate a
    split.
    """
    unique = sorted({str(site) for site in sites if str(site).strip()})
    if len(unique) < 2:
        raise ValueError(
            f"Leave-one-site-out requires at least 2 distinct sources; found {len(unique)}."
        )
    return [
        (tuple(site for site in unique if site != held_out), held_out)
        for held_out in unique
    ]


def assert_patient_disjoint(
    train_patients: Iterable[str], test_patients: Iterable[str]
) -> None:
    """Ensure no patient identifier appears in both partitions.

    For raw identifier iterables this compares the values as supplied. When
    validating data frames use :func:`assert_patient_frames_disjoint`, which
    builds the official composite ``SiteID_BDSPPatientID`` key first.
    """
    _assert_keys_disjoint(train_patients, test_patients)


def assert_patient_frames_disjoint(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    """Ensure no composite patient identity spans both partitions."""
    _assert_keys_disjoint(patient_keys(train_frame), patient_keys(test_frame))


def is_scoreable(labels: Sequence[int]) -> bool:
    """Return True only when both classes are present (AUROC is defined)."""
    return len({int(label) for label in labels}) >= 2


def fold_summary(
    held_out_site: str,
    labels: Sequence[int],
    train_sites: Sequence[str],
) -> dict:
    """Describe a held-out fold without exposing patient-level information."""
    labels = [int(label) for label in labels]
    return {
        "held_out_site": held_out_site,
        "training_sites": list(train_sites),
        "n_records": len(labels),
        "n_positive": sum(1 for label in labels if label == 1),
        "n_negative": sum(1 for label in labels if label == 0),
        "scoreable": is_scoreable(labels),
        "reason_if_not_scoreable": (
            "AUROC undefined: held-out source contains a single class."
            if not is_scoreable(labels)
            else None
        ),
    }
