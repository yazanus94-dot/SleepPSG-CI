"""Leakage-resistant split utilities for site-held-out evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


class LeakageError(ValueError):
    """Raised when training and evaluation partitions overlap."""


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
    """Ensure no patient identifier appears in both partitions."""
    overlap = set(train_patients) & set(test_patients)
    if overlap:
        raise LeakageError(
            f"{len(overlap)} patient identifier(s) appear in both train and test partitions."
        )


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
