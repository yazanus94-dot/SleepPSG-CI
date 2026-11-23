"""Composite patient identity shared by auditing, grouping and evaluation.

The official 2026 scorer aligns patients by ``SiteID_BDSPPatientID``. A bare
``BDSPPatientID`` is therefore **not** a patient identity across sources: the
same bare identifier at two sites denotes two different records. This module
provides the single helper used everywhere a patient key is required.

The key is used only for grouping, traceability and bootstrap resampling. It is
never a model predictor.
"""

from __future__ import annotations

import pandas as pd

PATIENT_KEY_COLUMN = "patient_key"
SITE_COLUMN = "SiteID"
PATIENT_COLUMN = "BDSPPatientID"


class LeakageError(ValueError):
    """Raised when training and evaluation partitions share a patient."""


def clean_identifier(value: object) -> str:
    """Normalize an identifier to a stripped string; blanks become ``''``."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def composite_patient_key(site: object, patient: object) -> str:
    """Return the official composite key ``SiteID_BDSPPatientID``.

    Raises ``ValueError`` when either component is missing or blank, so a
    malformed identity can never silently become a grouping or bootstrap key.
    """
    site_id = clean_identifier(site)
    patient_id = clean_identifier(patient)
    if not site_id or not patient_id:
        raise ValueError(
            "Missing or blank site/patient identifier; a composite patient key "
            "requires both SiteID and BDSPPatientID."
        )
    return f"{site_id}_{patient_id}"


def add_patient_key(
    frame: pd.DataFrame,
    *,
    column: str = PATIENT_KEY_COLUMN,
    site_column: str = SITE_COLUMN,
    patient_column: str = PATIENT_COLUMN,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a validated composite patient key column."""
    for name in (site_column, patient_column):
        if name not in frame.columns:
            raise ValueError(f"Cannot build patient key: missing column '{name}'.")
    site_ids = frame[site_column].map(clean_identifier)
    patient_ids = frame[patient_column].map(clean_identifier)
    invalid = int(((site_ids == "") | (patient_ids == "")).sum())
    if invalid:
        raise ValueError(
            f"Missing or blank site/patient identifier in {invalid} row(s); "
            "refusing to build patient keys."
        )
    result = frame.copy()
    result[column] = [
        composite_patient_key(site, patient)
        for site, patient in zip(site_ids, patient_ids, strict=True)
    ]
    return result


def patient_keys(
    frame: pd.DataFrame,
    *,
    column: str = PATIENT_KEY_COLUMN,
    site_column: str = SITE_COLUMN,
    patient_column: str = PATIENT_COLUMN,
) -> pd.Series:
    """Return validated composite patient keys for a frame."""
    return add_patient_key(
        frame, column=column, site_column=site_column, patient_column=patient_column
    )[column]


def assert_patient_disjoint(train_keys, test_keys) -> None:
    """Reject any overlap between training and held-out patient keys."""
    overlap = set(train_keys) & set(test_keys)
    if overlap:
        raise LeakageError(
            f"{len(overlap)} composite patient key(s) appear in both train and test partitions."
        )
