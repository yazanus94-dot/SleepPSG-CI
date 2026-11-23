"""Join per-recording PSG features to the labelled training demographics.

This module produces a *local, git-ignored* feature cohort for model training and
site-held-out evaluation. It keeps identifiers only as grouping/output metadata
and exposes them to the model layer through ``select_feature_matrix``, which
deliberately selects demographic and physiological predictor columns only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cohort import (
    FORBIDDEN_PREDICTORS,
    normalize_label,
    parse_recording_identity,
    validate_required_columns,
)
from .identity import PATIENT_KEY_COLUMN, add_patient_key, clean_identifier

GROUPING_COLUMNS = (
    "SiteID",
    "BDSPPatientID",
    "BidsFolder",
    "SessionID",
    PATIENT_KEY_COLUMN,
    "recording",
)
BASELINE_COLUMNS = ("Age", "Sex", "BMI")
LABEL_COLUMN = "Cognitive_Impairment"
RECORDING_KEY = ("BidsFolder", "SessionID")


class JoinError(ValueError):
    """Raised when the feature-to-metadata join is ambiguous or invalid."""


@dataclass
class JoinReport:
    """Aggregate, patient-free counts describing one feature-cohort join."""

    metadata_rows: int
    eligible_metadata_rows: int
    feature_rows: int
    matched: int
    unmatched: int
    unmatched_metadata: int
    duplicate_metadata_keys: int
    duplicate_feature_keys: int
    incomplete_keys: int
    excluded_indeterminate: int
    output_rows: int
    predictor_columns: int

    def to_dict(self) -> dict:
        return {
            "metadata_rows": self.metadata_rows,
            "eligible_metadata_rows": self.eligible_metadata_rows,
            "feature_rows": self.feature_rows,
            "matched": self.matched,
            "unmatched": self.unmatched,
            "unmatched_metadata": self.unmatched_metadata,
            "duplicate_metadata_keys": self.duplicate_metadata_keys,
            "duplicate_feature_keys": self.duplicate_feature_keys,
            "incomplete_keys": self.incomplete_keys,
            "excluded_indeterminate": self.excluded_indeterminate,
            "output_rows": self.output_rows,
            "predictor_columns": self.predictor_columns,
        }


def _recording_key(folder: object, session: object) -> str:
    folder_id = clean_identifier(folder)
    session_id = clean_identifier(session)
    return f"{folder_id}|{session_id}" if folder_id and session_id else ""


def _feature_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Return BidsFolder/SessionID/SiteID for each feature row."""
    if "BidsFolder" in frame.columns and "SessionID" in frame.columns:
        identity = frame[["BidsFolder", "SessionID"]].copy()
    elif "recording" in frame.columns:
        parsed = frame["recording"].map(parse_recording_identity)
        identity = pd.DataFrame(parsed.tolist(), index=frame.index)
    else:
        raise JoinError(
            "Feature table needs either BidsFolder/SessionID columns or a "
            "'recording' column with BIDS-style EDF names."
        )
    if "SiteID" in frame.columns:
        identity["SiteID"] = frame["SiteID"]
    return identity


def join_feature_cohort(
    demographics: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, JoinReport]:
    """Match each feature record to exactly one eligible metadata record.

    Only explicitly labelled ``0``/``1`` records enter the output. The returned
    frame includes grouping identifiers for validation and the normalized label
    for local training; both are excluded from the model feature-selection layer.
    """
    validate_required_columns(demographics)
    if features.empty:
        raise JoinError("Feature table is empty; nothing to join.")

    # Forbidden/post-PSG columns may never enter even the local cohort as
    # predictors. Grouping identifiers are the deliberate exception.
    predictor_forbidden = set(FORBIDDEN_PREDICTORS) - set(GROUPING_COLUMNS)
    offenders = sorted(predictor_forbidden.intersection(features.columns))
    if offenders:
        raise JoinError(
            "Feature table contains forbidden or post-PSG variables: "
            + ", ".join(offenders)
        )

    meta = demographics.copy()
    meta["_label"] = meta[LABEL_COLUMN].map(normalize_label)
    meta["_recording_key"] = [
        _recording_key(folder, session)
        for folder, session in zip(
            meta["BidsFolder"].map(clean_identifier),
            meta["SessionID"].map(clean_identifier),
            strict=True,
        )
    ]
    incomplete_meta = int((meta["_recording_key"] == "").sum())
    valid_meta = meta[meta["_recording_key"] != ""]
    dup_meta = int(valid_meta["_recording_key"].duplicated().sum())
    if dup_meta:
        raise JoinError(
            f"{dup_meta} duplicate BidsFolder/SessionID key(s) in metadata; "
            "cannot match recordings unambiguously."
        )
    meta_index = dict(zip(valid_meta["_recording_key"], valid_meta.index, strict=True))

    identity = _feature_identity(features)
    feature_keys = [
        _recording_key(folder, session)
        for folder, session in zip(
            identity["BidsFolder"].map(clean_identifier),
            identity["SessionID"].map(clean_identifier),
            strict=True,
        )
    ]
    incomplete = sum(1 for key in feature_keys if not key)
    dup_features = int(pd.Series([key for key in feature_keys if key]).duplicated().sum())
    if dup_features:
        raise JoinError(
            f"{dup_features} duplicate record key(s) in feature table; "
            "refusing an ambiguous join."
        )

    rows: list[dict] = []
    matched_keys: set[str] = set()
    matched = 0
    unmatched = 0
    excluded = 0
    for position, (feature_row, key) in enumerate(zip(
        features.to_dict(orient="records"), feature_keys, strict=True
    )):
        if not key or key not in meta_index:
            unmatched += 1
            continue
        meta_row = meta.loc[meta_index[key]]
        matched_keys.add(key)
        if "SiteID" in identity.columns:
            feature_site = clean_identifier(identity.iloc[position].get("SiteID", ""))
            meta_site = clean_identifier(meta_row.get("SiteID", ""))
            if feature_site and meta_site and feature_site != meta_site:
                raise JoinError("Site mismatch between feature record and metadata record.")
        label = meta_row["_label"]
        if label not in (0, 1):
            excluded += 1
            continue
        matched += 1

        row = dict(feature_row)
        for column in ("SiteID", "BDSPPatientID", "BidsFolder", "SessionID"):
            row[column] = meta_row[column]
        for column in BASELINE_COLUMNS:
            if column in meta_row.index:
                row[column] = meta_row[column]
        row[LABEL_COLUMN] = int(label)
        rows.append(row)

    if not rows:
        raise JoinError(
            "No feature record matched an eligible labelled metadata record; "
            "check recording names, BidsFolder/SessionID and label values."
        )

    joined = pd.DataFrame(rows).reset_index(drop=True)
    joined = add_patient_key(joined)

    predictor_columns = [
        column for column in joined.columns
        if column not in GROUPING_COLUMNS and column != LABEL_COLUMN
    ]
    # Defensive check: the final predictor list must still be leakage-free.
    from .modeling import assert_no_leakage  # local import avoids a cycle

    assert_no_leakage(predictor_columns)

    ordered = [column for column in GROUPING_COLUMNS if column in joined.columns]
    ordered += [column for column in (LABEL_COLUMN, *BASELINE_COLUMNS) if column in joined.columns]
    ordered += [column for column in predictor_columns if column not in ordered]
    joined = joined[ordered]

    report = JoinReport(
        metadata_rows=int(len(meta)),
        eligible_metadata_rows=int(meta["_label"].isin([0, 1]).sum()),
        feature_rows=int(len(features)),
        matched=matched,
        unmatched=unmatched,
        unmatched_metadata=int(len(set(valid_meta["_recording_key"]) - matched_keys)),
        duplicate_metadata_keys=dup_meta,
        duplicate_feature_keys=dup_features,
        incomplete_keys=incomplete + incomplete_meta,
        excluded_indeterminate=excluded,
        output_rows=int(len(joined)),
        predictor_columns=len(predictor_columns),
    )
    return joined, report
