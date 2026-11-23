"""Cohort metadata auditing for the PhysioNet 2026 training demographics file.

The schema implemented here was verified against the official challenge
documentation and the official ``python-example-2026`` example code at commit
``c0bcd78ddb892290b9d218be9be19660f4d8bdf3`` (inspected 2026-11-23).

Only aggregate statistics are emitted. No patient-level row is ever written to a
public report.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .identity import PATIENT_KEY_COLUMN, clean_identifier

_RECORDING_RE = re.compile(r"^(sub-[A-Za-z0-9]+)[-_]ses[-_]?([A-Za-z0-9]+)$")
_SITE_RE = re.compile(r"^[SI]\d{4}$")

#: Columns that must be present in a demographics.csv for the audit to run.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "SiteID",
    "BDSPPatientID",
    "BidsFolder",
    "SessionID",
    "Age",
    "Cognitive_Impairment",
)

#: Optional baseline variables that may be used as model predictors if permitted.
OPTIONAL_DEMOGRAPHIC_COLUMNS: tuple[str, ...] = ("Sex", "Race", "Ethnicity", "BMI")

#: Post-PSG / identifier variables that must never be used as predictors.
FORBIDDEN_PREDICTORS: tuple[str, ...] = (
    "SiteID",
    "BDSPPatientID",
    "BidsFolder",
    "SessionID",
    "CreationTime",
    "Cognitive_Impairment",
    "Time_to_Event",
    "Last_Known_Visit_Date",
    "Time_to_Last_Visit",
    "ICD",
    PATIENT_KEY_COLUMN,  # grouping key, never a predictor
    "recording",  # file name, traceability only
)


class CohortError(ValueError):
    """Raised when the cohort metadata is structurally invalid."""


def normalize_label(value: object) -> int | None:
    """Normalize an official outcome value to ``1``, ``0`` or ``None``.

    Mirrors the official ``sanitize_boolean_value`` semantics: numeric 0/1 and
    the tokens false/f/no/n and true/t/yes/y. Blank, missing, ``other`` and any
    unrecognized value are returned as ``None`` (excluded from binary training
    and scoring). An indeterminate record is never converted to a negative.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    token = str(value).strip().casefold()
    if token in {"", "nan", "none", "other", "unknown", "unavailable", "na", "n/a"}:
        return None
    if token in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    if token in {"0", "0.0", "false", "f", "no", "n"}:
        return 0
    try:
        number = float(token)
    except ValueError:
        return None
    if number == 1.0:
        return 1
    if number == 0.0:
        return 0
    return None


def validate_required_columns(frame: pd.DataFrame) -> None:
    """Raise :class:`CohortError` listing any missing required columns."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise CohortError(f"Missing required metadata columns: {', '.join(missing)}")


def _normalize_site(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


@dataclass
class CohortAudit:
    """Aggregate-only cohort audit result (no patient-level rows)."""

    total_records: int
    positive: int
    negative: int
    excluded_indeterminate: int
    missing_labels: int
    unique_patients: int
    duplicate_patient_id_rows: int
    unique_sessions: int
    duplicate_session_rows: int
    missing_site_ids: int
    missing_patient_ids: int
    incomplete_recording_rows: int
    missing_demographics: dict[str, int]
    records_by_site: dict[str, dict[str, int]]
    age_summary: dict[str, dict[str, float | int | None]]
    eligibility: dict[str, float | int]
    required_columns: list[str] = field(default_factory=lambda: list(REQUIRED_COLUMNS))

    def to_dict(self) -> dict:
        """Return a deterministic, JSON-serializable aggregate report.

        Non-finite floats (possible for empty age subgroups) are converted to
        ``None`` so that ``to_json(allow_nan=False)`` always produces valid JSON.
        """
        return _json_safe(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        """Serialize the aggregate report with stable key ordering."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


def _json_safe(value):
    """Recursively replace non-finite floats with JSON ``null``."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

    def to_json(self, indent: int = 2) -> str:
        """Serialize the aggregate report with stable key ordering."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


def _age_summary(ages: pd.Series) -> dict[str, float | int | None]:
    finite = pd.to_numeric(ages, errors="coerce").dropna()
    if finite.empty:
        # JSON null, not NaN: an empty subgroup has no distribution.
        return {"count": 0, "min": None, "max": None,
                "mean": None, "median": None, "std": None}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "std": float(finite.std(ddof=0)),
    }


def audit_cohort(frame: pd.DataFrame) -> CohortAudit:
    """Compute an aggregate eligibility and integrity report for a cohort.

    Parameters
    ----------
    frame:
        A demographics table containing at least :data:`REQUIRED_COLUMNS`.
    """
    validate_required_columns(frame)

    total = int(len(frame))
    labels = frame["Cognitive_Impairment"].map(normalize_label)
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    raw_labels = frame["Cognitive_Impairment"]
    blank = raw_labels.isna() | raw_labels.astype(str).str.strip().eq("")
    missing_labels = int(blank.sum())
    excluded = max(int(labels.isna().sum()) - missing_labels, 0)

    site = frame["SiteID"].map(_normalize_site)
    patient = frame["BDSPPatientID"].map(_normalize_id)
    folder = frame["BidsFolder"].map(_normalize_id)
    session = frame["SessionID"].map(_normalize_id)

    # Official identity: SiteID_BDSPPatientID. Identical bare IDs at two sites
    # are distinct patients and must not be merged.
    composite = [
        f"{site_value}_{patient_value}" if site_value and patient_value else ""
        for site_value, patient_value in zip(site, patient, strict=True)
    ]
    duplicate_patient_rows = int(pd.Series(composite).duplicated().sum())
    sessions = list(zip(folder, session, strict=False))
    duplicate_session_rows = int(pd.Series(sessions, dtype=object).duplicated().sum())
    missing_site_ids = int((site == "").sum())
    missing_patient_ids = int((patient == "").sum())
    incomplete_recording_rows = int(
        sum(1 for s, f, sess in zip(site, folder, session, strict=True) if not (s and f and sess))
    )

    missing_demographics: dict[str, int] = {}
    for column in ("Age", "Sex", "BMI", "Race", "Ethnicity"):
        if column in frame.columns:
            if column in {"Age", "BMI"}:
                # One mask: coercion to NaN already covers blank strings, so
                # adding a separate blank-string count would double-count.
                missing = pd.to_numeric(frame[column], errors="coerce").isna()
            else:
                missing = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
            missing_demographics[column] = int(missing.sum())

    by_site: dict[str, dict[str, int]] = {}
    for site_id, group in frame.groupby(site, dropna=False, sort=True):
        group_labels = labels.loc[group.index]
        key = site_id if site_id else "<missing>"
        by_site[key] = {
            "total": int(len(group)),
            "positive": int((group_labels == 1).sum()),
            "negative": int((group_labels == 0).sum()),
            "excluded": int(group_labels.isna().sum()),
        }

    eligible = positive + negative
    return CohortAudit(
        total_records=total,
        positive=positive,
        negative=negative,
        excluded_indeterminate=excluded,
        missing_labels=missing_labels,
        unique_patients=int(pd.Series(composite).nunique()),
        duplicate_patient_id_rows=duplicate_patient_rows,
        unique_sessions=int(len(set(sessions))),
        duplicate_session_rows=duplicate_session_rows,
        missing_site_ids=missing_site_ids,
        missing_patient_ids=missing_patient_ids,
        incomplete_recording_rows=incomplete_recording_rows,
        missing_demographics=missing_demographics,
        records_by_site=by_site,
        age_summary={
            "overall": _age_summary(frame["Age"]),
            "positive": _age_summary(frame.loc[labels == 1, "Age"]),
            "negative": _age_summary(frame.loc[labels == 0, "Age"]),
        },
        eligibility={
            "eligible_records": eligible,
            "excluded_records": total - eligible,
            "eligible_fraction": round(eligible / total, 6) if total else 0.0,
        },
    )


def build_recording_map(frame: pd.DataFrame) -> dict[str, str]:
    """Map ``(BidsFolder, SessionID)`` to the expected EDF file name.

    Official convention: ``BidsFolder=sub-S0001111197789``, ``SessionID=2`` maps
    to ``physiological_data/S0001/sub-S0001111197789-ses2.edf``. The returned
    mapping is for internal traceability only and must not be used as a feature.
    """
    validate_required_columns(frame)
    mapping: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        folder = clean_identifier(getattr(row, "BidsFolder", ""))
        session = clean_identifier(getattr(row, "SessionID", ""))
        site = clean_identifier(getattr(row, "SiteID", ""))
        # Skip incomplete rows instead of emitting literal "nan"/empty paths.
        if not (folder and session and site):
            continue
        mapping[f"{folder}|{session}"] = f"physiological_data/{site}/{folder}-ses{session}.edf"
    return mapping


def parse_recording_identity(path: str | Path) -> dict[str, str]:
    """Parse BIDS identity from an EDF file name or path.

    Official physiological names follow ``sub-<BidsFolder>-ses<SessionID>.edf``.
    The source site is taken from a path component when available, otherwise
    from the standard ``sub-S0001...`` prefix. Missing components are returned
    as empty strings; callers must validate them before joining.
    """
    file_path = Path(path)
    match = _RECORDING_RE.match(file_path.stem)
    folder = match.group(1) if match else ""
    session = match.group(2) if match else ""

    site = ""
    for part in file_path.parts:
        if _SITE_RE.match(str(part)):
            site = str(part)
            break
    if not site and folder.startswith("sub-"):
        candidate = folder[4:9]
        if _SITE_RE.match(candidate):
            site = candidate
    return {"BidsFolder": folder, "SessionID": session, "SiteID": site}


def load_cohort_csv(path: str | Path) -> pd.DataFrame:
    """Load a demographics CSV, raising a clear error when it is unavailable."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Demographics file not found: {csv_path}. Supply a labelled training "
            "demographics.csv (e.g. training_set_small/demographics.csv). "
            "No dataset was downloaded."
        )
    return pd.read_csv(csv_path)


def write_audit_report(audit: CohortAudit, path: str | Path) -> Path:
    """Write a deterministic aggregate JSON report to ``path``."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(audit.to_json() + "\n", encoding="utf-8")
    return output
