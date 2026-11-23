import json

import pandas as pd
import pytest

from sleeppsg_ci.cohort import (
    REQUIRED_COLUMNS,
    CohortError,
    audit_cohort,
    build_recording_map,
    load_cohort_csv,
    normalize_label,
    write_audit_report,
)


def make_frame():
    return pd.DataFrame({
        "SiteID": ["S0001", "S0001", "I0002", "I0002", "I0006"],
        "BDSPPatientID": ["p1", "p2", "p3", "p4", "p5"],
        "BidsFolder": ["sub-1", "sub-2", "sub-3", "sub-4", "sub-5"],
        "SessionID": ["1", "1", "1", "2", "1"],
        "Age": [60, 72, 55, 80, 67],
        "Sex": ["Male", "Female", "Female", "Male", "Unknown"],
        "BMI": [25.0, None, 30.0, 22.0, 28.0],
        "Cognitive_Impairment": [1, 0, "other", "", 1],
    })


def test_missing_columns_raises():
    frame = make_frame().drop(columns=["Age"])
    with pytest.raises(CohortError, match="Age"):
        audit_cohort(frame)


def test_label_normalization():
    assert normalize_label(1) == 1
    assert normalize_label("TRUE") == 1
    assert normalize_label("yes") == 1
    assert normalize_label(0) == 0
    assert normalize_label("false") == 0
    assert normalize_label("n") == 0
    assert normalize_label("other") is None
    assert normalize_label("") is None
    assert normalize_label(float("nan")) is None
    assert normalize_label(None) is None


def test_indeterminate_and_missing_excluded():
    audit = audit_cohort(make_frame())
    assert audit.positive == 2
    assert audit.negative == 1
    assert audit.excluded_indeterminate == 1  # "other"
    assert audit.missing_labels == 1  # blank
    assert audit.eligibility["eligible_records"] == 3


def test_duplicates_and_missing_site():
    frame = make_frame()
    frame.loc[5] = ["S0001", "p1", "sub-1", "1", 60, "Male", 25.0, 1]
    frame.loc[6] = ["", "p6", "sub-6", "1", 60, "Male", 25.0, 0]
    audit = audit_cohort(frame)
    assert audit.duplicate_patient_id_rows >= 1
    assert audit.duplicate_session_rows >= 1
    assert audit.missing_site_ids == 1


def test_source_level_counts_aggregate_only():
    audit = audit_cohort(make_frame())
    assert set(audit.records_by_site) == {"S0001", "I0002", "I0006"}
    assert audit.records_by_site["S0001"]["positive"] == 1
    assert audit.records_by_site["I0002"]["excluded"] == 2


def test_age_summary_and_deterministic_json(tmp_path):
    audit = audit_cohort(make_frame())
    assert audit.age_summary["overall"]["count"] == 5
    target = write_audit_report(audit, tmp_path / "audit.json")
    first = target.read_text(encoding="utf-8")
    write_audit_report(audit, target)
    assert target.read_text(encoding="utf-8") == first
    # No patient-level identifiers in the public report.
    assert "p1" not in first and "sub-1" not in first
    json.loads(first)


def test_recording_map_uses_official_convention():
    mapping = build_recording_map(make_frame())
    assert mapping["sub-1|1"] == "physiological_data/S0001/sub-1-ses1.edf"


def test_missing_dataset_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No dataset was downloaded"):
        load_cohort_csv(tmp_path / "missing.csv")


def test_required_columns_constant():
    assert "Cognitive_Impairment" in REQUIRED_COLUMNS
    assert "SiteID" in REQUIRED_COLUMNS
