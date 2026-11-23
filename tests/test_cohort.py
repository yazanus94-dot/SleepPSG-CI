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


def test_missing_demographics_not_double_counted():
    frame = make_frame()
    frame["Age"] = ["", None, "  ", 80, 67]
    frame["BMI"] = [25.0, "", None, 22.0, 28.0]
    audit = audit_cohort(frame)
    assert audit.missing_demographics["Age"] == 3
    assert audit.missing_demographics["BMI"] == 2


def test_unique_patients_use_composite_key():
    frame = make_frame()
    frame.loc[1, "BDSPPatientID"] = "p1"  # same bare ID, same site -> duplicate
    frame.loc[2, "BDSPPatientID"] = "p1"  # same bare ID, different site -> distinct
    audit = audit_cohort(frame)
    assert audit.unique_patients == 4
    assert audit.duplicate_patient_id_rows == 1
    assert audit.missing_patient_ids == 0


def test_empty_age_subgroups_serialize_as_null():
    frame = make_frame()
    frame["Cognitive_Impairment"] = ["other"] * len(frame)
    audit = audit_cohort(frame)
    assert audit.age_summary["positive"]["count"] == 0
    assert audit.age_summary["positive"]["mean"] is None
    payload = json.loads(audit.to_json())
    assert payload["age_summary"]["negative"]["min"] is None
    assert "NaN" not in audit.to_json()


def test_recording_map_skips_incomplete_rows(tmp_path):
    frame = make_frame()
    frame.loc[5] = ["", "p6", "sub-6", "1", 60, "Male", 25.0, 1]
    frame.loc[6] = ["S0001", "p7", float("nan"), "1", 60, "Male", 25.0, 1]
    frame.loc[7] = ["S0001", "p8", "sub-8", float("nan"), 60, "Male", 25.0, 1]
    mapping = build_recording_map(frame)
    assert "sub-6|1" not in mapping
    assert "nan" not in str(mapping).casefold()
    assert audit_cohort(frame).incomplete_recording_rows == 3


def test_parse_recording_identity():
    from sleeppsg_ci.cohort import parse_recording_identity

    parsed = parse_recording_identity(
        r"C:\data\physiological_data\S0001\sub-S0001111197789-ses2.edf"
    )
    assert parsed == {
        "BidsFolder": "sub-S0001111197789",
        "SessionID": "2",
        "SiteID": "S0001",
    }
    incomplete = parse_recording_identity("unexpected.edf")
    assert incomplete["BidsFolder"] == ""
    assert incomplete["SessionID"] == ""
