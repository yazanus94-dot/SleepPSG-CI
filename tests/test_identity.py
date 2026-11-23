"""Composite patient identity regressions (official SiteID_BDSPPatientID key)."""

import pandas as pd
import pytest

from sleeppsg_ci.identity import (
    add_patient_key,
    assert_patient_disjoint,
    clean_identifier,
    composite_patient_key,
    patient_keys,
)
from sleeppsg_ci.site_splits import LeakageError, assert_patient_frames_disjoint


def test_composite_key_uses_both_components():
    assert composite_patient_key("S0001", "p1") == "S0001_p1"
    assert composite_patient_key("I0002", "p1") == "I0002_p1"
    assert composite_patient_key("S0001", "p1") != composite_patient_key("I0002", "p1")


@pytest.mark.parametrize("site,patient", [("", "p1"), ("S0001", ""), (None, "p1"), ("S0001", None), ("  ", "p1")])
def test_missing_components_rejected(site, patient):
    with pytest.raises(ValueError, match="Missing or blank"):
        composite_patient_key(site, patient)


def test_same_bare_patient_id_at_two_sites_is_two_patients():
    frame = pd.DataFrame({
        "SiteID": ["S0001", "I0002"],
        "BDSPPatientID": ["p1", "p1"],
    })
    keys = patient_keys(frame)
    assert list(keys) == ["S0001_p1", "I0002_p1"]
    assert keys.nunique() == 2


def test_duplicate_patient_within_same_site():
    frame = pd.DataFrame({
        "SiteID": ["S0001", "S0001"],
        "BDSPPatientID": ["p1", "p1"],
    })
    keys = add_patient_key(frame)["patient_key"]
    assert keys.duplicated().sum() == 1


def test_multiple_sessions_one_patient_share_key():
    frame = pd.DataFrame({
        "SiteID": ["S0001", "S0001"],
        "BDSPPatientID": ["p1", "p1"],
        "SessionID": ["1", "2"],
    })
    keys = patient_keys(frame)
    assert keys.nunique() == 1
    assert list(keys) == ["S0001_p1", "S0001_p1"]


def test_frame_level_disjointness_uses_composite_key():
    train = pd.DataFrame({"SiteID": ["S0001"], "BDSPPatientID": ["p1"]})
    test_same_id_other_site = pd.DataFrame({"SiteID": ["I0002"], "BDSPPatientID": ["p1"]})
    test_same_patient = pd.DataFrame({"SiteID": ["S0001"], "BDSPPatientID": ["p1"]})
    assert_patient_frames_disjoint(train, test_same_id_other_site)
    with pytest.raises(LeakageError, match="composite"):
        assert_patient_frames_disjoint(train, test_same_patient)


def test_missing_site_or_patient_in_frame_rejected():
    for frame in (
        pd.DataFrame({"SiteID": [""], "BDSPPatientID": ["p1"]}),
        pd.DataFrame({"SiteID": ["S0001"], "BDSPPatientID": [None]}),
    ):
        with pytest.raises(ValueError, match="Missing or blank"):
            patient_keys(frame)


def test_clean_identifier_handles_nan():
    assert clean_identifier(float("nan")) == ""
    assert clean_identifier("  S0001  ") == "S0001"


def test_iterable_disjointness_still_rejects_overlap():
    assert_patient_disjoint(["S0001_p1"], ["I0002_p1"])
    with pytest.raises(LeakageError):
        assert_patient_disjoint(["S0001_p1"], ["S0001_p1"])
