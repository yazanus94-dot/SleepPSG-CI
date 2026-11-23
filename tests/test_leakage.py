import numpy as np
import pandas as pd
import pytest

from sleeppsg_ci.channel_audit import ChannelRule
from sleeppsg_ci.feature_table import extract_from_signals
from sleeppsg_ci.modeling import LeakageError, assert_no_leakage, select_feature_matrix
from benchmark.evaluator import check_leakage

FORBIDDEN = [
    "SiteID", "BDSPPatientID", "BidsFolder", "SessionID", "Time_to_Event",
    "Last_Known_Visit_Date", "Time_to_Last_Visit", "Cognitive_Impairment", "ICD_code",
]


def test_forbidden_predictors_rejected():
    for column in FORBIDDEN:
        with pytest.raises(LeakageError):
            assert_no_leakage(["Age", column])


def test_evaluator_leakage_check():
    assert check_leakage(["Age", "eeg_delta_power"]) == []
    assert "SiteID" in check_leakage(["Age", "SiteID"])


def test_feature_extraction_emits_no_identifiers():
    rules = (ChannelRule("EEG", "EEG", verified=True),)
    row = extract_from_signals({"EEG": (np.sin(np.arange(1000) / 10), 256.0)}, rules)
    assert not any(key in row for key in FORBIDDEN)
    assert all(key.startswith(("eeg_", "missing_", "n_verified")) for key in row)


def test_select_feature_matrix_excludes_identifiers():
    frame = pd.DataFrame({
        "SiteID": ["S0001"] * 3,
        "BDSPPatientID": ["p1", "p2", "p3"],
        "Age": [60, 70, 80],
        "BMI": [25.0, 26.0, 27.0],
        "Sex": ["Male", "Female", "Male"],
        "Cognitive_Impairment": [1, 0, 1],
        "eeg_delta_power": [1.0, 2.0, 3.0],
    })
    for kind in ("demographic", "signal", "combined"):
        columns = list(select_feature_matrix(frame, kind).columns)
        assert "SiteID" not in columns
        assert "BDSPPatientID" not in columns
        assert "Cognitive_Impairment" not in columns
