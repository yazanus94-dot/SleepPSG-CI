import pytest

from sleeppsg_ci.site_splits import (
    LeakageError,
    assert_patient_disjoint,
    fold_summary,
    is_scoreable,
    leave_one_site_out,
)


def test_leave_one_site_out_is_deterministic_and_disjoint():
    folds = leave_one_site_out(["I0006", "S0001", "I0002", "S0001"])
    assert folds == [
        (("I0006", "S0001"), "I0002"),
        (("I0002", "S0001"), "I0006"),
        (("I0002", "I0006"), "S0001"),
    ]
    unique_held = [held for _, held in folds]
    assert unique_held == ["I0002", "I0006", "S0001"]
    for train, held in folds:
        assert held not in train
        assert len(train) == 2


def test_single_site_refused():
    with pytest.raises(ValueError, match="at least 2"):
        leave_one_site_out(["S0001"])


def test_patient_disjointness_enforced():
    assert_patient_disjoint(["a", "b"], ["c"])
    with pytest.raises(LeakageError, match="both train and test"):
        assert_patient_disjoint(["a", "b"], ["b", "c"])


def test_single_class_fold_not_scoreable():
    assert is_scoreable([0, 1, 1])
    assert not is_scoreable([0, 0, 0])
    summary = fold_summary("S0001", [0, 0, 0], ("I0002", "I0006"))
    assert summary["scoreable"] is False
    assert "undefined" in summary["reason_if_not_scoreable"]
    assert summary["n_negative"] == 3
