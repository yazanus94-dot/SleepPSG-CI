"""Cross-site evaluation must not desynchronise train/test feature schemas."""

import pandas as pd

from sleeppsg_ci.evaluation import evaluate_site_held_out


def test_unit_provenance_difference_does_not_break_folds():
    frame = pd.DataFrame({
        "SiteID": ["S0001"] * 6 + ["I0002"] * 6,
        "BDSPPatientID": [f"p{i}" for i in range(12)],
        "Age": list(range(60, 72)),
        "BMI": [25.0] * 12,
        "Sex": ["Male"] * 6 + ["Female"] * 6,
        "Cognitive_Impairment": [i % 2 for i in range(12)],
        "eeg_delta_relative": list(range(12)),
        # Only one site flags unknown EEG units; eligibility must be frozen
        # on the full cohort so both train and test use the same columns.
        "unit_unknown_channels_eeg": [0.0] * 6 + [1.0] * 6,
    })
    result = evaluate_site_held_out(
        frame, model_names=("signal_lr",), bootstrap_n_boot=0
    )
    assert result["per_fold"]
    for fold in result["per_fold"]:
        assert "error" not in fold["models"]["signal_lr"]
