"""End-to-end synthetic pipeline: EDF -> features -> metadata join -> evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import edfio
import numpy as np
import pandas as pd
import pytest

from sleeppsg_ci.channel_audit import ChannelRule
from sleeppsg_ci.cohort import parse_recording_identity
from sleeppsg_ci.cohort_join import JoinError, join_feature_cohort
from sleeppsg_ci.evaluation import evaluate_site_held_out
from sleeppsg_ci.feature_table import extract_from_edf
from sleeppsg_ci.modeling import select_feature_matrix

SITES = ("S0001", "I0002", "I0006")
RULES = (
    ChannelRule("EEG", "EEG", verified=True),
    ChannelRule("ECG", "ECG", verified=True),
)


def _make_dataset(tmp_path: Path):
    """Create 3 sites x 2 patients of synthetic EDFs and demographics."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    physio = data_dir / "physiological_data"
    records = []
    t = np.arange(0, 60.0, 1 / 128.0)
    for site_index, site in enumerate(SITES):
        (physio / site).mkdir(parents=True, exist_ok=True)
        for patient_index in range(2):
            folder = f"sub-{site}1111{patient_index:04d}"
            session = "1"
            edf_path = physio / site / f"{folder}-ses{session}.edf"
            eeg = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 2 * t)
            ecg = np.sin(2 * np.pi * 1.2 * t)
            edfio.Edf([
                edfio.EdfSignal(eeg, 128.0, label="EEG", physical_dimension="uV",
                                physical_range=(-100, 100)),
                edfio.EdfSignal(ecg, 128.0, label="ECG", physical_dimension="mV",
                                physical_range=(-5, 5)),
            ]).write(edf_path)
            records.append({
                "SiteID": site,
                "BDSPPatientID": f"p{site_index}{patient_index}",
                "BidsFolder": folder,
                "SessionID": session,
                "Age": 55 + site_index * 5 + patient_index,
                "Sex": "Male" if patient_index == 0 else "Female",
                "BMI": 24.0 + patient_index,
                "Cognitive_Impairment": patient_index,
            })
    demographics = pd.DataFrame(records)
    demo_path = data_dir / "demographics.csv"
    demographics.to_csv(demo_path, index=False)

    rows = []
    for edf_path in sorted(physio.glob("*/*.edf")):
        features = extract_from_edf(edf_path, rules=RULES, chunk_seconds=20.0, epoch_seconds=30.0)
        features["recording"] = edf_path.stem
        features.update(parse_recording_identity(edf_path))
        rows.append(features)
    features_path = data_dir / "features.csv"
    pd.DataFrame(rows).to_csv(features_path, index=False)
    return demo_path, features_path, demographics, pd.DataFrame(rows)


def test_end_to_end_join_and_leakage_free_features(tmp_path):
    _, _, demographics, features = _make_dataset(tmp_path)
    cohort, report = join_feature_cohort(demographics, features)
    assert report.metadata_rows == 6
    assert report.feature_rows == 6
    assert report.matched == 6
    assert report.unmatched == 0
    assert report.output_rows == 6
    assert report.predictor_columns > 0

    for column in ("SiteID", "BDSPPatientID", "patient_key", "BidsFolder",
                   "SessionID", "Cognitive_Impairment", "Age", "Sex", "BMI"):
        assert column in cohort.columns
    assert cohort["patient_key"].nunique() == 6
    assert set(cohort["Cognitive_Impairment"].unique()) == {0, 1}

    for kind in ("demographic", "signal", "combined"):
        matrix = select_feature_matrix(cohort, kind)
        for forbidden in ("SiteID", "BDSPPatientID", "patient_key", "BidsFolder",
                          "SessionID", "Cognitive_Impairment", "recording"):
            assert forbidden not in matrix.columns
        assert not any(column.startswith("ch_") for column in matrix.columns)


def test_site_held_out_evaluation_runs_on_joined_cohort(tmp_path):
    _, _, demographics, features = _make_dataset(tmp_path)
    cohort, _ = join_feature_cohort(demographics, features)
    results = evaluate_site_held_out(cohort, bootstrap_n_boot=5)
    assert len(results["per_fold"]) == 3
    for fold in results["per_fold"]:
        assert fold["trainable"] is True
        assert "dummy" in fold["models"]
        assert "auroc_age_ci" in fold["models"]["dummy"]
        assert fold["models"]["dummy"]["auroc_age_ci"]["resampling_unit"] == "composite_patient"


def test_join_rejects_unmatched_and_duplicate_inputs(tmp_path):
    _, _, demographics, features = _make_dataset(tmp_path)
    _, report = join_feature_cohort(demographics, features.head(4))
    assert report.unmatched == 0
    assert report.unmatched_metadata == 2
    assert report.matched == 4

    with pytest.raises(JoinError, match="duplicate"):
        join_feature_cohort(demographics, pd.concat([features, features.head(1)], ignore_index=True))

    broken = demographics.copy()
    broken.loc[1, "BidsFolder"] = broken.loc[0, "BidsFolder"]
    broken.loc[1, "SessionID"] = broken.loc[0, "SessionID"]
    with pytest.raises(JoinError, match="duplicate BidsFolder/SessionID"):
        join_feature_cohort(broken, features)


def test_join_rejects_forbidden_predictor(tmp_path):
    _, _, demographics, features = _make_dataset(tmp_path)
    poisoned = features.copy()
    poisoned["Time_to_Event"] = 42
    with pytest.raises(JoinError, match="forbidden"):
        join_feature_cohort(demographics, poisoned)


def test_cli_build_feature_cohort_end_to_end(tmp_path):
    demo_path, features_path, _, _ = _make_dataset(tmp_path)
    out = tmp_path / "joined.csv"
    report = tmp_path / "report.json"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "build_feature_cohort.py"),
         str(demo_path), str(features_path), "--output", str(out), "--report", str(report)],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    joined = pd.read_csv(out)
    assert len(joined) == 6
    assert "patient_key" in joined.columns
    assert "Cognitive_Impairment" in joined.columns
    assert '"matched": 6' in report.read_text(encoding="utf-8")
