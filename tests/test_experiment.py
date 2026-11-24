"""Real-data experiment readiness: source validation, provenance and contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_models import build_parser, main, validate_sources


def cohort(tmp_path: Path):
    """Synthetic labelled cohort using only permitted training sources."""
    n = 18
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "SiteID": ["S0001"] * 6 + ["I0002"] * 6 + ["I0006"] * 6,
        "BDSPPatientID": [f"p{i}" for i in range(n)],
        "Age": rng.integers(50, 80, n),
        "BMI": rng.normal(27, 3, n),
        "Sex": rng.choice(["Male", "Female"], n),
        "Cognitive_Impairment": [i % 2 for i in range(n)],
        "eeg_delta_relative": rng.uniform(0, 1, n),
        "resp_rate_bpm": rng.normal(15, 2, n),
        "unit_unknown_channels_eeg": 0.0,
        "unit_unknown_channels_resp": 0.0,
    })
    path = tmp_path / "feature_cohort.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_validate_sources_accepts_training_sources_only():
    _, frame = cohort(Path("."))
    assert validate_sources(frame) == ["I0002", "I0006", "S0001"]


def test_validate_sources_rejects_hidden_and_unknown_sources():
    _, frame = cohort(Path("."))
    hidden = frame.copy()
    hidden.loc[0, "SiteID"] = "I0004"
    with pytest.raises(ValueError, match="Hidden validation/test"):
        validate_sources(hidden)

    unknown = frame.copy()
    unknown.loc[0, "SiteID"] = "UNKNOWN_SITE"
    with pytest.raises(ValueError, match="outside the labelled training partition"):
        validate_sources(unknown)


def test_main_writes_provenance_and_experiment_contract(tmp_path):
    feature_path, _ = cohort(tmp_path)
    metadata = tmp_path / "demographics.csv"
    pd.DataFrame({"SiteID": ["S0001"], "BDSPPatientID": ["p0"]}).to_csv(metadata, index=False)
    output = tmp_path / "results.json"
    comparison = tmp_path / "comparison.csv"
    manifest_path = tmp_path / "manifest.json"

    code = main([
        str(feature_path),
        "--metadata-csv", str(metadata),
        "--output", str(output),
        "--comparison", str(comparison),
        "--manifest-output", str(manifest_path),
        "--seed", "17",
    ])
    assert code == 0

    results = json.loads(output.read_text(encoding="utf-8"))
    manifest = results["manifest"]
    assert manifest["random_seed"] == 17
    assert set(manifest["input_checksums"]) == {"feature_cohort", "demographics"}
    assert all(len(value) == 64 for value in manifest["input_checksums"].values())

    experiment = results["experiment"]
    assert experiment["hidden_sources_present"] is False
    assert experiment["observed_sources"] == ["I0002", "I0006", "S0001"]
    assert experiment["age_gap_years"] == 2
    assert any("delta of 2 years" in item for item in experiment["contract"])
    assert any("Patient-cluster bootstrap" in item for item in experiment["contract"])
    assert len(results["per_fold"]) == 3
    assert results["per_fold"]
    for fold in results["per_fold"]:
        dummy = fold["models"]["dummy"]
        if dummy["scoreable"]:
            assert dummy["auroc_age_ci"]["resampling_unit"] == "composite_patient"
        else:
            assert "auroc_age_ci" not in dummy
            assert np.isnan(dummy["auroc_age"])

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["experiment"] == experiment
    assert comparison.exists()


def test_main_rejects_hidden_source_without_writing_results(tmp_path):
    feature_path, frame = cohort(tmp_path)
    frame.loc[0, "SiteID"] = "I0007"
    frame.to_csv(feature_path, index=False)
    output = tmp_path / "results.json"
    code = main([str(feature_path), "--output", str(output)])
    assert code == 2
    assert not output.exists()


def test_cli_experiment_end_to_end(tmp_path):
    feature_path, _ = cohort(tmp_path)
    output = tmp_path / "cli_results.json"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "evaluate_models.py"), str(feature_path),
         "--output", str(output), "--manifest-output", str(tmp_path / "cli_manifest.json")],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["experiment"]["hidden_sources_present"] is False


def test_parser_defaults():
    args = build_parser().parse_args(["cohort.csv"])
    assert args.seed == 42
    assert args.output.endswith("evaluation_results.json")
    assert args.manifest_output.endswith("experiment_manifest.json")
