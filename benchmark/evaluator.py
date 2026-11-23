"""Self-contained benchmark evaluator with contract and leakage checks.

Scope of automated leakage validation: this evaluator validates the *data
contract* — predictor manifests, identifier handling, label availability and
prediction shape. It cannot inspect an arbitrary participant's Python model code
and prove that no leakage occurred; that limitation is stated explicitly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sleeppsg_ci.cohort import FORBIDDEN_PREDICTORS, normalize_label
from sleeppsg_ci.metrics import score_predictions

ID_COLUMNS = ("SiteID", "BDSPPatientID")
LABEL = "Cognitive_Impairment"
PROBABILITY = "Cognitive_Impairment_Probability"
ALLOWED_PREDICTOR_PREFIXES = (
    "Age", "BMI", "Sex_male", "eeg_", "ecg_", "resp_", "airflow_",
    "emg_", "eog_", "spo2_", "missing_", "n_channels_", "signal_", "n_verified",
)
LEAKAGE_TOKENS = ("ICD", "FOLLOWUP", "LAST_VISIT", "TIME_TO_EVENT", "FILENAME", "RECORDING")

AUTOMATED_LEAKAGE_LIMITATION = (
    "Automated checks validate predictor manifests and the input/output contract only. "
    "They cannot inspect arbitrary participant model code or prove the absence of leakage."
)


def check_leakage(feature_columns) -> list[str]:
    """Return forbidden variables found in a participant feature manifest."""
    forbidden = {value.upper() for value in FORBIDDEN_PREDICTORS}
    found: list[str] = []
    for name in feature_columns:
        upper = str(name).upper()
        if upper in forbidden or any(token in upper for token in LEAKAGE_TOKENS):
            found.append(str(name))
    return found


def validate_feature_manifest(feature_columns) -> list[str]:
    """Enforce the explicit predictor whitelist; grouping IDs are allowed separately."""
    errors: list[str] = []
    columns = [str(name) for name in feature_columns]
    leaks = check_leakage(columns)
    if leaks:
        errors.append("Forbidden variables in feature manifest: " + ", ".join(sorted(leaks)))
    if LABEL in columns:
        errors.append("Target label must not be present in evaluation predictor columns.")
    grouping = set(ID_COLUMNS)
    unknown = [
        column for column in columns
        if column not in grouping and not column.startswith(ALLOWED_PREDICTOR_PREFIXES)
    ]
    if unknown:
        errors.append("Predictors outside the allowed whitelist: " + ", ".join(sorted(unknown)))
    return errors


def validate_predictions(predictions: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    """Validate the output contract and exact row alignment. Returns error strings."""
    errors: list[str] = []
    for column in (*ID_COLUMNS, PROBABILITY):
        if column not in predictions.columns:
            errors.append(f"Missing prediction column: {column}")
    if errors:
        return errors
    if LABEL in predictions.columns:
        errors.append("Target label must not be present in prediction rows.")
    if len(predictions) != len(labels):
        errors.append(f"Row count mismatch: {len(predictions)} predictions vs {len(labels)} labels")
        return errors

    left = predictions[list(ID_COLUMNS)].astype(str).reset_index(drop=True)
    right = labels[list(ID_COLUMNS)].astype(str).reset_index(drop=True)
    if not left.equals(right):
        errors.append("Row alignment mismatch on SiteID/BDSPPatientID")
    if predictions[list(ID_COLUMNS)].duplicated().any():
        errors.append("Duplicate SiteID/BDSPPatientID rows in predictions")
    if labels[list(ID_COLUMNS)].duplicated().any():
        errors.append("Duplicate SiteID/BDSPPatientID rows in evaluation labels")

    probabilities = pd.to_numeric(predictions[PROBABILITY], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(probabilities)):
        errors.append("Predictions contain NaN or infinite values")
    if np.any((probabilities < 0) | (probabilities > 1)):
        errors.append("Probabilities outside [0, 1]")
    return errors


def reproducibility_ok(first: pd.DataFrame, second: pd.DataFrame) -> bool:
    """Return True when two runs produce byte-identical probability vectors."""
    a = pd.to_numeric(first[PROBABILITY], errors="coerce").to_numpy(dtype=float).tobytes()
    b = pd.to_numeric(second[PROBABILITY], errors="coerce").to_numpy(dtype=float).tobytes()
    return hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def evaluate(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    ages: pd.Series | None = None,
    feature_columns=None,
    evaluation_features: pd.DataFrame | None = None,
) -> dict:
    """Validate inputs and, if valid, compute the official metric suite.

    Indeterminate labels are excluded from scoring; predictions and labels are
    filtered together so row alignment is preserved.
    """
    errors: list[str] = []
    if feature_columns is not None:
        errors.extend(validate_feature_manifest(feature_columns))
    if evaluation_features is not None:
        errors.extend(validate_feature_manifest(evaluation_features.columns))
    if errors:
        return {"valid": False, "errors": errors, "metrics": None,
                "automated_leakage_limitation": AUTOMATED_LEAKAGE_LIMITATION}

    if LABEL not in labels.columns:
        return {"valid": False, "errors": ["Missing label column"], "metrics": None}
    if len(predictions) != len(labels):
        errors.append(f"Row count mismatch: {len(predictions)} predictions vs {len(labels)} labels")
        return {"valid": False, "errors": errors, "metrics": None}

    normalized = labels[LABEL].map(normalize_label)
    eligible = normalized.isin([0, 1])
    n_excluded = int((~eligible).sum())
    eligible_labels = labels.loc[eligible].copy()
    eligible_labels[LABEL] = normalized.loc[eligible].astype(int)
    eligible_predictions = predictions.loc[eligible].reset_index(drop=True)
    eligible_labels = eligible_labels.reset_index(drop=True)

    errors = validate_predictions(eligible_predictions, eligible_labels)
    if errors:
        return {"valid": False, "errors": errors, "metrics": None,
                "n_excluded_labels": n_excluded}
    if eligible_labels.empty:
        return {"valid": False, "errors": ["No eligible binary labels"], "metrics": None,
                "n_excluded_labels": n_excluded}

    y = eligible_labels[LABEL].to_numpy(dtype=int)
    age_values = (
        ages.loc[eligible].reset_index(drop=True)
        if ages is not None else eligible_labels["Age"]
    )
    probabilities = pd.to_numeric(
        eligible_predictions[PROBABILITY], errors="coerce"
    ).to_numpy(dtype=float)
    metrics = score_predictions(y, probabilities, age_values.to_numpy(dtype=float))
    return {
        "valid": True,
        "errors": [],
        "metrics": metrics,
        "n_excluded_labels": n_excluded,
        "automated_leakage_limitation": AUTOMATED_LEAKAGE_LIMITATION,
    }


def evaluate_files(labels_csv: str | Path, predictions_csv: str | Path) -> dict:
    """Load two CSVs and evaluate them."""
    return evaluate(pd.read_csv(labels_csv), pd.read_csv(predictions_csv))
