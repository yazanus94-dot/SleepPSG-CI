"""Self-contained benchmark evaluator with contract, leakage and score checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sleeppsg_ci.cohort import FORBIDDEN_PREDICTORS
from sleeppsg_ci.metrics import score_predictions

ID_COLUMNS = ("SiteID", "BDSPPatientID")
LABEL = "Cognitive_Impairment"
PROBABILITY = "Cognitive_Impairment_Probability"


def check_leakage(feature_columns) -> list[str]:
    """Return forbidden predictors found in a participant feature list."""
    forbidden = {value.upper() for value in FORBIDDEN_PREDICTORS}
    return [name for name in feature_columns if str(name).upper() in forbidden]


def validate_predictions(predictions: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    """Validate the output contract and exact row alignment. Returns error strings."""
    errors: list[str] = []
    for column in (*ID_COLUMNS, PROBABILITY):
        if column not in predictions.columns:
            errors.append(f"Missing prediction column: {column}")
    if errors:
        return errors
    if len(predictions) != len(labels):
        errors.append(f"Row count mismatch: {len(predictions)} predictions vs {len(labels)} labels")
        return errors

    left = predictions[list(ID_COLUMNS)].astype(str).reset_index(drop=True)
    right = labels[list(ID_COLUMNS)].astype(str).reset_index(drop=True)
    if not left.equals(right):
        errors.append("Row alignment mismatch on SiteID/BDSPPatientID")
    if predictions[list(ID_COLUMNS)].duplicated().any():
        errors.append("Duplicate SiteID/BDSPPatientID rows in predictions")

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


def evaluate(labels: pd.DataFrame, predictions: pd.DataFrame, ages: pd.Series | None = None) -> dict:
    """Validate predictions and, if valid, compute the official metric suite."""
    errors = validate_predictions(predictions, labels)
    if errors:
        return {"valid": False, "errors": errors, "metrics": None}
    y = pd.to_numeric(labels[LABEL], errors="coerce").to_numpy(dtype=int)
    if not set(np.unique(y)).issubset({0, 1}):
        return {"valid": False, "errors": ["Labels must be binary 0/1"], "metrics": None}
    age_values = ages if ages is not None else labels["Age"]
    probabilities = pd.to_numeric(predictions[PROBABILITY], errors="coerce").to_numpy(dtype=float)
    metrics = score_predictions(y, probabilities, age_values.to_numpy(dtype=float))
    return {"valid": True, "errors": [], "metrics": metrics}


def evaluate_files(labels_csv: str | Path, predictions_csv: str | Path) -> dict:
    """Load two CSVs and evaluate them."""
    return evaluate(pd.read_csv(labels_csv), pd.read_csv(predictions_csv))
