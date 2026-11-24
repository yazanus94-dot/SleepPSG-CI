"""scikit-learn baseline models with fold-local imputation and scaling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cohort import FORBIDDEN_PREDICTORS
from .units import feature_modality, is_feature_eligible

RANDOM_SEED = 42
DEMOGRAPHIC_FEATURES = ("Age", "BMI", "Sex_male")
# All seven extracted modality families plus explicit missingness indicators.
# AIRFLOW and SPO2 produce features in feature_table.py and must be visible to
# signal-only and combined baselines as well.
SIGNAL_PREFIXES = (
    "eeg_", "ecg_", "resp_", "airflow_", "emg_", "eog_", "spo2_", "missing_",
)


class LeakageError(ValueError):
    """Raised when a forbidden predictor enters a model feature list."""


def assert_no_leakage(feature_names: Sequence[str]) -> None:
    """Reject identifiers, outcomes and post-PSG variables from predictors."""
    for name in feature_names:
        upper = str(name).upper()
        if upper in {value.upper() for value in FORBIDDEN_PREDICTORS}:
            raise LeakageError(f"Forbidden predictor '{name}' cannot be used as a model feature.")
        if any(token in upper for token in ("ICD", "FOLLOWUP", "LAST_VISIT", "TIME_TO_EVENT", "FILENAME")):
            raise LeakageError(f"Forbidden or post-PSG predictor '{name}' detected.")


def _max_frame_flag(frame, name: str, default: float = 0.0) -> float:
    """Conservative max of a per-row unit flag (any bad row disqualifies)."""
    if name not in frame.columns:
        return default
    values = pd.to_numeric(frame[name], errors="coerce").dropna()
    return float(values.max()) if not values.empty else default


def _min_frame_flag(frame, name: str, default: float = 0.0) -> float:
    """Conservative min of a per-row flag (all rows must be good)."""
    if name not in frame.columns:
        return default
    values = pd.to_numeric(frame[name], errors="coerce").dropna()
    return float(values.min()) if not values.empty else default


def eligible_signal_columns(frame, columns: Sequence[str]) -> list[str]:
    """Drop features that are not eligible under their channel's unit status.

    Unit provenance stays metadata: it is consumed here for eligibility only
    and never included as a predictor column.
    """
    eligible: list[str] = []
    for column in columns:
        modality = feature_modality(column)
        unknown = False
        harmonized = False
        if modality:
            lower = modality.casefold()
            unknown = (
                _max_frame_flag(frame, f"unit_unknown_channels_{lower}") > 0
                or _max_frame_flag(frame, f"unit_{lower}_unknown") > 0
            )
            harmonized = _min_frame_flag(frame, f"unit_harmonized_{lower}") == 1.0
        if is_feature_eligible(
            column, modality=modality, unit_unknown=unknown, unit_harmonized=harmonized
        ):
            eligible.append(column)
    return eligible


def encode_demographics(frame) -> "object":
    """Return numeric baseline demographics only (no identifiers or outcomes)."""
    output = pd.DataFrame(index=frame.index)
    output["Age"] = pd.to_numeric(frame.get("Age"), errors="coerce")
    output["BMI"] = pd.to_numeric(frame.get("BMI"), errors="coerce")

    # Explicit encoding: unknown/blank/missing/unrecognized values stay NaN so
    # the training-fold imputer handles them. They are never silently encoded
    # as female (0), which would fabricate a demographic value.
    sex = frame.get("Sex", pd.Series(index=frame.index, dtype=object))
    normalized = sex.astype(str).str.strip().str.casefold()
    encoded = pd.Series(np.nan, index=frame.index, dtype=float)
    encoded.loc[normalized.isin({"male", "m"})] = 1.0
    encoded.loc[normalized.isin({"female", "f"})] = 0.0
    output["Sex_male"] = encoded
    return output[list(DEMOGRAPHIC_FEATURES)]


def select_feature_matrix(
    frame, model_kind: str, allowed_signal_columns: Sequence[str] | None = None
) -> "object":
    """Select an allowed, unit-eligible feature subset for a named baseline.

    ``allowed_signal_columns`` must be computed once on the full labelled
    cohort. Evaluating per-partition eligibility would give train and test
    matrices different schemas whenever unit provenance differs by site.
    """
    demo = encode_demographics(frame)
    candidates = [
        column for column in frame.columns
        if column.startswith(SIGNAL_PREFIXES) and column not in demo.columns
    ]
    signal_columns = (
        list(allowed_signal_columns)
        if allowed_signal_columns is not None
        else eligible_signal_columns(frame, candidates)
    )
    signals = frame[signal_columns].apply(pd.to_numeric, errors="coerce") if signal_columns else pd.DataFrame(index=frame.index)
    if model_kind == "none":
        # The dummy baseline uses no features; keep an explicit neutral column.
        matrix = pd.DataFrame({"dummy_constant": 0.0}, index=frame.index)
    elif model_kind == "demographic":
        matrix = demo
    elif model_kind == "signal":
        matrix = signals
    elif model_kind == "combined":
        matrix = pd.concat([demo, signals], axis=1)
    else:  # pragma: no cover - guarded by make_model
        raise ValueError(f"Unknown feature kind: {model_kind}")
    assert_no_leakage(list(matrix.columns))
    return matrix


def make_model(name: str, random_state: int = RANDOM_SEED) -> Pipeline:
    """Construct one of the six documented baselines as a sklearn Pipeline."""
    specs: dict[str, tuple[str, object]] = {
        "dummy": ("none", DummyClassifier(strategy="prior")),
        "demo_lr": (
            "demographic",
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state),
        ),
        "signal_lr": (
            "signal",
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state),
        ),
        "signal_rf": (
            "signal",
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                random_state=random_state, n_jobs=1,
            ),
        ),
        "combined_lr": (
            "combined",
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state),
        ),
        "combined_hgb": (
            "combined",
            HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=random_state),
        ),
    }
    if name not in specs:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(sorted(specs))}")
    kind, estimator = specs[name]

    steps: list[tuple[str, object]] = []
    if kind != "none":
        steps.extend([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def feature_kind_for(name: str) -> str:
    """Return the permitted feature family for a named baseline."""
    return {
        "dummy": "none",
        "demo_lr": "demographic",
        "signal_lr": "signal",
        "signal_rf": "signal",
        "combined_lr": "combined",
        "combined_hgb": "combined",
    }[name]


def build_baselines(random_state: int = RANDOM_SEED) -> dict[str, Pipeline]:
    """Return all required baselines with a shared reproducible seed."""
    return {name: make_model(name, random_state) for name in
            ("dummy", "demo_lr", "signal_lr", "signal_rf", "combined_lr", "combined_hgb")}


def fit_predict(model: Pipeline, x_train, y_train, x_test):
    """Fit on training data only and return test-set probabilities."""
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    return probabilities


def model_config_hash(model: Pipeline) -> str:
    """Stable checksum of a fitted model's configuration and parameters."""
    params = {}
    for name, step in model.named_steps.items():
        params[name] = {key: repr(value) for key, value in sorted(step.get_params().items())}
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
