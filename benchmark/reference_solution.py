"""Transparent reference solution using the verified modeling pipeline."""

from __future__ import annotations

import pandas as pd

from sleeppsg_ci.modeling import (
    RANDOM_SEED,
    build_baselines,
    feature_kind_for,
    fit_predict,
    select_feature_matrix,
)

ID_COLUMNS = ("SiteID", "BDSPPatientID")
LABEL = "Cognitive_Impairment"
PROBABILITY = "Cognitive_Impairment_Probability"


def predict(train: pd.DataFrame, evaluation: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Train a combined multimodal logistic baseline and score the evaluation rows."""
    train = train[pd.to_numeric(train[LABEL], errors="coerce").isin([0, 1])].copy()
    kind = feature_kind_for("combined_lr")
    x_train = select_feature_matrix(train, kind)
    x_eval = select_feature_matrix(evaluation, kind)
    y_train = train[LABEL].to_numpy(dtype=int)
    model = build_baselines(seed)["combined_lr"]
    probabilities = fit_predict(model, x_train, y_train, x_eval)
    return pd.DataFrame({
        "SiteID": evaluation["SiteID"].to_numpy(),
        "BDSPPatientID": evaluation["BDSPPatientID"].to_numpy(),
        PROBABILITY: probabilities,
    })
