"""Honest constant-prevalence starter solution for the SleepPSG-CI benchmark.

Uses no physiological features and no identifiers as predictors. The probability
is the training-set positive prevalence, clipped to (0, 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMNS = ("SiteID", "BDSPPatientID")
LABEL = "Cognitive_Impairment"
PROBABILITY = "Cognitive_Impairment_Probability"


def predict(train: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    """Return one probability per evaluation row using training prevalence only."""
    labels = pd.to_numeric(train[LABEL], errors="coerce")
    labels = labels[labels.isin([0, 1])]
    prevalence = float(labels.mean()) if len(labels) else 0.5
    prevalence = float(np.clip(prevalence, 1e-6, 1 - 1e-6))
    return pd.DataFrame({
        "SiteID": evaluation["SiteID"].to_numpy(),
        "BDSPPatientID": evaluation["BDSPPatientID"].to_numpy(),
        PROBABILITY: np.full(len(evaluation), prevalence, dtype=float),
    })
