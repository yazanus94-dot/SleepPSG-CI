"""Leave-one-site-out evaluation with explicit scoreability handling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .cohort import normalize_label
from .identity import add_patient_key
from .metrics import bootstrap_ci, compute_auroc_age, score_predictions
from .modeling import build_baselines, feature_kind_for, fit_predict, model_config_hash, select_feature_matrix
from .site_splits import assert_patient_frames_disjoint, fold_summary, leave_one_site_out

REQUIRED_EVAL_COLUMNS = ("SiteID", "BDSPPatientID", "Age", "Cognitive_Impairment")


def prepare_labelled(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only explicitly labelled 0/1 records; never recode indeterminate.

    Uses the same :func:`cohort.normalize_label` as the metadata audit, so text
    booleans such as ``"true"``/``"no"`` are handled identically. Composite
    patient identities are validated here, rejecting blank site/patient IDs.
    """
    missing = [column for column in REQUIRED_EVAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Evaluation table missing columns: {', '.join(missing)}")
    labels = frame["Cognitive_Impairment"].map(normalize_label)
    eligible = labels.isin([0, 1])
    prepared = frame.loc[eligible].copy()
    prepared["Cognitive_Impairment"] = labels.loc[eligible].astype(int)
    return add_patient_key(prepared)


def evaluate_site_held_out(
    frame: pd.DataFrame,
    model_names: Sequence[str] = ("dummy", "demo_lr", "signal_lr", "signal_rf", "combined_lr", "combined_hgb"),
    seed: int = 42,
    bootstrap_n_boot: int = 200,
) -> dict:
    """Run leave-one-site-out evaluation for each requested baseline.

    Hyperparameters are fixed a priori; no tuning is performed on a held-out
    site. All imputation/scaling occurs inside the sklearn pipeline fitted on
    training sources only. Confidence intervals use a patient-cluster bootstrap
    on the composite ``SiteID_BDSPPatientID`` identity.
    """
    labelled = prepare_labelled(frame)
    sites = labelled["SiteID"].astype(str)
    folds = leave_one_site_out(list(sites))
    results: dict = {
        "split_method": "leave-one-source-out",
        "seed": seed,
        "bootstrap": {
            "resampling_unit": "composite_patient",
            "n_boot": bootstrap_n_boot,
            "metrics_with_ci": ["auroc_age"],
        },
        "cohort": {
            "n_records": int(len(labelled)),
            "n_sites": int(sites.nunique()),
            "n_patients": int(labelled["patient_key"].nunique()),
            "n_positive": int((labelled["Cognitive_Impairment"] == 1).sum()),
            "n_negative": int((labelled["Cognitive_Impairment"] == 0).sum()),
        },
        "per_fold": [],
        "pooled_oof": {},
        "caveats": [
            "Only a few labelled training sources exist; this is not broad geographic validation.",
            "Pooled out-of-fold results mix folds with different training cohorts and are exploratory only.",
            "Single-class held-out sites are reported as not scoreable, never silently removed.",
            "Bootstrap CIs resample composite patients, keeping repeated sessions together.",
        ],
    }

    oof: dict[str, list[dict]] = {name: [] for name in model_names}
    for train_sites, held_out in folds:
        train_mask = sites.isin(train_sites)
        test_mask = sites == held_out
        train_df = labelled.loc[train_mask]
        test_df = labelled.loc[test_mask]
        assert_patient_frames_disjoint(train_df, test_df)

        fold = fold_summary(held_out, test_df["Cognitive_Impairment"].tolist(), train_sites)
        fold["models"] = {}
        y_train = train_df["Cognitive_Impairment"].to_numpy(dtype=int)
        y_test = test_df["Cognitive_Impairment"].to_numpy(dtype=int)
        ages_test = pd.to_numeric(test_df["Age"], errors="coerce").to_numpy(dtype=float)
        keys_test = test_df["patient_key"].to_numpy(dtype=object)
        prevalence_labels = y_train
        prevalence_ages = pd.to_numeric(train_df["Age"], errors="coerce").to_numpy(dtype=float)

        if len(np.unique(y_train)) < 2:
            fold["trainable"] = False
            fold["reason"] = "Training sources contain a single class; model cannot be fitted."
            results["per_fold"].append(fold)
            continue
        fold["trainable"] = True

        for name in model_names:
            kind = feature_kind_for(name)
            x_train = select_feature_matrix(train_df, kind)
            x_test = select_feature_matrix(test_df, kind)
            model = build_baselines(seed)[name]
            try:
                probabilities = fit_predict(model, x_train, y_train, x_test)
            except ValueError as exc:
                fold["models"][name] = {"error": str(exc), "scoreable": False}
                continue
            metrics = score_predictions(
                y_test,
                probabilities,
                ages_test,
                prevalence_labels=prevalence_labels,
                prevalence_ages=prevalence_ages,
            )
            metrics["model_config_sha256"] = model_config_hash(model)
            metrics["feature_kind"] = kind
            if bootstrap_n_boot > 0 and metrics["scoreable"]:
                metrics["auroc_age_ci"] = bootstrap_ci(
                    compute_auroc_age, y_test, probabilities, ages_test, keys_test,
                    n_boot=bootstrap_n_boot, seed=seed,
                )
            fold["models"][name] = metrics
            oof[name].append({
                "labels": y_test, "probabilities": probabilities,
                "ages": ages_test, "patient_keys": keys_test,
            })

        results["per_fold"].append(fold)

    for name, chunks in oof.items():
        if not chunks:
            results["pooled_oof"][name] = {"available": False, "reason": "No scoreable folds."}
            continue
        y = np.concatenate([chunk["labels"] for chunk in chunks])
        probs = np.concatenate([chunk["probabilities"] for chunk in chunks])
        ages = np.concatenate([chunk["ages"] for chunk in chunks])
        keys = np.concatenate([chunk["patient_keys"] for chunk in chunks])
        pooled = score_predictions(y, probs, ages)
        pooled["available"] = True
        pooled["caveat"] = "Exploratory pooled OOF; not an official score and not site-independent."
        if bootstrap_n_boot > 0 and pooled["scoreable"]:
            pooled["auroc_age_ci"] = bootstrap_ci(
                compute_auroc_age, y, probs, ages, keys,
                n_boot=bootstrap_n_boot, seed=seed,
            )
        results["pooled_oof"][name] = pooled
    return results


def comparison_table(results: dict) -> pd.DataFrame:
    """Build the demographic/signal/multimodal model comparison table."""
    rows: list[dict] = []
    for fold in results.get("per_fold", []):
        for name, metrics in fold.get("models", {}).items():
            if "auroc_age" not in metrics:
                continue
            ci = metrics.get("auroc_age_ci", {})
            rows.append({
                "model": name,
                "held_out_site": fold["held_out_site"],
                "feature_kind": metrics.get("feature_kind"),
                "n": metrics.get("n"),
                "n_positive": metrics.get("n_positive"),
                "n_negative": metrics.get("n_negative"),
                "scoreable": metrics.get("scoreable"),
                "auroc_age": metrics.get("auroc_age"),
                "auroc_age_ci_low": ci.get("ci_low"),
                "auroc_age_ci_high": ci.get("ci_high"),
                "auroc": metrics.get("auroc"),
                "auprc": metrics.get("auprc"),
                "brier": metrics.get("brier"),
            })
    return pd.DataFrame(rows)
