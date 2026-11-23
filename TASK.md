# Task specification

## Scientific objective

Predict the officially defined future cognitive-impairment outcome from baseline
demographics and PSG-derived features, evaluated by generalization to an unseen
source. Independent research only; not an official challenge entry.

## Target definition

Use the supplied `Cognitive_Impairment` field verbatim. Only explicit `0` and `1`
values are eligible. `other`, blank, missing and indeterminate outcomes are
excluded from binary training and scoring and are never converted to negatives.
The official positive/negative definition is documented in
`CHALLENGE_REFERENCES.md` and is not reconstructed locally.

## Permitted inputs

- Baseline demographics: `Age`, `Sex`, `BMI` (where permitted and available).
- Features derived from verified PSG channels: EEG, EOG, EMG, ECG, respiratory
  effort, airflow.
- Explicit missing-modality indicators.

## Forbidden inputs

`SiteID`, `BDSPPatientID`, `BidsFolder`, `SessionID`, file names,
`Cognitive_Impairment`, `Time_to_Event`, `Last_Known_Visit_Date`,
`Time_to_Last_Visit`, ICD/diagnosis information, and any post-PSG variable.
Identifiers are used only for grouping, alignment and leakage checks.

## Validation

`SiteID`/`BDSPPatientID` are grouping keys. Leave-one-source-out folds hold out an
entire source; patient disjointness is asserted; imputation, scaling and any
selection are fit on training sources only. Single-class held-out folds are
reported as not scoreable.

## Metrics

Primary: official age-conditioned AUROC (δ = 2 years). Secondary: age-weighted
AUROC, ROC-AUC, PR-AUC, Brier, calibration slope/intercept, sensitivity,
specificity, balanced accuracy, confusion matrix and bootstrap CIs.

## Data governance

Only the public labelled training partition may be used. Hidden validation/test
labels are never accessed or inferred. Raw data and patient-level records stay in
git-ignored directories. Reports contain aggregate statistics only.

## Reproducibility

Fixed seeds, pinned configuration checksums, package-version manifests and
deterministic JSON reports.
