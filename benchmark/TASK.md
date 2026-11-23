# SleepPSG-CI Benchmark Task

## Scientific objective

Predict the officially defined future cognitive-impairment outcome of a sleep-study
patient from baseline demographics and polysomnography-derived features. This is
an **independent, challenge-inspired benchmark**. It is not an official
PhysioNet submission, no team/leaderboard/score is claimed, and hidden
validation/test labels are never used.

## Target definition

Use the supplied `Cognitive_Impairment` column verbatim. Only explicit `0`
(negative) and `1` (positive) values enter training and scoring. `other`, blank,
missing and indeterminate values are **excluded**, never recoded as negative.

The official definition (challenge page, inspected 2026-11-23): positive =
two or more qualifying CI diagnoses, first at least 1 and at most 6 years after
the PSG with at least one week between diagnoses; negative = no CI diagnosis and
at least 6 years of encounters. Do not reconstruct this rule locally.

## Input contract

A training CSV and an evaluation CSV, each with:

| Column | Role |
|---|---|
| `SiteID` | grouping only (must NOT be a predictor) |
| `BDSPPatientID` | grouping only (must NOT be a predictor) |
| `Age` | permitted baseline demographic |
| `Sex`, `BMI` | permitted baseline demographics where available |
| `Cognitive_Impairment` | label; present in training, withheld from participants for evaluation |
| `*_power`, `*_relative`, `ecg_*`, `resp_*`, `emg_*`, `eog_*`, `missing_*` | permitted physiological features |

## Output contract

A predictions CSV with exactly one row per evaluation record and columns:

- `SiteID`
- `BDSPPatientID`
- `Cognitive_Impairment_Probability` — finite numeric value in `[0, 1]`

Optional: `Cognitive_Impairment` binary column in `{0, 1}`.

## Permitted features

Age, sex and BMI recorded at baseline, plus physiological features derived from
verified PSG channels and explicit missing-modality indicators.

## Forbidden features

Patient/session/site identifiers, file names, the outcome label itself,
diagnosis codes, `Time_to_Event`, `Last_Known_Visit_Date`,
`Time_to_Last_Visit`, follow-up duration and any post-PSG clinical variable.
Identifiers may be used only for row alignment and leakage checks.

## Validation rules

The evaluator checks row count, numeric/finite probabilities, `[0, 1]` range,
unique identifiers, exact row alignment, absence of forbidden predictors and
reproducibility under a fixed seed.

## Scoring criteria

Primary metric: the exact official 2026 age-conditioned AUROC (age gap δ = 2
years), implemented in `sleeppsg_ci.metrics.compute_auroc_age` with the official
BSD-3-Clause notice. Secondary metrics: age-weighted AUROC, ROC-AUC, PR-AUC,
Brier score, calibration slope/intercept, sensitivity, specificity and balanced
accuracy. A fold whose held-out source has only one class is reported as **not
scoreable**, never as zero and never silently dropped.

## Reproducibility requirements

Fix `random_state`, record package versions and configuration checksums, fit all
imputation/scaling inside the training data, and evaluate each held-out source
once.

## Edge cases

Single-class folds, missing modalities (explicit indicators, training-fold-only
imputation), short/flatline/clipped channels, duplicate sessions, indeterminate
labels and missing demographics.

## Known limitations

No real-data results are published (data access is blocked by missing Kaggle
credentials and the data are ~214 GiB / ~1.2 TiB). Synthetic data is used only
for software tests, never as a substitute for clinical performance. Only three
labelled training sources exist, so cross-source generalization estimates are
narrow and caveated.
