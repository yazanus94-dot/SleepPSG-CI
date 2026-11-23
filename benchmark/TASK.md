# SleepPSG-CI Benchmark Task

## Scientific objective

Predict the officially defined future cognitive-impairment outcome from baseline
demographics and PSG-derived features, evaluated on an unseen source. Independent
challenge-inspired research; not an official PhysioNet entry. No team, placement,
percentile or score is claimed, and hidden validation/test labels are never used.

## Target definition

Use the supplied `Cognitive_Impairment` field verbatim via the shared
`normalize_label`. Only explicit positive/negative values (including text
booleans) enter scoring. `other`, blank, missing and unrecognized outcomes are
excluded, never converted to negatives. The official definition is in
`CHALLENGE_REFERENCES.md` and is not reconstructed locally.

## Input contract

A training CSV (with `Cognitive_Impairment`) and an evaluation CSV (without it),
each keyed by `SiteID` and `BDSPPatientID`, carrying `Age`, optional `BMI`/
`Sex_male`, and permitted physiological feature columns:

All **seven** modality families are permitted and are used by the reference
model: `eeg_*`, `ecg_*`, `resp_*`, `airflow_*`, `emg_*`, `eog_*`, `spo2_*`, plus
`missing_*`, `n_channels_*`, `signal_*`, `n_verified*`, `Age`, `BMI`, `Sex_male`.
Identifiers are supplied separately for alignment only. `unit_*` columns describe
signal-unit provenance and are **not** predictors.

Signal units are harmonized before feature extraction using explicit rules:
EEG/EOG/EMG → uV, ECG → mV, SpO2 → percent (fraction→percent only when header or
values confirm 0..1). Respiratory effort and airflow units are source-specific
and unconverted; only unit-invariant features should be interpreted and
`resp_signal_range` must not be compared across sites. Unknown units are flagged,
not silently equated. No real EDF recordings have been processed, so real-data
unit harmonization is unvalidated.

## Output contract

One row per evaluation record with `SiteID`, `BDSPPatientID` and
`Cognitive_Impairment_Probability` (finite, in `[0, 1]`). The target label must
not appear in evaluation inputs or predictions.

## Forbidden features

Identifiers (`SiteID`, `BDSPPatientID`, `patient_key`, `BidsFolder`,
`SessionID`), file names, outcome, diagnosis/ICD codes and all post-PSG
variables (follow-up dates/times, `Time_to_Event`, `Last_Known_Visit_Date`).

## Validation rules

The evaluator checks row count, exact ID alignment, finite `[0, 1]`
probabilities, duplicate evaluation IDs, exclusion of indeterminate labels and
that feature manifests contain no forbidden variables (explicit whitelist).
Reproducibility is checked by comparing two deterministic runs.

## Scoring criteria

Primary: exact official 2026 age-conditioned AUROC (δ = 2 years) with the
official BSD-3-Clause notice. Secondary metrics as in the main `TASK.md`.
Single-class or age-incomparable folds are reported as not scoreable.

## Limits of automated leakage validation

The evaluator validates the **data contract and predictor manifests only**. It
cannot inspect arbitrary participant Python model code or prove the absence of
leakage. Participants are responsible for their training procedure; grouping
identifiers must never be used as predictors.

## Reproducibility, edge cases, limitations

Fixed seeds and configuration checksums are required. Edge cases: single-class
folds, missing modalities, short/flatline/clipped channels, duplicate sessions,
indeterminate labels. Only three labelled sources exist; no real-data results
are published, and synthetic data is used only for software tests.
