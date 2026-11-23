# Task specification

## Scientific objective

Predict the officially defined future cognitive-impairment outcome from baseline
demographics and PSG-derived features, evaluated by generalization to an unseen
source. Independent research only; not an official challenge entry.

## Target definition

Use the supplied `Cognitive_Impairment` field verbatim via
`cohort.normalize_label` (shared with the audit and evaluation layers). Only
explicit `0`/`1` (including text booleans such as `true`/`no`) enter binary
training/evaluation. `other`, blank, missing and unrecognized values are
excluded, never recoded as negatives. The official positive/negative definition
is documented in `CHALLENGE_REFERENCES.md` and is not reconstructed locally.

## Patient identity

The composite key `SiteID_BDSPPatientID` is the sole patient identity. It is used
for unique-patient counts, duplicate detection, split grouping, bootstrap
resampling and join validation. Missing/blank site or patient IDs are rejected
where a key is required. A bare patient ID may repeat across sites.

## Permitted inputs

- Baseline demographics: `Age`, `Sex`, `BMI` (unknown sex stays NaN until imputation).
- Physiological features from verified channels: all **seven** modality
  families — EEG, ECG, RESP, AIRFLOW, EMG, EOG and SpO2 (descriptive saturation
  summaries only). Signal-only and combined baselines use every family.
- Explicit `missing_<modality>` indicators.

## Forbidden inputs

`SiteID`, `BDSPPatientID`, composite `patient_key`, `BidsFolder`, `SessionID`,
file/recording names, `Cognitive_Impairment`, `Time_to_Event`,
`Last_Known_Visit_Date`, `Time_to_Last_Visit`, ICD/diagnosis information and any
post-PSG variable. Grouping identifiers are used only for validation/output.

## Feature-to-cohort pipeline

`build_feature_cohort.py` joins each BIDS-named feature row to exactly one
eligible metadata record on `(BidsFolder, SessionID)`, validates site
consistency, rejects duplicate/ambiguous keys, attaches baseline variables and
the normalized label, and emits aggregate matched/unmatched/duplicated/excluded
counts. The resulting cohort stays in git-ignored storage.

## Signal processing

Full available recording is processed with bounded partial EDF reads
(`get_data_slice`); 30-second epochs are aggregated across chunks with documented
robust statistics. Non-finite gaps split contiguous segments; time-dependent
features are never computed across gaps.

Explicit unit harmonization: EEG/EOG/EMG → uV, ECG → mV, SpO2 → percent (fraction
units converted only when header/values confirm 0..1). RESP/AIRFLOW units are
source-specific and unconverted; only unit-invariant features are interpreted
and `resp_signal_range` must not be compared across sites. Unknown units are
flagged, never silently equated. Original units and conversion status are kept
as `unit_*` provenance metadata and are excluded from model predictors.

Predictor eligibility is enforced before matrix construction: amplitude-dependent
features from channels with unknown/incompatible units are excluded, SpO2
thresholds require a confirmed percent scale, and RESP/AIRFLOW signal-range
features are excluded unless harmonized. Unit-invariant features and missingness
indicators remain eligible; remaining gaps are handled by fold-local imputation.
No real EDF recordings have been processed, so cross-site unit harmonization is
unvalidated.

## Validation

Leave-one-source-out folds hold out an entire source; composite patients remain
grouped; imputation/scaling fit on training sources only. Single-class folds are
reported as not scoreable. Bootstrap CIs resample composite patients as clusters.

## Metrics

Primary: official age-conditioned AUROC (δ = 2 years; differential tests against
the public scorer). Secondary: age-weighted AUROC, ROC-AUC, PR-AUC, Brier,
calibration slope/intercept, sensitivity, specificity, balanced accuracy,
confusion matrix and patient-cluster bootstrap CIs.

## Data governance

Only the public labelled training partition may be used. Hidden validation/test
labels are never accessed or inferred. Raw data and patient-level records stay
in git-ignored directories; public reports are aggregate only.
