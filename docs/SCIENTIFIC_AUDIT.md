# Scientific audit — SleepPSG-CI

## A. Scoring (verified)

- `scoreable` is now True only when the **primary** age-conditioned AUROC is finite. A fold with
  two classes but no comparable age pair is explicitly reported as not scoreable.
- Single-class held-out sites are reported, never silently removed.
- AUROC/AUROC-age and prevalence use the documented delta-2 definition (differential tests vs
  the cited scorer).
- Bootstrap resamples composite `SiteID_BDSPPatientID` identities, keeping repeated sessions of
  a patient together.
- Indeterminate labels are excluded by `normalize_label`, never recoded negative.
- Pooled OOF results are labelled exploratory in `evaluation.py` caveats; they are not presented
  as official challenge performance.

## B. Feature-schema consistency

Unit-eligible columns are computed **once on the full labelled cohort** and passed to each fold.
The unit metadata used (`unit_unknown_channels_*`, `unit_harmonized_*`) is measurement
provenance available at deployment for a new site — it is **not** label, outcome or performance
information. No held-out labels or model scores select features.

Limitation (disclosed): this is transductive with respect to *unit metadata of the cohort*; if a
deployment site's unit provenance is unknown, its eligibility must be resolved from its own
header audit. The rule is prespecified and auditable; a strictly inductive alternative would
freeze the eligibility rule at training time and apply the same unit predicates per site.

## C. scikit-learn warning (fixed, not suppressed)

`UserWarning: Skipping features without any observed values: ['Sex_male']` occurred when a
training fold contained no recorded sex values. The pipeline now includes a
`_DropAllMissing` step fitted on training data: columns with no observed training values are
dropped from both train and test via a frozen schema. This preserves alignment, avoids the
warning legitimately (no fabricated sex value), and is covered by tests in
`tests/test_feature_schema.py`. All 199 tests pass with no such warning.

## D. Data access

All Kaggle/BDSP safeguards are preserved: no terms accepted, no authentication, no dataset
download, no hidden-label access. Real EDF validation remains **documented as pending** in
`docs/REAL_DATA_VALIDATION.md` until authorized data is obtained and processed by the owner.
