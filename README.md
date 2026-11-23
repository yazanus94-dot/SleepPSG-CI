# SleepPSG-CI

**SleepPSG-CI: Multimodal Polysomnography-Based Cognitive Impairment Risk Prediction**

> **Independence statement.** This is an independent retrospective research and
> benchmarking project inspired by the George B. Moody PhysioNet Challenge 2026.
> It is **not** an official competition entry; no registration, participation,
> team name, leaderboard placement, percentile or challenge score is claimed.
> Hidden validation/test labels were never accessed or inferred. This is research
> software only — not a diagnostic system, medical device or clinical
> decision-support tool.

---

## Status at a glance

| Category | Status |
|---|---|
| **Implemented software** | Full package: metadata audit & join, EDF chunk reader, quality checks, preprocessing, **7 modality feature extractors**, explicit unit harmonization, 6 baselines using all signal families, site-held-out evaluation, patient-cluster bootstrap, benchmark evaluator |
| **Tested on synthetic fixtures** | Yes — all automated tests use synthetic metadata/EDFs (including a full end-to-end join and chunk-read tests) |
| **Tested on real data** | **No** — no Kaggle credentials/BDSP terms acceptance; no labelled CSV or EDF downloaded |
| **Model training actually completed** | Only on synthetic software-test fixtures; no real patient-data model was fitted |
| **Real performance measured** | **None** — no clinical metrics or scores are claimed |

See [`AUDIT_REPORT.md`](AUDIT_REPORT.md) for exact counts and limitations.

---

## 1. Project overview and research question

SleepPSG-CI studies whether overnight polysomnography (PSG) signals plus baseline
demographics can predict the challenge-defined future cognitive-impairment
outcome, with emphasis on leakage-resistant validation, deterministic processing
and honest reporting rather than leaderboard performance.

> Can a model trained on labelled PSG recordings from some contributing sources
> generalize to recordings from an unseen source when predicting future
> cognitive-impairment risk?

## 2. Official challenge inspiration

Inspired by the 2026 George B. Moody PhysioNet Challenge (official phase
concluded; results released 2026-09-28). The official task, data dictionary,
example code and scorer were inspected on 2026-11-23; exact details, the scorer
commit and licenses are in [`CHALLENGE_REFERENCES.md`](CHALLENGE_REFERENCES.md).

## 3. Dataset and access

Public labelled training data: small (~214 GiB; Kaggle `totalBytes` =
230,024,509,878) and large (~1.2 TiB) training sets. Training sources: `S0001`,
`I0002`, `I0006`. Hidden sources `I0004` and `I0007` are never used as labelled
data. The read-only access audit runs with:

```bash
python scripts/inventory_dataset.py
```

Selective retrieval of `demographics.csv` requires Kaggle credentials and BDSP
terms acceptance. **No dataset content was downloaded.** If unavailable, the
metadata command exits with a clear error and never writes a fake empty report.

## 4. Data governance and privacy

Raw recordings, patient-level rows, identifiers, credentials and local
environment files are excluded by `.gitignore`. Public reports contain aggregate
statistics only. Indeterminate labels are excluded, never recoded as negatives.
Composite patient identity (`SiteID_BDSPPatientID`) is used for grouping,
bootstrap resampling and leakage checks — never as a predictor. Model predictors
are restricted to demographics and verified physiological features; see
[`TASK.md`](TASK.md) and [`benchmark/TASK.md`](benchmark/TASK.md).

## 5. Modalities and feature extraction

Implemented extractors, each producing documented night-level features from
verified channels:

- **EEG** — delta/theta/alpha/sigma/beta absolute & relative power, spectral
  ratios, entropy, 95% spectral edge frequency.
- **ECG** — R-peak detection, HR/HRV (SDNN, RMSSD, pNN50) with beat-quality flags
  (ECG-derived only, never PPG-derived PRV).
- **Respiration (RESP)** and **AIRFLOW** — respiratory rate, amplitude CV,
  regularity, signal range.
- **EMG** — RMS, mean absolute amplitude, activity fraction, spectral centroid.
- **EOG** — amplitude, blink-like events, slow/fast ratio.
- **SpO2** — mean/median/min/std saturation and fraction below 90% (descriptive
  oximetry summaries, not event-based apnea/hypopnea scoring).

These are **seven modality extractors** in total. Every channel with a verified
mapping contributes: channel-specific features use stable
`<modality>_ch_<channel>_` prefixes, and unprefixed features are the mean across
channels. Missing modalities are explicit `missing_<modality>` indicators; no
physiological value is fabricated. Airflow and SpO2 are first-class modality
families and are included in model feature selection (see section 7).

## 6. Full-night signal processing

`EdfChunkReader` uses `edfio`'s `get_data_slice(start, stop)` for genuine partial
reads and never accesses `EdfSignal.data`. Fixed-duration 30-second epochs are
buffered across configurable chunks (default 300 s) and aggregated with robust
statistics (mean/median/std). Epochs are anchored at sample 0, so results are
equivalent across chunk sizes. Per-recording outputs record duration, valid and
excluded epoch counts, channel counts and gap/segment counts. Header-only
inspection derives duration from `Edf.duration`. Quality audits aggregate
per-chunk reports. Non-finite samples split signals into contiguous segments;
time-dependent ECG/respiratory features are never computed across gaps.

> This is **all-recording** processing (the full available EDF signal), validated
> on synthetic EDFs. It has **not** been run on real clinical recordings.

Preprocessing primitives (detrending, baseline correction, Butterworth SOS
band-pass with documented padding, IIR notch, polyphase anti-aliased resampling)
are in `preprocessing.py`.

### 6.1 Signal-unit harmonization (`units.py`)

EDF physical dimensions are read from headers and converted using explicit,
tested rules before feature computation:

| Modality | Canonical unit | Rule |
|---|---|---|
| EEG / EOG / EMG | **uV** | V → 1e6, mV → 1e3, uV → 1, nV → 1e-3 |
| ECG | **mV** | V → 1e3, uV → 1e-3, nV → 1e-6, mV → 1 |
| SpO2 | **%** | fraction/ratio units ×100 **only** when the EDF physical range or values confirm 0..1; otherwise flagged and left untouched |
| RESP / AIRFLOW | source-specific | **no conversion invented**; only unit-invariant features (rate, amplitude CV, regularity) are interpreted |

`resp_signal_range` is unit-dependent and must not be compared across sites with
different sensor units. Unrecognized or incompatible units are flagged
(`unit_*_unknown`) rather than silently treated as equivalent. Original units,
canonical units, scale factors and statuses are preserved in `unit_*` columns of
the feature table and in `RecordingAudit.unit_handling`; these are **provenance
metadata, not predictors** and are excluded from model feature matrices.

> Unit equivalence has been verified on synthetic signals expressed in V, mV and
> uV. **Real EDF unit harmonization has not been validated** because no real
> recordings were processed. Unit differences remain a cross-site limitation.

## 7. Machine-learning baselines

Six scikit-learn pipelines with fold-local imputation/scaling: `dummy`,
`demo_lr`, `signal_lr`, `signal_rf`, `combined_lr`, `combined_hgb`. Unknown or
missing sex values remain NaN until the training-fold imputer handles them.
After the modality-selection fix, **all supported signal feature families**
(`eeg_`, `ecg_`, `resp_`, `airflow_`, `emg_`, `eog_`, `spo2_` and `missing_`
indicators) enter the signal-only and combined matrices; identifiers, recording
names, labels and unit-provenance columns never do.

## 8. Patient/site validation and metrics

Leave-one-source-out evaluation holds out an entire site, keeps composite
patient identities grouped, tunes nothing on the held-out site, and asserts
patient disjointness. Single-class folds are reported as **not scoreable**.

Primary metric: the exact official age-conditioned AUROC (δ = 2 years), adapted
from the official scorer with its BSD-3-Clause notice preserved. Secondary:
age-weighted AUROC, ROC-AUC, PR-AUC, Brier, calibration slope/intercept,
sensitivity, specificity, balanced accuracy, confusion matrix and **patient-cluster
bootstrap CIs** (composite patients resampled as whole units, integrated into
fold and pooled OOF results).

## 9. Reproducibility

Fixed seeds, deterministic reports (`sort_keys=True`), configuration checksums
and a provenance manifest accompany every run. Official example commit:
`c0bcd78ddb892290b9d218be9be19660f4d8bdf3`.

## 10. Installation and commands

```bash
python -m pip install -e ".[dev]"
```

Requires Python 3.10+; CI covers 3.10, 3.11 and 3.12.

```bash
# 1. Audit the labelled demographics (fails clearly if absent)
python scripts/audit_metadata.py data/demographics.csv --output outputs/metadata_audit.json

# 2. Read-only dataset-access audit
python scripts/inventory_dataset.py

# 3. Inspect EDF channels (chunked quality audit)
python scripts/audit_channels.py RECORDING.edf --mapping channel_mapping.csv

# 4. Extract full-recording features (BIDS names required for the join)
python scripts/extract_features.py physio/S0001/sub-S0001111197789-ses1.edf \
  --mapping channel_mapping.csv --output data/features.csv

# 5. Join features to metadata -> local, git-ignored feature cohort
python scripts/build_feature_cohort.py data/demographics.csv data/features.csv \
  --output data/feature_cohort.csv --report outputs/join_report.json

# 6. Train a baseline / run site-held-out evaluation
python scripts/train_baselines.py data/feature_cohort.csv --model combined_lr
python scripts/evaluate_models.py data/feature_cohort.csv
```

## 11. Tests and results

```bash
python -m pytest -q
ruff check .
```

CI runs without Kaggle credentials or PhysioNet downloads. **No real-data
results are published**; `benchmark/expected_results.json` states this
explicitly. Synthetic fixtures are software checks only, never clinical
performance.

## 12. Limitations

No real-data validation; only three labelled sources; site-held-out estimates
are narrow; channel mappings require external verification; the official metric
is undefined for single-class or age-incomparable folds; chunk processing is
validated on synthetic EDFs only.

## 13. References and licenses

Official example code © 2026 George B. Moody PhysioNet Challenges, BSD-3-Clause;
notice preserved in `src/sleeppsg_ci/metrics.py` and [`NOTICE`](NOTICE). Dataset
© BDSP under the BDSP Credentialed Health Data License v1.0.0. Project code is
MIT-licensed (see [`LICENSE`](LICENSE)).

Architecture: metadata audit → access audit → channel inventory → chunked EDF
reader → quality/preprocessing → per-modality features → feature-to-cohort join
→ baselines → site-held-out evaluation with patient bootstrap → benchmark evaluator.
