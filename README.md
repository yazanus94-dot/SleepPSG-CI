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
| **Tested on real data** | **No** — no Kaggle credentials/BDSP terms acceptance; no labelled CSV or EDF downloaded. Public metadata inventory verified: `demographics.csv` is 131,863 bytes, but authenticated retrieval is blocked |
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

### 3.1 Verified access status (inspected 2026-11-23)

- Public Kaggle metadata API and file inventory are reachable without credentials.
- The small training set is 230,024,509,878 bytes (~214 GiB); the large set is
  ~1.2 TiB. Neither is downloaded.
- `demographics.csv` exists in the small training set and is **131,863 bytes**
  (~129 KiB) — comfortably below the 500 MiB selective-retrieval cap.
- Labelled training sources: `S0001`, `I0002`, `I0006`. Hidden sources `I0004`
  and `I0007` are never used as labelled data.
- Local blockers: no Kaggle API token, access-token file, OAuth credentials,
  username/key pair or legacy `kaggle.json` is configured, and the `kaggle` CLI is
  not installed. Acceptance of the BDSP Credentialed Health Data License v1.0.0
  cannot be verified without an authenticated account, and terms must never be
  accepted on another person's behalf.

#### Supported Kaggle CLI authentication methods

Verified against the official Kaggle CLI documentation
(`Kaggle/kaggle-api/docs/README.md`, authentication section) and CLI/SDK source
on 2026-11-23. Detection reports only the **source label** and a presence
boolean — never a token, key or file value.

| Method | Where the user configures it | Our source label |
|---|---|---|
| **API token (current)** — "Generate New Token" at https://www.kaggle.com/settings/api | `KAGGLE_API_TOKEN` env (value, or a path to a token file) | `kaggle_api_token_env` |
| **API token file (current)** | `~/.kaggle/access_token` (or `access_token.txt`) | `access_token_file` |
| **OAuth (current)** — user runs `kaggle auth login` interactively | `~/.kaggle/credentials.json` written by the CLI | `oauth_credentials` |
| **Legacy key pair** | `KAGGLE_USERNAME` + `KAGGLE_KEY` env | `environment` |
| **Legacy API key file** — "Create Legacy API Key" at https://www.kaggle.com/settings/api | `~/.kaggle/kaggle.json` (`{"username", "key"}`) | `home_kaggle_json` |
| **Legacy key in custom dir** (supported by the installed CLI) | `$KAGGLE_CONFIG_DIR/kaggle.json` | `kaggle_config_dir` |
| **Legacy key via XDG fallback** (Linux CLI behaviour) | `$XDG_CONFIG_HOME/kaggle/kaggle.json` | `xdg_kaggle_json` |

Do not confuse the current **Generate New Token** API-token workflow with the
**legacy Create Legacy API Key** `kaggle.json` workflow. We never run
`kaggle auth login` or accept terms automatically.

#### CLI installation requirement

The current Kaggle CLI declares **Python 3.11+**. SleepPSG-CI itself remains
compatible with its declared Python 3.10+ range; keep the CLI in an isolated
environment so it does not affect the project interpreter:

```bash
pipx install kaggle        # recommended
# or: uv tool install kaggle
# alternative: python -m pip install kaggle   (in a Python 3.11+ environment)
```

**Exact manual setup and retrieval commands** (run these yourself; no terms are
accepted programmatically):

```bash
# 1. Project environment (Python 3.10+ is fine)
python -m pip install -e ".[dev]"

# 2. Install the CLI in its own Python 3.11+ environment (pipx or uv)
pipx install kaggle    # or: uv tool install kaggle

# 3. Configure ONE authentication method, e.g. the current API token:
#    https://www.kaggle.com/settings/api -> "Generate New Token"
export KAGGLE_API_TOKEN='...'          # or write the raw token to ~/.kaggle/access_token
# Legacy alternative: "Create Legacy API Key" -> ~/.kaggle/kaggle.json
# (or KAGGLE_USERNAME/KAGGLE_KEY, or $KAGGLE_CONFIG_DIR/kaggle.json)
# OAuth alternative: run `kaggle auth login` yourself; we never auto-login.

# 4. Personally ACCEPT the BDSP terms at
#    https://www.kaggle.com/datasets/physionet/physionetchallenge2026data

# 5. Read-only access audit (shows only presence/source, never values)
python scripts/inventory_dataset.py

# 6. Download ONLY demographics.csv. The explicit flag is your assertion that
#    you already accepted the terms; the script never accepts them for you.
python scripts/download_metadata.py \
  --confirm-bsdp-terms-accepted \
  --destination data \
  --output-report outputs/metadata_download_report.json

# 7. Real aggregate metadata audit (never commit the patient-level CSV)
python scripts/audit_metadata.py data/demographics.csv \
  --output outputs/metadata_audit.json
```

`download_metadata.py` verifies the remote size from the public Kaggle file
inventory, refuses anything above 500 MiB *before* downloading, and checks the
extracted file size against the remote size afterwards. Downloaded data stays in
the git-ignored `data/` directory.

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
indicators) are candidates for the signal-only and combined matrices; identifiers,
recording names, labels and unit-provenance columns never are.

**Unit-aware eligibility gate.** `units.is_feature_eligible` filters candidates
before matrix construction: amplitude-dependent EEG/ECG/EOG/EMG/SpO2 features are
dropped when any contributing channel has an unknown/incompatible unit; SpO2's
90%-threshold fraction is dropped without a confirmed percent scale; and
`resp_signal_range`/`airflow_signal_range` are dropped unless those units were
explicitly harmonized (the default pipeline never harmonizes them). Explicitly
unit-invariant features (band-power ratios/relative powers, entropy, spectral
edge, time-domain HRV, activity fraction, blink counts, respiratory rate/CV/
regularity) and missingness indicators remain eligible. Excluded features simply
do not enter the matrix; any remaining gaps are handled by the training-fold
imputer. Nothing is silently converted or guessed, and `unit_*` columns remain
provenance metadata only.

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

# 6. Train a baseline / run the reproducible site-held-out experiment
python scripts/train_baselines.py data/feature_cohort.csv --model combined_lr
python scripts/evaluate_models.py data/feature_cohort.csv \
  --metadata-csv data/demographics.csv \
  --output outputs/evaluation_results.json \
  --comparison outputs/comparison_table.csv \
  --manifest-output outputs/experiment_manifest.json
```

`evaluate_models.py` accepts only `S0001`, `I0002` and `I0006`, rejects hidden
sources, records SHA-256 checksums of the feature cohort and demographics in a
provenance manifest, keeps composite patients grouped, uses fold-local
imputation, applies the δ = 2 age-conditioned AUROC, reports unscoreable folds,
and writes patient-cluster bootstrap CIs. It writes aggregate results only.

## 11. Tests and results

```bash
python -m pytest -q
ruff check .
```

CI runs without Kaggle credentials or PhysioNet downloads. Evidence categories
are kept strictly separate:

- **Implemented software:** the full pipeline (see status table).
- **Synthetic test results:** 176 automated tests (32 mocked data-access tests
  covering all supported credentials, size verification and confirmation gating)
  exercise algorithm behavior without real downloads; these are software checks,
  never clinical performance.
- **Real metadata audit results:** none — `demographics.csv` was not retrieved.
- **Real EDF processing results:** none — no recording was downloaded or read.
- **Actual model-performance results:** none — no real patient-data model was fitted.

`benchmark/expected_results.json` states this explicitly. No hidden labels,
official score or competition participation is claimed.

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
