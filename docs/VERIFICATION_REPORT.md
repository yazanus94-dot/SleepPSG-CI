# Independent verification report — SleepPSG-CI

**Verification date:** 2026-11-26
**Repository:** `https://github.com/yazanus94-dot/SleepPSG-CI`
**Verified commit:** `935051f` (`main`)
**Method:** inspection of the source tree, execution of the full test suite and
linter in a clean local environment, read-only execution of the dataset access
audit, and inspection of the GitHub Actions run history.
**Scope:** software correctness and reproducibility claims only. No claim in this
document is based on real polysomnography data.

---

## 1. Headline result

| Question | Verified answer |
|---|---|
| Does the code implement the described pipeline? | **Yes** — 24 modules: metadata audit/join, EDF chunk reader, quality gates, preprocessing, 7 modality feature extractors, unit harmonisation, 6 baselines, site-held-out evaluation, patient-cluster bootstrap, provenance. |
| Do the automated tests pass? | **Yes** — `209 passed` (10 new guard tests added in this pass), `ruff check .` clean. |
| Is CI green on `main`? | **Yes** — run [`45372491516`](https://github.com/yazanus94-dot/SleepPSG-CI/actions/runs/45372491516), Python 3.10/3.11/3.12 matrix. |
| Was any **real** PSG data processed? | **No.** |
| Were any **real** performance metrics produced? | **No.** |
| Is credentialed access available to this project? | **No** — see §4. |
| Does the repository contain patient data? | **No** — see §5. |

---

## 2. What was verified, and how

### 2.1 Software inventory

| Component | File(s) | Verified by |
|---|---|---|
| Patient identity + leakage guards | `identity.py`, `site_splits.py` | tests + code inspection (`assert_patient_disjoint`, `patient_keys`) |
| Cohort audit and join | `cohort.py`, `cohort_join.py` | `test_cohort.py`, `test_cohort_join.py` |
| EDF chunk reader (no full materialisation) | `signal_reader.py` (`EdfChunkReader`) | `test_data_access.py`, `test_channel_audit.py` |
| Channel verification + quality gates | `channel_audit.py`, `quality.py` | `test_channel_audit.py`, `test_quality.py` |
| Unit harmonisation | `units.py` | `test_units.py` |
| Modality feature extractors | `features_eeg.py`, `features_ecg.py`, `features_eog.py`, `features_emg.py`, `features_resp.py`, `features_spo2.py`, resp/airflow | `test_features.py`, `test_feature_eligibility.py`, `test_feature_schema.py` |
| Six ML baselines | `modeling.py::build_baselines` | `test_modeling.py` |
| Published challenge metrics | `metrics.py` (`compute_reward`, `auroc_age`, `auroc_weighted`, AUROC, AUPRC, sensitivity, specificity, balanced accuracy, calibration slope/intercept) | `test_metrics.py`, `test_official_scorer.py` (differential test against the official scorer) |
| Patient-cluster bootstrap CIs | `metrics.py::bootstrap_ci` | `test_metrics.py` |
| Provenance / manifests | `provenance.py` (`sha256_file`, `run_manifest`) | `test_experiment.py` |
| Real-data experiment entry point | `scripts/evaluate_models.py` | `test_evaluate_models_guard.py` (new) |

The six baselines, read from `modeling.py`, are exactly:
`dummy`, `demo_lr`, `signal_lr`, `signal_rf`, `combined_lr`, `combined_hgb`.

### 2.2 Test execution (local, clean environment)

```
$ python -m pytest -q
209 passed

$ python -m ruff check .
All checks passed!
```

The 10 additional tests in `tests/test_evaluate_models_guard.py` pin the
source-partition guard used by the real-data entry point:

* all labels must come from `S0001`, `I0002`, `I0006`;
* hidden sources `I0004`, `I0007` are rejected;
* unknown sources are rejected;
* blank/absent `SiteID` values are rejected;
* returned sources are sorted and de-duplicated.

These tests are **synthetic/contract tests**. They prove the guard works; they do
not prove anything about clinical performance.

### 2.3 CI evidence

| Run | Branch | Result | Python |
|---|---|---|---|
| [`45372491516`](https://github.com/yazanus94-dot/SleepPSG-CI/actions/runs/45372491516) | `main` | success | 3.10 / 3.11 / 3.12 |
| [`45371622176`](https://github.com/yazanus94-dot/SleepPSG-CI/actions/runs/45371622176) | `main` | success | 3.10 / 3.11 / 3.12 |

---

## 3. Synthetic vs real: what the current evidence supports

| Evidence class | Source | Supports real-data claims? |
|---|---|---|
| 209 automated tests | Synthetic metadata, synthetic/mock EDFs, differential scorer tests | **No** |
| CI runs | Same synthetic suite on three Python versions | **No** |
| `feature_cohort.csv` (local, git-ignored, 10 rows) | Synthetic fixture | **No** |
| Read-only dataset inventory | Public Kaggle metadata API | **No** — inventory only, no signals |
| `docs/REAL_DATA_VALIDATION.md` | Written procedure | **No** — documented, never executed |

No confusion-matrix value, AUROC, AUPRC, reward, calibration slope, confidence
interval or cross-site generalisation number reported anywhere in this
repository is derived from real patient data.

---

## 4. Data access verification (executed, read-only)

Command:

```bash
python scripts/inventory_dataset.py
```

Observed output (verbatim excerpt):

```json
{
  "blocker": "No Kaggle API token, access-token file, legacy username/key pair, kaggle.json or OAuth credentials are configured. Selective retrieval of demographics.csv cannot be authenticated; no dataset content was downloaded. Configure credentials yourself and accept the BDSP data terms on your own Kaggle account before retrying.",
  "credential_source": "none",
  "kaggle_cli_available": false,
  "kaggle_credentials_present": false,
  "public_metadata_api_reachable": true,
  "raw_signatures_blocked": true,
  "selective_file_retrieval_supported": false,
  "download_allowlist": ["demographics.csv"],
  "download_cap_bytes": 500000000,
  "small_dataset_bytes": 230024509878,
  "large_dataset_bytes_approx": 1318230425600
}
```

Interpretation:

* Public dataset **metadata** is reachable, so the inventory itself is real.
* No credential source exists in this environment (`credential_source: none`).
* Raw signal retrieval is not possible (`selective_file_retrieval_supported:
  false`, `raw_signatures_blocked: true`).
* The access layer **fails closed**: it exits with a documented blocker rather
  than writing an empty or fabricated report.
* `data/` contains only `.gitkeep`. No EDF/BDF/CSV payload is present.

### Why this is a hard blocker, not a missing feature

The labelled training corpus is behind a credentialed data-use agreement:

* the small training set is ≈214 GiB and the large set ≈1.2 TiB;
* access requires a personal Kaggle account, acceptance of the BDSP data terms,
  and (for the clinical metadata) the data-use agreement;
* the credentials are personal and non-transferable — they cannot be supplied,
  borrowed, or reconstructed by a third party;
* hidden validation/test labels are off-limits by construction.

Producing "real-data results" without those credentials and without downloading
the permitted recordings would necessarily mean fabricating them. This project
does not do that.

### Exact owner actions required to unblock

1. Create/confirm a Kaggle account and accept the BDSP data terms personally.
2. Export credentials to one of the supported sources (`KAGGLE_API_TOKEN`,
   `~/.kaggle/kaggle.json`, `KAGGLE_CONFIG_DIR`, or OAuth credentials).
3. Allocate ≥250 GiB free disk for the **small** training set.
4. Run, in order:

   ```bash
   python scripts/inventory_dataset.py          # must report credentials present
   python scripts/download_metadata.py          # demographics.csv only, ≤500 MB cap
   python scripts/audit_metadata.py
   python scripts/extract_features.py           # authorised recordings only
   python scripts/build_feature_cohort.py
   python scripts/train_baselines.py
   python scripts/evaluate_models.py            # writes manifest + provenance
   ```

5. Publish results only after the manifest records input SHA-256 checksums.

Until step 4 completes, tasks 4–9 of the originating request (process real EDFs,
validate EEG/ECG/EOG/EMG/RESP/SpO2, patient-independent and site-held-out
experiments, six baselines on real data, published metrics, real confidence
intervals) remain **not executed**.

---

## 5. Licensing, privacy and non-claims

| Item | Status |
|---|---|
| Repository licence | MIT (`LICENSE`), third-party attribution in `NOTICE` |
| Patient data in repository | None. `.gitignore` excludes `data/*`, `*.edf`, `*.bdf`, `*.csv`, `*.tsv`, credentials and tokens |
| Identifiers in committed content | None — no patient-level records are committed |
| Data redistribution | Not performed and not permitted by the BDSP terms |
| Official PhysioNet Challenge participation | **Not claimed.** This is an independent retrospective study inspired by the challenge. No registration, team name, leaderboard position, percentile or challenge score is asserted |
| Clinical claims | None. Research software, not a medical device or diagnostic tool |
| Hidden labels | Never accessed or inferred |

---

## 6. Reproduction instructions

```bash
git clone https://github.com/yazanus94-dot/SleepPSG-CI.git
cd SleepPSG-CI
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python scripts/inventory_dataset.py               # expect the documented blocker
```

Expected result on a machine without credentials: 209 tests pass, Ruff is clean,
and the inventory exits with `kaggle_credentials_present: false`.

---

## 7. Limitations of this report

* It verifies software behaviour and access control, **not** scientific validity
  on real recordings.
* Feature extractors are exercised on synthetic and mocked signals; real-channel
  edge cases (missing leads, unusual sampling rates, unit ambiguities) can only
  be validated against authorised recordings.
* The 209-test suite cannot detect distribution shift, label noise or site
  effects present only in the real corpus.
* This report is an engineering verification artefact. It is not a peer review,
  not a regulatory assessment, and not evidence of clinical utility.
