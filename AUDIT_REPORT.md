# Audit report — SleepPSG-CI (real-data readiness sprint)

Inspection date: **2026-11-23**. Baseline commit: `48bfcdd...`.
Status: software implemented and tested on synthetic fixtures; **no real-data
experiments were run, no clinical model performance was measured, and no real EDF
recording was processed**. This sprint verified access, fixed two reproducible
real-data defects, and prepared the experiment entry point.

## Work completed

- Composite patient identity (`SiteID_BDSPPatientID`) introduced as the single
  grouping/bootstrap/join key across cohort, splits, evaluation and bootstrap.
- Label normalization unified: `evaluation.prepare_labelled` now uses the same
  `cohort.normalize_label` as the audit; indeterminate labels are excluded.
- Sex encoding fixed: unknown/blank/missing values remain NaN for the training
  fold imputer instead of being encoded as female.
- New `build_feature_cohort.py` + `cohort_join.py` join extracted features to
  labelled metadata with aggregate matched/unmatched/duplicated/excluded reports.
- Full-recording processing implemented via `signal_reader.EdfChunkReader` using
  `edfio` partial reads (`get_data_slice`); no `EdfSignal.data` materialization.
  30-second epochs are buffered across chunks and aggregated robustly.
- All verified channels per modality are retained with stable channel-specific
  prefixes; all **seven** modality extractors (EEG, ECG, RESP, AIRFLOW, EMG, EOG,
  SpO2) are included in signal-only and combined model feature selection.
- Explicit unit harmonization (`units.py`): EEG/EOG/EMG → uV, ECG → mV, SpO2 → %
  (fraction→percent only when confirmed), RESP/AIRFLOW unconverted. Original
  units and conversion status are kept as non-predictor `unit_*` metadata.
- Unit-aware feature eligibility: amplitude features from unknown/incompatible
  EEG/ECG/EOG/EMG/SpO2 channels and RESP/AIRFLOW signal-range features are gated
  out of model matrices; unit-invariant features remain eligible.
- Non-finite gaps split contiguous segments; ECG/respiratory features are not
  computed across gaps.
- Bootstrap is a true patient-cluster bootstrap and CIs are integrated into fold
  and pooled OOF results.
- Benchmark evaluator now enforces a predictor whitelist, rejects labels/IDs in
  evaluation inputs, excludes indeterminate labels and documents its limits.
- Cohort audit edge cases fixed (no double-counted blanks, composite patient
  counts, JSON-null age summaries, incomplete recording rows skipped).
- Differential tests against the official scorer added (self-contained copy with
  BSD notice); δ = 2 unchanged and the NaN-vs-ZeroDivisionError difference is
  explicit.
- Pyproject metadata updated to PEP 639 (`setuptools>=77`, SPDX `MIT`).

### This sprint (real-data readiness)

- Pulled `main`, confirmed clean tree at `48bfcdd`, re-ran 136 tests and Ruff, and
  confirmed the unit-safety gate is still present.
- Verified the public Kaggle metadata/file inventory and located `demographics.csv`
  at **131,863 bytes** (well under the 500 MiB cap). No file was downloaded.
- Fixed a reproducible real-data defect: duplicate EDF channel labels (e.g. two
  channels both named `EEG`) previously collapsed into one channel, silently
  discarding a recording. Labels are now deterministically disambiguated
  (`EEG`, `EEG__2`), raw labels are preserved for provenance, and modality rules
  resolve against the raw label.
- Fixed a reproducible epoch-count inconsistency: the epoch generator is now
  capped at the header-derived epoch count so a duration/sample mismatch cannot
  report more epochs than declared.
- Strengthened `scripts/evaluate_models.py` into the real-data experiment entry
  point: it accepts only `S0001`/`I0002`/`I0006`, rejects hidden/unknown sources,
  records SHA-256 input checksums, writes an explicit evaluation contract and
  separate manifest, and keeps the existing patient-grouped, δ = 2, bootstrap
  evaluation unchanged.
- Added regression tests for duplicate labels, epoch capping, source validation,
  provenance and the experiment contract.

## Files inspected

Official reference material (read-only): `project-4-physionet-2026-psg-cognitive-impairment/evaluate_model.py`
(scorer), `helper_code.py` (schema/labels), challenge page, Kaggle public API
metadata, official repo commit `c0bcd78ddb892290b9d218be9be19660f4d8bdf3` and
its BSD-3-Clause license. No existing repository was overwritten.

The original SleepPSG-CI starter ZIP was not present in the workspace; the
scaffold was created from the documented specification in the prior commit.

## Dataset access status

| Check | Result |
|---|---|
| Public metadata API reachable | Yes (unauthenticated) |
| Small training set size | 230,024,509,878 bytes (~214 GiB) |
| Large training set size | ~1.2 TiB |
| `demographics.csv` size | **131,863 bytes** (public file inventory, 2026-11-23) |
| Kaggle credentials configured | No (`KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLE_CONFIG_DIR` unset) |
| `kaggle` CLI installed | No |
| BDSP terms acceptance verified | Not possible without authenticated account |
| Dataset content downloaded | **None** |
| Real EDF recordings processed | **None** |

Blocker: Kaggle credentials and the user's own acceptance of the BDSP terms are
required even for the standalone `demographics.csv`. Terms were **not** accepted
on anyone's behalf and no authentication was bypassed. No hidden validation/test
labels were accessed.

**Phase 3 outcome:** no real metadata audit was performed because the file was
not legitimately retrievable; no results were fabricated. The privacy-safe audit
command is ready and documented.

**Exact next user action:** log into the Kaggle account, open
https://www.kaggle.com/datasets/physionet/physionetchallenge2026data, accept the
BDSP Credentialed Health Data License v1.0.0 personally, create a Kaggle API
token, install the `kaggle` CLI, configure `KAGGLE_USERNAME`/`KAGGLE_KEY`, then
run `python scripts/inventory_dataset.py` and the documented single-file
download followed by `python scripts/audit_metadata.py data/demographics.csv`.

## Exact verified counts

- Official commit: `c0bcd78ddb892290b9d218be9be19660f4d8bdf3`; age gap δ = 2.
- Training sources: `S0001`, `I0002`, `I0006`; hidden `I0004`, `I0007`.
- Small-set physiological records: S0001 857, I0002 54, I0006 192.
- Large-set: S0001 5,139, I0002 319, I0006 1,142.
- Baselines: 6. Modalities with extractors: 7 (EEG, ECG, RESP, AIRFLOW, EMG,
  EOG, SpO2). Default chunk 300 s, epoch 30 s.
- Real cohort counts (positives/negatives/exclusions) remain **unavailable**.

## Test results

- `python -m pytest` → **144 passed, 0 failed, 0 skipped** (synthetic fixtures only;
  includes synthetic EDFs, end-to-end join + CLI, chunk-read/no-`.data` test, gap
  tests, official-scorer differential tests, bootstrap cluster test, airflow/SpO2
  feature-selection tests, incomplete-identity counting tests, V/mV/uV unit
  equivalence tests, unit-aware feature-eligibility tests, duplicate-label and
  epoch-cap regressions, and experiment source/provenance tests).
- `ruff check .` → **All checks passed!**
- CLI smoke tests for all scripts passed on synthetic data; site-held-out
  evaluation produced 3 folds × 6 models with integrated patient bootstrap CIs.

## Lint results

Clean. CI matrix: Python 3.10, 3.11, 3.12; no Kaggle credentials required.

## Known issues

- No real-data validation of any module.
- Channel mappings are empty by default; a verified site-specific CSV is required.
- The official reward needs an external prevalence cohort; local evaluation uses
  the training sources as prevalence reference and records this in the results.
- `compute_auroc_age` returns NaN for unscoreable cohorts by design, unlike the
  official raw scorer's ZeroDivisionError (made explicit in differential tests).

## Scientific limitations

Only three labelled sources exist; no broad geographic generalization claim.
The challenge label is used verbatim; no outcome is reconstructed. Ages >89 are
encoded as 90 in the source. All smoke-run outputs are synthetic software checks,
**not** clinical performance. Full-night chunking and unit-harmonization rules
are validated on synthetic EDFs only. SpO2 features are descriptive and not
event-based hypoxemia indices. RESP/AIRFLOW units are unconverted, so
`resp_signal_range` is not comparable across sites; unit differences limit
cross-site validation until real EDFs are processed.

## Unfinished work / blockers

- Obtain Kaggle credentials and accept BDSP terms to selectively retrieve
  `demographics.csv` (<500 MiB) and run the real metadata audit.
- Build a verified site-specific channel mapping from official channel tables.
- Process legitimately accessible real EDF subsets and produce real features.
- Train/evaluate real site-held-out baselines; populate `expected_results.json`
  only with genuinely measured results.
