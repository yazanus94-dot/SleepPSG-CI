# Audit report — SleepPSG-CI

Inspection date: **2026-11-23**. Status: scaffold implemented, tested and linted;
**no real-data experiments were run**.

## Work completed

- Built the full modular package, scripts, benchmark, tests and CI from the
  supplied specification.
- Audited the official task specification, scorer, schema, label rules and data
  partitions against the official example code and challenge page.
- Implemented metadata auditing, access safeguards, channel inventory, quality
  checks, preprocessing, five modality feature extractors, six baselines,
  leave-one-source-out evaluation, the benchmark evaluator and provenance.
- Ran the complete automated test suite and Ruff locally.

## Files inspected

Official reference material (read-only, outside the new repository):

- `project-4-physionet-2026-psg-cognitive-impairment/evaluate_model.py` — official
  scorer (lines 27–45 headers, 47–198 metric functions, 201–334 evaluator).
- `.../helper_code.py` — headers (lines 22–44), demographic loaders (359–422),
  `sanitize_boolean_value` (646–653).
- `.../README.md` — project/data/methodology notes.
- Challenge page `https://moody-challenge.physionet.org/2026/` — task, label and
  data definition.
- Kaggle public API v1 `datasets/view` and `datasets/list` — dataset size and
  file inventory.
- Official example repository at commit `c0bcd78ddb892290b9d218be9be19660f4d8bdf3`
  and its BSD-3-Clause `LICENSE`.

**Starter ZIP:** the referenced SleepPSG-CI starter ZIP was not present in the
workspace. None of the files listed in section 3 of the instructions
(`AGENT_START.md`, `cohort.py`, `site_splits.py`, `audit_metadata.py`, etc.) existed
anywhere in the workspace. The scaffold was therefore created from the documented
architecture. No existing repository was overwritten or modified.

## Dataset access status

| Check | Result |
|---|---|
| Public metadata API reachable | Yes (unauthenticated) |
| Small training set size | 230,024,509,878 bytes (~214 GiB) |
| Large training set size | ~1.2 TiB (challenge page) |
| `ICD_codes_CI.csv` size | 88,731 bytes (Kaggle listing) |
| Selective single-file retrieval supported by Kaggle | Yes, with credentials |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` configured | No |
| `kaggle` CLI installed | No |
| Dataset content downloaded | **None** |
| Raw EDF recordings processed | **None** |

Blocker: Kaggle credentials and terms acceptance are required even for the
standalone `demographics.csv`. The repository fails closed with a clear error and
does not emit a fake empty report. No hidden validation/test labels were accessed.

## Exact verified counts

- Official example commit: `c0bcd78ddb892290b9d218be9be19660f4d8bdf3`.
- Official age gap δ: **2 years** for the primary AUROC and prevalence reward.
- Training sources: `S0001`, `I0002`, `I0006`; hidden `I0004`, `I0007`.
- Small-set physiological record counts: S0001 857, I0002 54, I0006 192.
- Large-set counts: S0001 5,139, I0002 319, I0006 1,142.
- Secondary metrics implemented: age-weighted AUROC, ROC-AUC, PR-AUC, Brier,
  calibration slope/intercept, sensitivity, specificity, balanced accuracy,
  confusion matrix and patient-level bootstrap CIs.
- Model baselines implemented: **6** (dummy, demo_lr, signal_lr, signal_rf,
  combined_lr, combined_hgb).

No real cohort counts (positives, negatives, exclusions, missing demographics)
can be reported: the labelled `demographics.csv` was never available locally.

## Test results

- Final local run: `python -m pytest` → **59 passed in 7.31s** (synthetic fixtures
  only; one synthetic-EDF integration test uses `edfio`).
- CLI smoke tests passed for `audit_metadata.py`, `inventory_dataset.py`,
  `train_baselines.py` and `evaluate_models.py` using synthetic CSVs in a
  git-ignored external temp directory. The site-held-out evaluator produced 3
  folds, 6 models per fold and 6 pooled OOF result entries in the smoke run.

## Lint results

- `ruff check .` (ruff 0.17.5) → **All checks passed!**
- CI matrix: Python 3.10, 3.11, 3.12; runs `pip install -e ".[dev]"`, pytest and
  Ruff without Kaggle credentials or PhysioNet downloads.

## Known issues

- No real-data validation of any module has been possible.
- Channel mappings are empty by default by design; a verified, site-specific
  mapping CSV must be supplied before physiological channels can be labelled.
- The official scorer's reward metric requires an external prevalence cohort; in
  local evaluation the training sources are used as the prevalence reference and
  this is documented in the results structure.
- `compute_auroc_age` returns NaN when no comparable positive/negative age pair
  exists; this is deliberate and reported as "not scoreable".

## Scientific limitations

- Only three labelled training sources exist, so site-held-out results cannot
  establish broad geographic generalization.
- The challenge label is used verbatim; no clinical outcome is reconstructed.
- Age >89 is encoded as 90 in the source data, which limits age-conditioning
  precision at the upper range.
- All metrics from the smoke run are synthetic software checks and are explicitly
  **not** clinical performance estimates. No model scores are claimed.

## Unfinished work

- Obtain Kaggle credentials and accept the BDSP terms to selectively retrieve
  `demographics.csv` (<500 MiB) and run the real metadata audit.
- Build a verified, site-specific channel-mapping configuration from official
  channel tables.
- Process a small, legally accessible subset of real EDF recordings and generate
  real per-recording features.
- Train and evaluate real site-held-out baselines and populate
  `benchmark/expected_results.json` with measured results only if they exist.
