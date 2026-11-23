# Agent / developer start guide

This repository was built from the supplied SleepPSG-CI specification. The
referenced starter ZIP was not present in the workspace at build time; the
scaffold was therefore created from the documented architecture, and no existing
repository was modified.

## First actions

1. `python -m pip install -e ".[dev]"`
2. `python -m pytest -q` and `ruff check .`
3. `python scripts/inventory_dataset.py` — read-only access audit.
4. Provide a labelled training `demographics.csv` under `data/` (git-ignored) and
   run `python scripts/audit_metadata.py data/demographics.csv`.

## Governance reminders

- Never commit raw EDF/BDF/VHDR files, patient-level rows, credentials, `.env`
  files or personal CV material.
- Never access hidden validation/test labels.
- Never use identifiers, outcomes, diagnosis codes or post-PSG variables as
  predictors.
- Fit imputation/scaling only on training folds; report single-class folds as
  not scoreable instead of dropping them.

## Architecture

`cohort` / `site_splits` / `data_access` → `channel_audit` → `quality` /
`preprocessing` / `epoching` → `features_*` / `feature_table` → `modeling` /
`metrics` / `evaluation` → `benchmark` evaluator. `provenance` records versions
and checksums for every run.
