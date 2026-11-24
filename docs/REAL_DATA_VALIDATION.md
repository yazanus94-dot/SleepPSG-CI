# Real-data validation — reproducible workflow (NOT EXECUTED)

**Status: no real EDF recordings or patient-level labels have been downloaded or evaluated in
this repository.** This document records the exact commands to run *after* you personally accept
the data terms. Nothing here authorizes acceptance on your behalf or downloads data now.

## Permitted sources and constraints

- Source: PhysioNet Challenge 2026 training data via Kaggle (`physionetchallenge2026data`).
- Authorized training sources only: `S0001`, `I0002`, `I0006`. Hidden-test sources are rejected.
- Small set ~214 GiB (`totalBytes` 230,024,509,878), large set ~1.2 TiB. **Neither is downloaded.**
- Only `demographics.csv` (~129 KiB) is eligible for selective retrieval, under the existing
  500 MiB pre-download size guard in `scripts/download_metadata.py`.
- EDF files must be obtained by you, from permitted sources, under terms you accept yourself.
- Raw EDFs, patient-level CSVs, tokens and environment files stay in git-ignored `data/`.

## Prerequisites you must perform

```bash
python -m pip install -e ".[dev]"
pipx install kaggle                 # or: uv tool install kaggle (Python 3.11+)
export KAGGLE_API_TOKEN='...'       # or write the token to ~/.kaggle/access_token
# Personally accept the BDSP terms at the Kaggle dataset page before proceeding.
```

## Reproduction commands (run only after the prerequisites)

```bash
python scripts/inventory_dataset.py                       # read-only access audit
python scripts/download_metadata.py \
  --confirm-bsdp-terms-accepted --destination data \
  --output-report outputs/metadata_download_report.json
python scripts/audit_metadata.py data/demographics.csv \
  --output outputs/metadata_audit.json
python scripts/audit_channels.py RECORDING.edf --mapping channel_mapping.csv
python scripts/extract_features.py physio/S0001/sub-....edf \
  --mapping channel_mapping.csv --output data/features.csv
python scripts/build_feature_cohort.py data/demographics.csv data/features.csv \
  --output data/feature_cohort.csv --report outputs/join_report.json
python scripts/train_baselines.py data/feature_cohort.csv --model combined_lr
python scripts/evaluate_models.py data/feature_cohort.csv \
  --metadata-csv data/demographics.csv \
  --output outputs/evaluation_results.json \
  --comparison outputs/comparison_table.csv \
  --manifest-output outputs/experiment_manifest.json
```

`evaluate_models.py` uses leave-one-site-out folds, composite patient identities, fold-local
imputation/scaling, the age-conditioned I' metric, explicit unscoreable-fold reporting and a
patient-cluster bootstrap. It writes aggregate results and a SHA-256 provenance manifest only.

## Current blockers

1. No Kaggle credentials, access-token file, OAuth session or legacy key pair is configured here.
2. BDSP terms must be accepted by you in an authenticated account; we never accept them.
3. Downloading the full 214 GiB / 1.2 TiB datasets is explicitly out of scope without approval.
4. Therefore **no real-data metric is claimed**. All committed results come from synthetic
   fixtures and the deterministic public benchmark.
