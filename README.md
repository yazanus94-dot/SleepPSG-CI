# SleepPSG-CI

**SleepPSG-CI: Multimodal Polysomnography-Based Cognitive Impairment Risk Prediction**

> **Independence statement.** This is an independent retrospective research and
> benchmarking project inspired by the George B. Moody PhysioNet Challenge 2026.
> It is **not** an official competition entry, no registration or participation is
> claimed, and no team name, leaderboard placement, percentile, or challenge score
> is claimed. Hidden validation/test labels were never accessed or inferred. The
> repository is research software only — not a diagnostic system, medical device
> or clinical decision-support tool.

---

## 1. Project overview

SleepPSG-CI is an installable Python package and benchmark for studying whether
overnight polysomnography (PSG) signals and baseline demographics can predict the
challenge-defined future cognitive-impairment outcome. The emphasis is on scientific
honesty, leakage-resistant validation, deterministic preprocessing and a
self-contained evaluator, rather than on leaderboard performance.

## 2. Research question

> Can a model trained on labelled PSG recordings from some contributing sources
> generalize to recordings from an unseen source when predicting future
> cognitive-impairment risk?

## 3. Official challenge inspiration

Inspired by the 2026 George B. Moody PhysioNet Challenge, *Screening for Cognitive
Impairment During Sleep Studies* (official phase concluded; results released
2026-09-28). The official task, data dictionary, example code and scorer were
inspected on 2026-11-23; exact details are in [`CHALLENGE_REFERENCES.md`](CHALLENGE_REFERENCES.md).

## 4. Independent work

No official submission exists. All metrics reported locally, if any, are
training-data experiments only and are not comparable with hidden-set results.

## 5. Dataset source and access

Public labelled training data: the small (~214 GiB, Kaggle `totalBytes` =
230,024,509,878) and large (~1.2 TiB) training sets from PhysioNet/Kaggle.
Training sources are `S0001`, `I0002`, `I0006`. Hidden sources `I0004` and `I0007`
are never used as labelled data.

```bash
python scripts/inventory_dataset.py
```

The access audit is metadata-only. Selective retrieval of `demographics.csv`
requires Kaggle credentials and acceptance of the BDSP data terms. **No dataset
content was downloaded in this repository.** See [`AUDIT_REPORT.md`](AUDIT_REPORT.md).

## 6. Data governance and privacy

Raw recordings, patient-level rows, identifiers, credentials and local
environment files are excluded by `.gitignore`. Public reports contain aggregate
statistics only. Indeterminate labels are excluded, never recoded as negatives.
See [`TASK.md`](TASK.md) for the input/output and forbidden-feature contract.

## 7. Modality overview

Supported, subject to verified channel mappings: EEG, EOG, EMG, ECG, respiratory
effort, airflow and SpO₂. Channel modality is never inferred from an abbreviation:
`channel_audit.ChannelRule` requires an explicitly verified, site-specific mapping.

## 8. Signal preprocessing

Modular primitives in `preprocessing.py`: linear detrending, baseline correction,
Butterworth band-pass filtering (SOS + zero-phase `sosfiltfilt`, documented order
and padding), IIR notch filtering, polyphase anti-aliased resampling and fixed
30-second epoch windowing. Quality checks (flatline, clipping, non-finite samples,
duration) live in `quality.py`; exclusions are reported with reasons.

## 9. Feature extraction

Per-modality extractors produce documented night-level features:

- **EEG**: delta/theta/alpha/sigma/beta absolute and relative power, spectral
  ratios, spectral entropy and 95% spectral edge frequency.
- **ECG**: R-peak detection, HR and HRV (SDNN, RMSSD, pNN50) with beat-quality
  flags; ECG-derived only, never PPG-derived PRV.
- **Respiration**: respiratory rate, amplitude CV and regularity.
- **EMG / EOG**: amplitude, activity/burst and spectral summaries.

Missing modalities are represented with explicit `missing_<modality>` indicators;
no physiological values are fabricated.

## 10. Machine-learning baselines

Six scikit-learn pipelines with fold-local imputation and scaling:
`dummy`, `demo_lr`, `signal_lr`, `signal_rf`, `combined_lr`, `combined_hgb`.
Configuration checksums and package versions are recorded for reproducibility.

## 11. Patient/site validation design

Leave-one-source-out evaluation holds out an entire site, keeps patients grouped,
tunes nothing on the held-out site, and asserts patient disjointness. Single-class
held-out folds are reported as **not scoreable** and are never silently dropped.

## 12. Evaluation metrics

**Primary:** the exact official 2026 age-conditioned AUROC (δ = 2 years), adapted
from the official scorer with its BSD-3-Clause notice preserved.
**Secondary:** age-weighted AUROC, ROC-AUC, PR-AUC, Brier score, calibration
slope/intercept, sensitivity, specificity, balanced accuracy, confusion matrix
and patient-level bootstrap confidence intervals.

## 13. Reproducibility

Fixed random seeds, deterministic reports (`sort_keys=True`), configuration
checksums and a provenance manifest (`provenance.py`) accompany every run.

## 14. Installation

```bash
python -m pip install -e ".[dev]"
```

Requires Python 3.10+. CI is tested on 3.10, 3.11 and 3.12.

## 15. Commands

```bash
python scripts/audit_metadata.py data/demographics.csv --output outputs/metadata_audit.json
python scripts/inventory_dataset.py
python scripts/audit_channels.py RECORDING.edf --mapping channel_mapping.csv
python scripts/extract_features.py RECORDING.edf --mapping channel_mapping.csv
python scripts/train_baselines.py outputs/features.csv --model combined_lr
python scripts/evaluate_models.py outputs/features.csv
```

If the dataset is unavailable, the metadata command exits with a clear error and
does **not** emit a fake successful report.

## 16. Tests

```bash
python -m pytest -q
ruff check .
```

CI runs without Kaggle credentials or PhysioNet downloads.

## 17. Results

**No real-data results are published.** With no dataset access, no EDF recordings
were processed and no models were trained on real patient data. Synthetic fixtures
are used only for software tests and are never reported as clinical performance.
See `benchmark/expected_results.json`.

## 18. Limitations

No real-data evaluation; only three labelled training sources exist; site-held-out
estimates are narrow and caveated; channel mappings require external verification;
the official metric is undefined for single-class folds.

## 19. References and license notices

Official example code © 2026 George B. Moody PhysioNet Challenges, BSD-3-Clause;
the adapted scorer notice is retained in `src/sleeppsg_ci/metrics.py` and in
[`NOTICE`](NOTICE). Dataset © BDSP, licensed under the BDSP Credentialed Health
Data License v1.0.0. Project code is MIT-licensed (see [`LICENSE`](LICENSE)).

Architecture: metadata audit → access audit → channel inventory → preprocessing →
per-modality features → baselines → site-held-out evaluation → benchmark evaluator.
