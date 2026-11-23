# Official specification audit

All items below were inspected on **2026-11-23**. This project is independent of
the official challenge; these notes document the public specification only.

## Official sources

| Item | URL |
|---|---|
| Challenge page | https://moody-challenge.physionet.org/2026/ |
| Official results | https://moody-challenge.physionet.org/2026/results/ |
| Python example + scorer | https://github.com/physionetchallenges/python-example-2026 |
| Small training set | https://www.kaggle.com/datasets/physionet/physionetchallenge2026data |
| Large training set | https://www.kaggle.com/datasets/physionet/physionetchallenge2026datalargeversion |
| Supplementary set | https://www.kaggle.com/datasets/physionet/physionet2026datasupplementary |

**Exact example-code commit:** `c0bcd78ddb892290b9d218be9be19660f4d8bdf3`
("Update README", Matthew Reyna, 2026-07-10). Local mirror inspected at
`../project-4-physionet-2026-psg-cognitive-impairment`.

**Code license:** BSD 3-Clause License, Copyright (c) 2026 George B. Moody
PhysioNet Challenges. The adapted scorer in `src/sleeppsg_ci/metrics.py` retains
the copyright and redistribution notice.

**Data license:** BDSP Credentialed Health Data License v1.0.0, Copyright (c)
2026 Brain Data Science Platform. Terms acceptance and credentials are required.

## Official scoring functions

From `evaluate_model.py` at the commit above (signatures):

```python
compute_prevalence(ages, prevalence_labels, prevalence_ages, gap=0)
compute_reward(labels, predictions, ages, age_to_prevalence)
compute_auroc_age(labels, predictions, ages, gap=0)
compute_auroc_weighted(labels, predictions, ages, gap=0)
compute_auroc(labels, predictions)
compute_auprc(labels, predictions)
compute_confusion_matrix(labels, predictions)
evaluate_model(labels_files, predictions_files, prevalence_files)
```

The **primary** metric is `compute_auroc_age`, the probability that a positive
patient is ranked above a negative patient of approximately the same age. The
official evaluator uses `gap=2` years for `compute_auroc_age`,
`compute_auroc_weighted` and `compute_prevalence`. `evaluate_model` filters labels
to `{0, 1}`, aligns rows by `SiteID_BDSPPatientID`, and uses the training set to
compute prevalence. Our adaptation returns `NaN` (not zero) when no comparable
positive/negative pair exists, so a single-class fold is explicitly not scoreable.

## Dataset schema (demographics.csv)

`SiteID`, `BDSPPatientID`, `CreationTime`, `BidsFolder`, `SessionID`, `Age`, `Sex`,
`Race`, `Ethnicity`, `BMI`, `Time_to_Event`, `Cognitive_Impairment`,
`Last_Known_Visit_Date`, `Time_to_Last_Visit`.

Official prediction columns: `Cognitive_Impairment` (binary),
`Cognitive_Impairment_Probability` (real-valued).

Records link to signals via `BidsFolder` + `SessionID`; e.g. `BidsFolder =
sub-S0001111197789`, `SessionID = 2` maps to
`physiological_data/S0001/sub-S0001111197789-ses2.edf`.

## Label rules

- **Positive:** ≥2 qualifying CI diagnoses; first ≥1 and ≤6 years after PSG, with
  ≥1 week between diagnoses.
- **Negative:** no CI diagnosis and ≥6 years of encounters after PSG.
- **Other (excluded):** first diagnosis <1 or >6 years after PSG, only one CI
  diagnosis, or <6 years of follow-up without a diagnosis.

The supplied `Cognitive_Impairment` label is used verbatim; no approximate
reconstruction is attempted. Ages >89 are reported as 90 in the source data.

## Partitions

| Partition | Sources | Status |
|---|---|---|
| Training (small/large) | S0001, I0002, I0006 | Public, labelled |
| Validation | I0004 | Hidden, unlabelled to participants |
| Test | I0007 | Hidden, unlabelled to participants |
| Supplementary | I0004, I0007 (10 each) | Unlabelled examples only |

Small training counts (physiological_data): S0001 857, I0002 54, I0006 192.
Large: S0001 5139, I0002 319, I0006 1142.

## Available modalities

EEG (unipolar and bipolar, as recorded), EOG, EMG (chin and limb), ECG, chest and
abdominal respiratory effort, thermal/pressure airflow, SpO₂. Channels, sampling
rates, references and units vary by site and recording. CAISR algorithmic
annotations (30-second sleep stages, 0.5-second arousals, 1-second respiratory and
limb events) and human annotations are provided in EDF form. No channel modality
is inferred from an unverified abbreviation in this project.

## File organization

Each partition contains `physiological_data/`, `algorithmic_annotations/`,
`human_annotations/` (training only), `demographics.csv` and `ICD_codes_CI.csv`.

## Dataset-access audit (2026-11-23)

- Kaggle public metadata API reachable without credentials; dataset `totalBytes`
  for the small set = 230,024,509,878 (~214 GiB); large set ~1.2 TiB.
- `ICD_codes_CI.csv` = 88,731 bytes; annotations are small EDFs. `demographics.csv`
  is a standalone partition-root file and the only metadata file needed locally.
- Kaggle supports selective single-file retrieval (`kaggle datasets download -f`),
  but it requires credentials and terms acceptance.
- Local blockers: no `KAGGLE_USERNAME`/`KAGGLE_KEY`, no `kaggle` CLI installed.
- Therefore **no dataset content was downloaded**, and no signal experiments were
  run. This is reported openly rather than replaced with fabricated results.
