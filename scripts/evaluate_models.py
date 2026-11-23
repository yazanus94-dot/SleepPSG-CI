#!/usr/bin/env python
"""Run a reproducible site-held-out experiment on a prepared feature cohort.

This script is the real-data experiment entry point. It accepts only the public
labelled training sources, records dataset/checksum provenance, keeps patients
grouped, and writes aggregate results plus an explicit evaluation contract. It
does not access hidden validation/test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeppsg_ci.data_access import HIDDEN_SOURCES, TRAINING_SOURCES  # noqa: E402
from sleeppsg_ci.evaluation import comparison_table, evaluate_site_held_out  # noqa: E402
from sleeppsg_ci.provenance import run_manifest, sha256_file  # noqa: E402


def validate_sources(frame: pd.DataFrame) -> list[str]:
    """Return the sorted observed sources, rejecting hidden/unknown sources."""
    if "SiteID" not in frame.columns:
        raise ValueError("Feature cohort has no SiteID column; grouping is impossible.")
    observed = sorted({str(value).strip() for value in frame["SiteID"] if str(value).strip()})
    if not observed:
        raise ValueError("Feature cohort contains no SiteID values.")
    hidden = sorted(set(observed) & set(HIDDEN_SOURCES))
    if hidden:
        raise ValueError(
            "Hidden validation/test sources may not be used as labelled data: "
            + ", ".join(hidden)
        )
    unknown = sorted(set(observed) - set(TRAINING_SOURCES))
    if unknown:
        raise ValueError(
            "Sources outside the labelled training partition are not permitted: "
            + ", ".join(unknown)
        )
    return observed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_csv", help="Local feature cohort CSV (git-ignored)")
    parser.add_argument("--metadata-csv", default=None,
                        help="Optional demographics CSV whose checksum is recorded as provenance")
    parser.add_argument("--output", default="outputs/evaluation_results.json")
    parser.add_argument("--comparison", default="outputs/comparison_table.csv")
    parser.add_argument("--manifest-output", default="outputs/experiment_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    feature_path = Path(args.feature_csv)
    if not feature_path.is_file():
        print(f"ERROR: feature cohort not found: {feature_path}", file=sys.stderr)
        return 2

    frame = pd.read_csv(feature_path)
    try:
        observed_sources = validate_sources(frame)
        results = evaluate_site_held_out(frame, seed=args.seed)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    input_checksums = {"feature_cohort": sha256_file(feature_path)}
    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv)
        if not metadata_path.is_file():
            print(f"ERROR: metadata CSV not found: {metadata_path}", file=sys.stderr)
            return 2
        input_checksums["demographics"] = sha256_file(metadata_path)

    manifest = run_manifest("evaluate_models", seed=args.seed, inputs=input_checksums)
    results["manifest"] = manifest
    results["experiment"] = {
        "status": "prepared_and_executed_on_supplied_local_cohort",
        "allowed_sources": list(TRAINING_SOURCES),
        "observed_sources": observed_sources,
        "hidden_sources_present": False,
        "age_gap_years": 2,
        "contract": [
            "Only labelled training sources are accepted.",
            "Composite SiteID_BDSPPatientID patients stay grouped in every fold.",
            "Imputation and scaling are fitted inside the training sources only.",
            "Identifiers, labels, post-PSG variables and unit provenance are excluded from predictors.",
            "Held-out sources are evaluated once; single-class folds are reported as not scoreable.",
            "Age-conditioned AUROC uses the official delta of 2 years.",
            "Patient-cluster bootstrap CIs are computed on composite patient keys.",
        ],
        "notes": [
            "No hidden validation/test labels were accessed.",
            "No official challenge participation or score is claimed.",
            "Synthetic software tests are kept separate from this experiment.",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    comparison = Path(args.comparison)
    comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison_table(results).to_csv(comparison, index=False)
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"manifest": manifest, "experiment": results["experiment"]},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote site-held-out results to {output}, comparison to {comparison}, "
          f"manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
