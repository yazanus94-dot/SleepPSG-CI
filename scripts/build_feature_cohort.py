#!/usr/bin/env python
"""Join extracted PSG features to labelled demographics for local training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeppsg_ci.cohort import load_cohort_csv  # noqa: E402
from sleeppsg_ci.cohort_join import JoinError, join_feature_cohort  # noqa: E402
from sleeppsg_ci.provenance import run_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demographics_csv", help="Local labelled training demographics.csv")
    parser.add_argument("features_csv", help="Feature table produced by extract_features.py")
    parser.add_argument("--output", default="data/feature_cohort.csv", help="Local joined cohort (git-ignored)")
    parser.add_argument("--report", default="outputs/join_report.json", help="Aggregate join report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        demographics = load_cohort_csv(args.demographics_csv)
        features = pd.read_csv(args.features_csv)
        cohort, report = join_feature_cohort(demographics, features)
    except (FileNotFoundError, JoinError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["manifest"] = run_manifest("build_feature_cohort")
    payload["storage_note"] = "The joined cohort contains identifiers and must remain git-ignored."
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {report.output_rows} joined row(s) to {output}")
    print(f"Aggregate join report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
