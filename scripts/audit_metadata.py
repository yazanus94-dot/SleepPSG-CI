#!/usr/bin/env python
"""Audit a labelled training demographics.csv and write an aggregate report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeppsg_ci.cohort import CohortError, audit_cohort, load_cohort_csv, write_audit_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the labelled training demographics.csv")
    parser.add_argument("--output", default="outputs/metadata_audit.json", help="Aggregate JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = load_cohort_csv(args.csv_path)
        audit = audit_cohort(frame)
    except (FileNotFoundError, CohortError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    target = write_audit_report(audit, args.output)
    print(f"Wrote aggregate metadata audit to {target}")
    print(f"Eligible records: {audit.eligibility['eligible_records']} / {audit.total_records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
