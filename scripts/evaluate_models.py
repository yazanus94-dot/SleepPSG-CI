#!/usr/bin/env python
"""Run leave-one-site-out evaluation and write aggregate results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeppsg_ci.evaluation import comparison_table, evaluate_site_held_out  # noqa: E402
from sleeppsg_ci.provenance import run_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_csv", help="CSV containing SiteID, patient ID, demographics, features and labels")
    parser.add_argument("--output", default="outputs/evaluation_results.json")
    parser.add_argument("--comparison", default="outputs/comparison_table.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.feature_csv)
    results = evaluate_site_held_out(frame, seed=args.seed)
    results["manifest"] = run_manifest("evaluate_models", seed=args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    comparison = Path(args.comparison)
    comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison_table(results).to_csv(comparison, index=False)
    print(f"Wrote site-held-out results to {output} and comparison table to {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
