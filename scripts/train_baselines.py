#!/usr/bin/env python
"""Train one baseline on a prepared, labelled feature table (smoke/repro run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeppsg_ci.evaluation import prepare_labelled  # noqa: E402
from sleeppsg_ci.modeling import (  # noqa: E402
    build_baselines,
    feature_kind_for,
    fit_predict,
    model_config_hash,
    select_feature_matrix,
)
from sleeppsg_ci.provenance import run_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_csv", help="CSV containing demographics, features and labels")
    parser.add_argument("--model", default="combined_lr", help="Baseline name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/training_report.json")
    args = parser.parse_args(argv)

    frame = prepare_labelled(pd.read_csv(args.feature_csv))
    model = build_baselines(args.seed)[args.model]
    x = select_feature_matrix(frame, feature_kind_for(args.model))
    y = frame["Cognitive_Impairment"].to_numpy(dtype=int)
    fit_predict(model, x, y, x)
    report = {
        "model": args.model,
        "feature_kind": feature_kind_for(args.model),
        "n_records": int(len(frame)),
        "model_config_sha256": model_config_hash(model),
        "in_sample_note": "In-sample smoke run only; use evaluate_models.py for site-held-out results.",
        "manifest": run_manifest(f"train_baselines --model {args.model}", seed=args.seed),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"Trained {args.model}; report written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
