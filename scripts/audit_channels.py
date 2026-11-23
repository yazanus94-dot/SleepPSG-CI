#!/usr/bin/env python
"""Audit EDF channel availability and signal quality (aggregate output only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeppsg_ci.channel_audit import (  # noqa: E402
    audit_recording,
    channel_availability_table,
    load_channel_mapping,
    sampling_rate_inventory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf_files", nargs="+", help="EDF recordings to inspect")
    parser.add_argument("--mapping", default=None, help="Verified channel mapping CSV")
    parser.add_argument("--output", default="outputs/channel_audit.json", help="Aggregate JSON output")
    parser.add_argument("--header-only", action="store_true", help="Skip sample-level quality checks")
    args = parser.parse_args(argv)

    rules = load_channel_mapping(args.mapping) if args.mapping else ()
    audits = [audit_recording(path, rules, inspect_samples=not args.header_only) for path in args.edf_files]
    report = {
        "n_recordings": len(audits),
        "recordings": [audit.to_dict() for audit in audits],
        "channel_availability": channel_availability_table(audits).to_dict(orient="index"),
        "sampling_rates": sampling_rate_inventory(audits).to_dict(orient="records"),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"Wrote aggregate channel audit for {len(audits)} recording(s) to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
