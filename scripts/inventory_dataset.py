#!/usr/bin/env python
"""Report dataset-access status and known file inventory without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeppsg_ci.data_access import access_audit_to_dict, audit_access  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    report = access_audit_to_dict(audit_access())
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote dataset-access audit to {target}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
