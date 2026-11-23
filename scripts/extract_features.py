#!/usr/bin/env python
"""Extract full-recording physiological features into an aggregate feature table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeppsg_ci.channel_audit import load_channel_mapping  # noqa: E402
from sleeppsg_ci.cohort import parse_recording_identity  # noqa: E402
from sleeppsg_ci.feature_table import (  # noqa: E402
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_EPOCH_SECONDS,
    extract_from_edf,
    features_to_dataframe,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf_files", nargs="+", help="EDF recordings to process (full recording)")
    parser.add_argument("--mapping", default=None, help="Verified channel mapping CSV")
    parser.add_argument("--site", default="*", help="Site identifier for site-specific mappings")
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
                        help="Partial-read chunk size (I/O granularity only)")
    parser.add_argument("--epoch-seconds", type=float, default=DEFAULT_EPOCH_SECONDS,
                        help="Fixed-duration epoch used for per-epoch features")
    parser.add_argument("--output", default="outputs/features.csv", help="Feature table output path")
    args = parser.parse_args(argv)

    rules = load_channel_mapping(args.mapping) if args.mapping else ()
    rows = []
    for path in args.edf_files:
        features = extract_from_edf(
            path,
            rules=rules,
            site=args.site,
            chunk_seconds=args.chunk_seconds,
            epoch_seconds=args.epoch_seconds,
        )
        features["recording"] = Path(path).stem
        features.update(parse_recording_identity(path))
        rows.append(features)
    table = features_to_dataframe(rows)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    print(f"Wrote {len(table)} full-recording feature row(s) to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
