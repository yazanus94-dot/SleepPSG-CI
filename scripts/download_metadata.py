#!/usr/bin/env python
"""Download the labelled training demographics.csv with explicit user consent.

This script never accepts legal terms on anyone's behalf. The user must already
have accepted the BDSP Credentialed Health Data License v1.0.0 on their own
Kaggle account and must assert that with ``--confirm-bsdp-terms-accepted``. Only
``demographics.csv`` is fetched; no EDF or other dataset content is downloaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sleeppsg_ci.data_access import (  # noqa: E402
    DOWNLOAD_ALLOWLIST,
    MAX_METADATA_BYTES,
    DataAccessError,
    audit_access,
    download_metadata_files,
    fetch_remote_file_size,
    validate_remote_size,
)
from sleeppsg_ci.provenance import run_manifest, sha256_file  # noqa: E402

FILE_NAME = "demographics.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-bsdp-terms-accepted",
        action="store_true",
        help=(
            "Assert that YOU have already accepted the BDSP data terms on your "
            "Kaggle account. This script never accepts them for you."
        ),
    )
    parser.add_argument("--destination", default="data", help="Git-ignored data directory")
    parser.add_argument("--output-report", default="outputs/metadata_download_report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_bsdp_terms_accepted:
        print(
            "ERROR: explicit confirmation is required. Accept the BDSP terms "
            "yourself on https://www.kaggle.com/datasets/physionet/"
            "physionetchallenge2026data and rerun with "
            "--confirm-bsdp-terms-accepted. This script never accepts terms "
            "automatically.",
            file=sys.stderr,
        )
        return 2

    audit = audit_access()
    if audit.blocker is not None:
        print(f"ERROR: {audit.blocker}", file=sys.stderr)
        return 2

    try:
        remote_size = fetch_remote_file_size(FILE_NAME)
        validate_remote_size(remote_size)
        paths = download_metadata_files(
            args.destination,
            files=(FILE_NAME,),
            confirm_bsdp_terms=True,
        )
    except DataAccessError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    path = paths[0]
    report = {
        "downloaded_file": path.name,
        "allowlist": sorted(DOWNLOAD_ALLOWLIST),
        "remote_size_bytes": remote_size,
        "local_size_bytes": path.stat().st_size,
        "size_limit_bytes": MAX_METADATA_BYTES,
        "sha256": sha256_file(path),
        "credential_source": audit.credential_source,
        "manifest": run_manifest("download_metadata", inputs={"demographics": sha256_file(path)}),
        "storage_note": "Downloaded data stays in a git-ignored directory; do not commit it.",
    }
    output = Path(args.output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Downloaded {path.name} ({remote_size} bytes) to {path}")
    print(f"Aggregate download report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
