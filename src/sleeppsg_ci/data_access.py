"""Dataset access safeguards and the 2026 training-data access audit.

No credentials, tokens or secrets are ever printed or stored. The audit records
only whether credentials are *present* (boolean), never their values.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Verified from the public Kaggle API and the official challenge page on 2026-11-23.
SMALL_DATASET = "physionet/physionetchallenge2026data"
LARGE_DATASET = "physionet/physionetchallenge2026datalargeversion"
SUPPLEMENTARY_DATASET = "physionet/physionet2026datasupplementary"

SMALL_DATASET_BYTES = 230_024_509_878  # ~214 GiB (Kaggle API totalBytes)
LARGE_DATASET_BYTES_APPROX = 1_318_230_425_600  # ~1.2 TiB (challenge page)
MAX_DOWNLOAD_BYTES = 1_000_000_000  # 1 GiB hard cap without explicit approval
MAX_METADATA_BYTES = 500_000_000  # 500 MiB selective metadata limit

#: Metadata/aggregate files that are independently retrievable and small.
KNOWN_METADATA_FILES: frozenset[str] = frozenset({"demographics.csv", "ICD_codes_CI.csv"})
RAW_SIGNATURE_EXTENSIONS: frozenset[str] = frozenset({".edf", ".bdf", ".vhdr", ".eeg", ".fif"})

#: Training sources documented by the official challenge page.
TRAINING_SOURCES = ("S0001", "I0002", "I0006")
HIDDEN_SOURCES = ("I0004", "I0007")  # validation / test; labels never accessed


class DataAccessError(RuntimeError):
    """Raised when a data-access request violates governance rules."""


@dataclass(frozen=True)
class AccessAudit:
    """Outcome of the non-destructive dataset-access audit."""

    kaggle_cli_available: bool
    kaggle_credentials_present: bool
    public_metadata_api_reachable: bool
    selective_file_retrieval_supported: bool
    small_dataset_bytes: int
    large_dataset_bytes_approx: int
    download_cap_bytes: int
    permitted_metadata_files: tuple[str, ...]
    raw_signatures_blocked: bool
    blocker: str | None
    notes: tuple[str, ...]


def kaggle_credentials_present() -> bool:
    """Return True only if both Kaggle environment variables are set."""
    return bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))


def kaggle_cli_available() -> bool:
    """Return True if the Kaggle CLI is installed on PATH."""
    return shutil.which("kaggle") is not None


def audit_access(public_metadata_reachable: bool = True) -> AccessAudit:
    """Summarize dataset-access status without downloading or printing secrets.

    ``public_metadata_reachable`` reflects whether the unauthenticated Kaggle
    metadata endpoint responded during the inspection (recorded off-line here).
    """
    credentials = kaggle_credentials_present()
    cli = kaggle_cli_available()
    if not credentials:
        blocker = (
            "Kaggle credentials are not configured (KAGGLE_USERNAME/KAGGLE_KEY absent) "
            "and the kaggle CLI is not installed. Selective retrieval of "
            "demographics.csv therefore cannot be authenticated; no dataset content "
            "was downloaded. Supply credentials and accept the BDSP data terms to "
            "retrieve the small metadata file."
        )
    elif not cli:
        blocker = (
            "Kaggle credentials are present but the kaggle CLI is unavailable; install "
            "kaggle to use selective single-file retrieval."
        )
    else:
        blocker = None

    return AccessAudit(
        kaggle_cli_available=cli,
        kaggle_credentials_present=credentials,
        public_metadata_api_reachable=public_metadata_reachable,
        selective_file_retrieval_supported=bool(credentials and cli),
        small_dataset_bytes=SMALL_DATASET_BYTES,
        large_dataset_bytes_approx=LARGE_DATASET_BYTES_APPROX,
        download_cap_bytes=MAX_DOWNLOAD_BYTES,
        permitted_metadata_files=tuple(sorted(KNOWN_METADATA_FILES)),
        raw_signatures_blocked=True,
        blocker=blocker,
        notes=(
            "Only the publicly labelled training partition may be used.",
            "Hidden validation/test labels are never accessed or inferred.",
            "Raw EDF recordings are never downloaded in bulk.",
            "The small training set is ~214 GiB; the large set is ~1.2 TiB.",
        ),
    )


def validate_download_request(files: Sequence[str], total_bytes: int) -> None:
    """Enforce governance limits on any prospective metadata download."""
    if total_bytes > MAX_DOWNLOAD_BYTES:
        raise DataAccessError(
            f"Requested {total_bytes} bytes exceeds the {MAX_DOWNLOAD_BYTES} byte cap."
        )
    for name in files:
        suffix = Path(name).suffix.casefold()
        if suffix in RAW_SIGNATURE_EXTENSIONS:
            raise DataAccessError(f"Raw physiological recordings may not be downloaded: {name}")
        if Path(name).name not in KNOWN_METADATA_FILES:
            raise DataAccessError(f"Only known small metadata files may be fetched: {name}")
    if len(files) > len(KNOWN_METADATA_FILES):
        raise DataAccessError("Too many files requested for a metadata-only retrieval.")


def download_metadata_files(
    destination: str | Path,
    files: Sequence[str] = ("demographics.csv",),
) -> list[Path]:
    """Selectively fetch small metadata files using the Kaggle CLI.

    This function intentionally fails closed when credentials or the CLI are
    missing. It never logs the environment or command secrets.
    """
    validate_download_request(files, total_bytes=MAX_METADATA_BYTES)
    audit = audit_access()
    if audit.blocker is not None:
        raise DataAccessError(audit.blocker)

    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for name in files:
        command = [
            "kaggle", "datasets", "download", "-d", SMALL_DATASET,
            "-f", name, "-p", str(target), "--unzip",
        ]
        result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise DataAccessError(
                f"Kaggle refused to download {name} (exit {result.returncode}). "
                "Confirm that the BDSP terms were accepted and credentials are valid."
            )
        path = target / Path(name).name
        if not path.is_file():
            raise DataAccessError(f"Kaggle reported success but {name} is absent.")
        if path.stat().st_size > MAX_METADATA_BYTES:
            raise DataAccessError(f"{name} exceeds the 500 MiB selective metadata limit.")
        downloaded.append(path)
    return downloaded


def access_audit_to_dict(audit: AccessAudit) -> dict:
    """Convert an access audit to a deterministic, secret-free dictionary."""
    return {
        "kaggle_cli_available": audit.kaggle_cli_available,
        "kaggle_credentials_present": audit.kaggle_credentials_present,
        "public_metadata_api_reachable": audit.public_metadata_api_reachable,
        "selective_file_retrieval_supported": audit.selective_file_retrieval_supported,
        "small_dataset_bytes": audit.small_dataset_bytes,
        "large_dataset_bytes_approx": audit.large_dataset_bytes_approx,
        "download_cap_bytes": audit.download_cap_bytes,
        "permitted_metadata_files": list(audit.permitted_metadata_files),
        "raw_signatures_blocked": audit.raw_signatures_blocked,
        "blocker": audit.blocker,
        "notes": list(audit.notes),
    }
