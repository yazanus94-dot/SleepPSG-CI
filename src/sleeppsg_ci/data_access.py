"""Dataset access safeguards and the 2026 training-data access audit.

No credentials, tokens or secrets are ever printed, logged or stored. The audit
records only *whether* credentials are present and *where* they were found
(environment, ``~/.kaggle/kaggle.json`` or ``KAGGLE_CONFIG_DIR/kaggle.json``),
never their values.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Verified from the public Kaggle API and the official challenge page on 2026-11-23.
SMALL_DATASET = "physionet/physionetchallenge2026data"
LARGE_DATASET = "physionet/physionetchallenge2026datalargeversion"
SUPPLEMENTARY_DATASET = "physionet/physionet2026datasupplementary"
KAGGLE_FILE_LIST_API = "https://www.kaggle.com/api/v1/datasets/list"

SMALL_DATASET_BYTES = 230_024_509_878  # ~214 GiB (Kaggle API totalBytes)
LARGE_DATASET_BYTES_APPROX = 1_318_230_425_600  # ~1.2 TiB (challenge page)
MAX_DOWNLOAD_BYTES = 1_000_000_000  # 1 GiB general cap
MAX_METADATA_BYTES = 500_000_000  # 500 MiB selective metadata limit

#: Files that are independently retrievable and small.
KNOWN_METADATA_FILES: frozenset[str] = frozenset({"demographics.csv", "ICD_codes_CI.csv"})
#: This workflow downloads exactly one file, and never any raw signal data.
DOWNLOAD_ALLOWLIST: frozenset[str] = frozenset({"demographics.csv"})
RAW_SIGNATURE_EXTENSIONS: frozenset[str] = frozenset({".edf", ".bdf", ".vhdr", ".eeg", ".fif"})

#: Training sources documented by the official challenge page.
TRAINING_SOURCES = ("S0001", "I0002", "I0006")
HIDDEN_SOURCES = ("I0004", "I0007")  # validation / test; labels never accessed

CREDENTIAL_SOURCE_NONE = "none"
CREDENTIAL_SOURCE_ENV = "environment"
CREDENTIAL_SOURCE_HOME = "home_kaggle_json"
CREDENTIAL_SOURCE_CONFIG_DIR = "kaggle_config_dir"


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
    credential_source: str = CREDENTIAL_SOURCE_NONE


def _valid_kaggle_json(path: Path) -> bool:
    """Return True only for a readable kaggle.json with both fields non-blank.

    The parsed values are deliberately discarded: no credential value is ever
    returned, logged or included in an exception message.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    username = payload.get("username") or payload.get("USERNAME")
    key = payload.get("key") or payload.get("KEY")
    return bool(
        isinstance(username, str) and username.strip()
        and isinstance(key, str) and key.strip()
    )


def credential_source(
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> str:
    """Identify where usable Kaggle credentials are configured (never their values)."""
    env = os.environ if environ is None else environ
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return CREDENTIAL_SOURCE_ENV

    custom_dir = env.get("KAGGLE_CONFIG_DIR")
    if custom_dir and _valid_kaggle_json(Path(custom_dir) / "kaggle.json"):
        return CREDENTIAL_SOURCE_CONFIG_DIR

    home_dir = Path.home() if home is None else Path(home)
    if _valid_kaggle_json(home_dir / ".kaggle" / "kaggle.json"):
        return CREDENTIAL_SOURCE_HOME
    return CREDENTIAL_SOURCE_NONE


def kaggle_credentials_present(
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> bool:
    """Return True if credentials exist in env, kaggle.json or KAGGLE_CONFIG_DIR."""
    return credential_source(environ=environ, home=home) != CREDENTIAL_SOURCE_NONE


def kaggle_cli_available() -> bool:
    """Return True if the Kaggle CLI is installed on PATH."""
    return shutil.which("kaggle") is not None


def audit_access(
    public_metadata_reachable: bool = True,
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> AccessAudit:
    """Summarize dataset-access status without downloading or exposing secrets."""
    source = credential_source(environ=environ, home=home)
    credentials = source != CREDENTIAL_SOURCE_NONE
    cli = kaggle_cli_available()
    if not credentials:
        blocker = (
            "Kaggle credentials are not configured in environment variables, "
            "~/.kaggle/kaggle.json or KAGGLE_CONFIG_DIR/kaggle.json. Selective "
            "retrieval of demographics.csv cannot be authenticated; no dataset "
            "content was downloaded. Configure credentials yourself and accept "
            "the BDSP data terms on your own Kaggle account before retrying."
        )
    elif not cli:
        blocker = (
            "Kaggle credentials are present but the kaggle CLI is unavailable; "
            "install kaggle to use selective single-file retrieval."
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
        download_cap_bytes=MAX_METADATA_BYTES,
        permitted_metadata_files=tuple(sorted(KNOWN_METADATA_FILES)),
        raw_signatures_blocked=True,
        blocker=blocker,
        notes=(
            "Only the publicly labelled training partition may be used.",
            "Hidden validation/test labels are never accessed or inferred.",
            "Raw EDF recordings are never downloaded in bulk.",
            "The small training set is ~214 GiB; the large set is ~1.2 TiB.",
            "This workflow downloads only demographics.csv, never raw signals.",
        ),
        credential_source=source,
    )


def validate_download_request(files: Sequence[str], total_bytes: int) -> None:
    """Enforce the allowlist and 500 MiB cap for this metadata-only workflow."""
    if total_bytes > MAX_METADATA_BYTES:
        raise DataAccessError(
            f"Requested {total_bytes} bytes exceeds the {MAX_METADATA_BYTES} byte cap."
        )
    if len(files) != 1:
        raise DataAccessError("This workflow downloads exactly one file: demographics.csv.")
    for name in files:
        if Path(name).suffix.casefold() in RAW_SIGNATURE_EXTENSIONS:
            raise DataAccessError(f"Raw physiological recordings may not be downloaded: {name}")
        if Path(name).name not in DOWNLOAD_ALLOWLIST:
            raise DataAccessError(
                f"Only demographics.csv may be downloaded in this workflow: {name}"
            )


def validate_remote_size(remote_bytes: int) -> None:
    """Refuse a download when the remote file is missing, empty or too large."""
    if remote_bytes <= 0:
        raise DataAccessError("Remote demographics.csv has no known size; refusing to download.")
    if remote_bytes > MAX_METADATA_BYTES:
        raise DataAccessError(
            f"Remote demographics.csv is {remote_bytes} bytes, above the "
            f"{MAX_METADATA_BYTES} byte limit; refusing to download."
        )


def fetch_remote_file_size(
    filename: str = "demographics.csv",
    *,
    dataset: str = SMALL_DATASET,
    opener=None,
    max_pages: int = 220,
    timeout: int = 30,
) -> int:
    """Read a file's exact size from the public paginated Kaggle file inventory.

    This is a metadata-only HTTP read. ``opener`` is injectable so tests can use
    mocked responses; no credentials are sent or printed.
    """
    if Path(filename).name not in DOWNLOAD_ALLOWLIST:
        raise DataAccessError(f"Only demographics.csv may be inspected: {filename}")
    request = opener or urllib.request.urlopen
    token: str | None = None

    for _ in range(max_pages):
        url = f"{KAGGLE_FILE_LIST_API}/{dataset}"
        if token:
            url += "?pageToken=" + urllib.parse.quote(token, safe="")
        response = request(url, timeout=timeout)
        try:
            raw = response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        payload = json.loads(raw.decode("utf-8"))
        for item in payload.get("datasetFiles", []):
            if item.get("name") == filename:
                return int(item["totalBytes"])
        token = payload.get("nextPageToken")
        if not token:
            break
    raise DataAccessError(
        f"{filename} was not found in the public file inventory; no data was downloaded."
    )


def download_metadata_files(
    destination: str | Path,
    files: Sequence[str] = ("demographics.csv",),
    *,
    confirm_bsdp_terms: bool = False,
    opener=None,
    runner=None,
) -> list[Path]:
    """Fetch demographics.csv only, after explicit user terms confirmation.

    Order of checks: allowlist and size cap → terms confirmation → credentials →
    exact remote size → download → extracted size and integrity check. Legal
    terms are never accepted programmatically; ``confirm_bsdp_terms`` is a user
    assertion supplied by the caller (see ``scripts/download_metadata.py``).
    """
    validate_download_request(files, total_bytes=MAX_METADATA_BYTES)
    if not confirm_bsdp_terms:
        raise DataAccessError(
            "The BDSP terms must be explicitly confirmed by the user. This "
            "function never accepts legal terms automatically."
        )

    audit = audit_access()
    if audit.blocker is not None:
        raise DataAccessError(audit.blocker)

    name = files[0]
    remote_size = fetch_remote_file_size(name, opener=opener)
    validate_remote_size(remote_size)

    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle", "datasets", "download", "-d", SMALL_DATASET,
        "-f", name, "-p", str(target), "--unzip",
    ]
    invoke = runner or subprocess.run
    result = invoke(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # Only the exit code is reported; process output is never logged.
        raise DataAccessError(
            f"Kaggle refused to download {name} (exit {result.returncode}). "
            "Confirm that you accepted the BDSP terms and that credentials are valid."
        )

    path = target / Path(name).name
    if not path.is_file():
        raise DataAccessError(f"Kaggle reported success but {name} is absent.")
    local_size = path.stat().st_size
    if local_size > MAX_METADATA_BYTES:
        raise DataAccessError(f"{name} exceeds the 500 MiB selective metadata limit.")
    if local_size != remote_size:
        raise DataAccessError(
            f"Downloaded {name} size mismatch (remote {remote_size}, local {local_size}); "
            "refusing to use the file."
        )
    return [path]


def access_audit_to_dict(audit: AccessAudit) -> dict:
    """Convert an access audit to a deterministic, secret-free dictionary."""
    return {
        "kaggle_cli_available": audit.kaggle_cli_available,
        "kaggle_credentials_present": audit.kaggle_credentials_present,
        "credential_source": audit.credential_source,
        "public_metadata_api_reachable": audit.public_metadata_api_reachable,
        "selective_file_retrieval_supported": audit.selective_file_retrieval_supported,
        "small_dataset_bytes": audit.small_dataset_bytes,
        "large_dataset_bytes_approx": audit.large_dataset_bytes_approx,
        "download_cap_bytes": audit.download_cap_bytes,
        "permitted_metadata_files": list(audit.permitted_metadata_files),
        "download_allowlist": sorted(DOWNLOAD_ALLOWLIST),
        "raw_signatures_blocked": audit.raw_signatures_blocked,
        "blocker": audit.blocker,
        "notes": list(audit.notes),
    }
