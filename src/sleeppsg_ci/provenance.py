"""Reproducibility manifests: package versions, seeds and file checksums."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pandas
import scipy
import sklearn

from . import __version__

KEY_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "pytest", "ruff")


def package_versions() -> dict[str, str]:
    """Return installed versions of the key scientific packages."""
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sleeppsg_ci": __version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for name in KEY_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return dict(sorted(versions.items()))


def sha256_file(path: str | Path) -> str:
    """Compute a SHA-256 checksum of a file without loading it wholly."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_manifest(
    command: str,
    seed: int = 42,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
) -> dict:
    """Create a deterministic manifest (timestamps excluded from checksums)."""
    return {
        "project": "SleepPSG-CI",
        "project_version": __version__,
        "command": command,
        "random_seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": package_versions(),
        "input_checksums": dict(sorted((inputs or {}).items())),
        "output_checksums": dict(sorted((outputs or {}).items())),
        "official_example_commit": "c0bcd78ddb892290b9d218be9be19660f4d8bdf3",
        "independence_statement": (
            "Independent research inspired by the 2026 PhysioNet Challenge; not an official entry."
        ),
    }
