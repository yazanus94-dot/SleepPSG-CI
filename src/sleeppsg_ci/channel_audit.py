"""EDF channel inventory with explicit, verifiable channel mappings.

Channel modality is *never* inferred from an unverified abbreviation. A channel
is labelled only when a user-supplied, site-specific mapping marks it as
verified; otherwise it is reported as ``UNVERIFIED``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .quality import signal_quality_report

VALID_MODALITIES = ("EEG", "EOG", "EMG", "ECG", "RESP", "AIRFLOW", "SPO2", "OTHER")


class ChannelMappingError(ValueError):
    """Raised for an invalid or unverified channel mapping."""


@dataclass(frozen=True)
class ChannelRule:
    """One site-specific mapping from a recorded channel label to a modality."""

    channel: str
    modality: str
    site: str = "*"
    verified: bool = False

    def __post_init__(self) -> None:
        if self.modality not in VALID_MODALITIES:
            raise ChannelMappingError(
                f"Unknown modality '{self.modality}'. Expected one of {VALID_MODALITIES}."
            )
        if not self.channel.strip():
            raise ChannelMappingError("Channel label must not be empty.")


def default_channel_rules() -> tuple[ChannelRule, ...]:
    """Return an empty-by-default mapping: nothing is assumed verified."""
    return ()


def load_channel_mapping(path: str | Path) -> tuple[ChannelRule, ...]:
    """Load a verified mapping CSV with columns channel, modality, site, verified.

    Only rows whose ``verified`` field is explicitly true become active rules.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Channel mapping file not found: {csv_path}")
    frame = pd.read_csv(csv_path, dtype=str).fillna("")
    required = {"channel", "modality"}
    if not required.issubset(frame.columns):
        raise ChannelMappingError(f"Mapping CSV must contain columns: {sorted(required)}")
    rules: list[ChannelRule] = []
    for row in frame.itertuples(index=False):
        raw_verified = str(getattr(row, "verified", "")).strip().casefold()
        verified = raw_verified in {"1", "true", "yes", "y"}
        if not verified:
            continue
        rules.append(
            ChannelRule(
                channel=str(row.channel).strip(),
                modality=str(row.modality).strip().upper(),
                site=str(getattr(row, "site", "*")).strip() or "*",
                verified=True,
            )
        )
    return tuple(rules)


def normalize_channel_name(name: str) -> str:
    """Normalize a channel label for matching without changing its meaning."""
    return "".join(char for char in str(name).strip().upper() if char.isalnum() or char in "-_+")


def resolve_modality(channel: str, rules: tuple[ChannelRule, ...], site: str = "*") -> str:
    """Resolve a channel to a verified modality, else ``UNVERIFIED``."""
    normalized = normalize_channel_name(channel)
    for rule in rules:
        if not rule.verified:
            continue
        if rule.site not in {"*", site}:
            continue
        if normalize_channel_name(rule.channel) == normalized:
            return rule.modality
    return "UNVERIFIED"


@dataclass
class EdfHeader:
    """Header-only EDF metadata (no signal samples retained)."""

    path: str
    channels: tuple[str, ...]
    sampling_rates: dict[str, float]
    units: dict[str, str]
    physical_limits: dict[str, tuple[float | None, float | None]]
    duration_seconds: float | None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "channels": list(self.channels),
            "sampling_rates": dict(self.sampling_rates),
            "units": dict(self.units),
            "physical_limits": {key: list(value) for key, value in self.physical_limits.items()},
            "duration_seconds": self.duration_seconds,
        }


def inspect_edf_header(path: str | Path) -> EdfHeader:
    """Inspect an EDF header using ``edfio`` (header-only, memory efficient)."""
    try:
        import edfio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("edfio is required for EDF inspection: pip install edfio") from exc

    edf_path = Path(path)
    if not edf_path.is_file():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    try:
        edf = edfio.read_edf(str(edf_path), lazy_load_data=True)
    except Exception as exc:
        raise RuntimeError(f"Could not read EDF header for {edf_path}: {exc}") from exc

    channels: list[str] = []
    rates: dict[str, float] = {}
    units: dict[str, str] = {}
    limits: dict[str, tuple[float | None, float | None]] = {}
    durations: list[float] = []
    for channel in edf.signals:
        label = str(channel.label).strip() or f"CH{len(channels) + 1}"
        channels.append(label)
        rate = float(channel.sampling_frequency)
        rates[label] = rate
        units[label] = str(getattr(channel, "physical_dimension", "")).strip()
        low = getattr(channel, "physical_min", None)
        high = getattr(channel, "physical_max", None)
        limits[label] = (float(low) if low is not None else None, float(high) if high is not None else None)
        n_samples = len(channel.data) if hasattr(channel, "data") else 0
        if rate > 0 and n_samples:
            durations.append(n_samples / rate)
    return EdfHeader(
        path=str(edf_path),
        channels=tuple(channels),
        sampling_rates=rates,
        units=units,
        physical_limits=limits,
        duration_seconds=max(durations) if durations else None,
    )


@dataclass
class RecordingAudit:
    """Per-recording channel and quality audit (aggregate values only)."""

    header: EdfHeader
    modality_by_channel: dict[str, str]
    available_modalities: list[str]
    missing_modalities: list[str]
    channel_quality: dict[str, dict]
    excluded_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            **self.header.to_dict(),
            "modality_by_channel": dict(self.modality_by_channel),
            "available_modalities": list(self.available_modalities),
            "missing_modalities": list(self.missing_modalities),
            "channel_quality": dict(self.channel_quality),
            "excluded_reasons": list(self.excluded_reasons),
        }


def audit_recording(
    path: str | Path,
    rules: tuple[ChannelRule, ...] = (),
    expected_modalities: tuple[str, ...] = ("EEG", "EOG", "EMG", "ECG", "RESP"),
    inspect_samples: bool = True,
) -> RecordingAudit:
    """Audit one EDF recording; optionally inspect sample-level quality.

    Only a small leading segment is read when ``inspect_samples`` is true, so a
    full night is never loaded into memory during inventory.
    """
    header = inspect_edf_header(path)
    site = ""
    parts = Path(path).parts
    for part in parts:
        if part in {"S0001", "I0002", "I0004", "I0006", "I0007"}:
            site = part
            break

    modality = {
        channel: resolve_modality(channel, rules, site=site) for channel in header.channels
    }
    available = sorted({value for value in modality.values() if value != "UNVERIFIED"})
    missing = sorted(set(expected_modalities) - set(available))

    quality: dict[str, dict] = {}
    reasons: list[str] = []
    if inspect_samples:
        try:
            import edfio
        except ImportError:  # pragma: no cover
            edfio = None
        if edfio is not None:
            edf = edfio.read_edf(str(path), lazy_load_data=False)
            for channel in edf.signals:
                label = str(channel.label).strip()
                data = np.asarray(channel.data, dtype=float)
                low, high = header.physical_limits.get(label, (None, None))
                report = signal_quality_report(data, header.sampling_rates.get(label), low, high)
                quality[label] = report
                for reason in report["reasons"]:
                    reasons.append(f"{label}:{reason}")

    return RecordingAudit(
        header=header,
        modality_by_channel=modality,
        available_modalities=available,
        missing_modalities=missing,
        channel_quality=quality,
        excluded_reasons=sorted(set(reasons)),
    )


def channel_availability_table(audits: list[RecordingAudit]) -> pd.DataFrame:
    """Build a channel x recording availability matrix (0/1)."""
    channels = sorted({channel for audit in audits for channel in audit.header.channels})
    rows = {}
    for audit in audits:
        rows[audit.header.path] = {channel: int(channel in audit.header.channels) for channel in channels}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=channels, fill_value=0)


def sampling_rate_inventory(audits: list[RecordingAudit]) -> pd.DataFrame:
    """Summarize sampling rates by channel across recordings."""
    records: list[dict] = []
    for audit in audits:
        for channel, rate in audit.header.sampling_rates.items():
            records.append({"channel": channel, "fs_hz": rate, "units": audit.header.units.get(channel, "")})
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=["channel", "fs_hz", "count", "min", "max", "n_distinct"])
    return (
        frame.groupby("channel")["fs_hz"]
        .agg(count="size", min="min", max="max", n_distinct="nunique")
        .reset_index()
    )
