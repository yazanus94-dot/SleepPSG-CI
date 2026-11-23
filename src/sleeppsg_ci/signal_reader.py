"""Memory-conscious EDF reading using genuine partial (chunk) reads.

``edfio.EdfSignal.get_data_slice(start_second, stop_second)`` reads only the
requested slice when the signal has not been fully materialized. This module
never touches ``EdfSignal.data``, so a full-night recording is processed one
bounded chunk at a time. Header-only duration comes from ``Edf.duration``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class EdfReadError(RuntimeError):
    """Raised when an EDF cannot be opened or a channel is unavailable."""


@dataclass(frozen=True)
class ChannelEpochInfo:
    """Header-derived metadata about the epoch stream for one channel."""

    fs: float
    epoch_seconds: float
    epoch_samples: int
    n_epochs: int
    n_excluded_samples: int
    n_chunks: int
    duration_seconds: float


class EdfChunkReader:
    """Open an EDF once and stream bounded slices from individual channels."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._edf = None
        self._channel_map_cache: dict[str, object] | None = None
        self._original_labels_cache: dict[str, str] | None = None

    def __enter__(self) -> "EdfChunkReader":
        try:
            import edfio
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("edfio is required for EDF access") from exc
        if not self.path.is_file():
            raise FileNotFoundError(f"EDF file not found: {self.path}")
        try:
            self._edf = edfio.read_edf(str(self.path), lazy_load_data=True)
        except Exception as exc:
            raise EdfReadError(f"Could not read EDF header for {self.path}: {exc}") from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._edf = None
        self._channel_map_cache = None
        self._original_labels_cache = None

    @property
    def edf(self):
        if self._edf is None:
            raise EdfReadError("EdfChunkReader must be used as a context manager.")
        return self._edf

    @property
    def duration_seconds(self) -> float:
        """Header-derived recording duration; no signal samples are read."""
        return float(self.edf.duration)

    def _channel_map(self) -> dict[str, object]:
        """Map unique display labels to signals, disambiguating duplicates.

        Real EDF files can repeat a label (e.g. two channels both named ``EEG``).
        Collapsing them would silently discard a channel, so later duplicates
        get a deterministic ``__2``, ``__3`` suffix. The raw label is retained in
        :meth:`original_channel_labels` for provenance.
        """
        if self._channel_map_cache is None or self._original_labels_cache is None:
            mapping: dict[str, object] = {}
            originals: dict[str, str] = {}
            counts: dict[str, int] = {}
            for signal in self.edf.signals:
                raw = str(signal.label).strip() or f"CH{len(mapping) + 1}"
                counts[raw] = counts.get(raw, 0) + 1
                display = raw if counts[raw] == 1 else f"{raw}__{counts[raw]}"
                mapping[display] = signal
                originals[display] = raw
            self._channel_map_cache = mapping
            self._original_labels_cache = originals
        return self._channel_map_cache

    def channels(self) -> list[str]:
        return list(self._channel_map())

    def original_channel_labels(self) -> dict[str, str]:
        """Map disambiguated display labels back to the raw EDF labels."""
        self._channel_map()
        assert self._original_labels_cache is not None
        return dict(self._original_labels_cache)

    def duplicate_channel_labels(self) -> list[str]:
        """Raw labels that occurred more than once in the EDF header."""
        counts: dict[str, int] = {}
        for raw in self.original_channel_labels().values():
            counts[raw] = counts.get(raw, 0) + 1
        return sorted(label for label, count in counts.items() if count > 1)

    def _signal(self, label: str):
        try:
            return self._channel_map()[label]
        except KeyError as exc:
            raise EdfReadError(f"Channel '{label}' not found in {self.path.name}") from exc

    def sampling_rate(self, label: str) -> float:
        return float(self._signal(label).sampling_frequency)

    def physical_limits(self, label: str) -> tuple[float | None, float | None]:
        signal = self._signal(label)
        low = getattr(signal, "physical_min", None)
        high = getattr(signal, "physical_max", None)
        return (float(low) if low is not None else None, float(high) if high is not None else None)

    def units(self, label: str) -> str:
        """Raw EDF physical dimension, preserved exactly as recorded."""
        return str(getattr(self._signal(label), "physical_dimension", "")).strip()

    def channel_units(self) -> dict[str, str]:
        """Header-only map of channel label -> original physical dimension."""
        return {label: self.units(label) for label in self.channels()}

    def channel_metadata(self, label: str) -> dict:
        """Header-only unit/rate metadata used for unit harmonization."""
        low, high = self.physical_limits(label)
        return {
            "label": label,
            "original_unit": self.units(label),
            "sampling_frequency": self.sampling_rate(label),
            "physical_min": low,
            "physical_max": high,
        }

    def read_channel(self, label: str) -> np.ndarray:
        """Read a whole channel through the partial-read API (short files only)."""
        signal = self._signal(label)
        return np.asarray(signal.get_data_slice(0.0, self.duration_seconds), dtype=float)

    def iter_chunks(self, label: str, chunk_seconds: float) -> Iterator[tuple[float, np.ndarray]]:
        """Yield ``(start_second, samples)`` for each bounded chunk of a channel."""
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive.")
        signal = self._signal(label)
        duration = self.duration_seconds
        start = 0.0
        while start < duration - 1e-9:
            stop = min(start + chunk_seconds, duration)
            samples = np.asarray(signal.get_data_slice(start, stop), dtype=float)
            if samples.size:
                yield start, samples
            start = stop

    def iter_epochs(
        self, label: str, chunk_seconds: float, epoch_seconds: float
    ) -> tuple[Iterator[np.ndarray], ChannelEpochInfo]:
        """Yield globally aligned, fixed-duration epochs without loading the night.

        Epoch boundaries are computed from absolute sample offsets, so using a
        different chunk size changes I/O granularity but not the epochs yielded.
        """
        if epoch_seconds <= 0:
            raise ValueError("epoch_seconds must be positive.")
        fs = self.sampling_rate(label)
        if not np.isfinite(fs) or fs <= 0:
            raise EdfReadError(f"Invalid sampling rate for channel '{label}': {fs}")
        duration = self.duration_seconds
        epoch_samples = int(round(fs * epoch_seconds))
        if epoch_samples < 1:
            raise ValueError("Epoch is shorter than one sample at this sampling rate.")
        n_epochs = int(duration // epoch_seconds)
        n_chunks = max(1, int(np.ceil(duration / chunk_seconds))) if chunk_seconds > 0 else 1

        info = ChannelEpochInfo(
            fs=fs,
            epoch_seconds=epoch_seconds,
            epoch_samples=epoch_samples,
            n_epochs=n_epochs,
            n_excluded_samples=int(round((duration - n_epochs * epoch_seconds) * fs)),
            n_chunks=n_chunks,
            duration_seconds=duration,
        )

        def generator() -> Iterator[np.ndarray]:
            # Buffer across chunks so a chunk smaller than one epoch still
            # contributes. Epoch boundaries are anchored at sample 0, making
            # the yielded epochs identical for every chunk size. Yielding is
            # capped at the header-derived count so a duration/sample-count
            # mismatch can never produce more epochs than were declared.
            buffer = np.empty(0, dtype=float)
            yielded = 0
            for _, samples in self.iter_chunks(label, chunk_seconds):
                buffer = np.concatenate([buffer, samples]) if buffer.size else samples
                while buffer.size >= epoch_samples and yielded < n_epochs:
                    yield buffer[:epoch_samples].copy()
                    buffer = buffer[epoch_samples:]
                    yielded += 1
                if yielded >= n_epochs:
                    break

        return generator(), info
