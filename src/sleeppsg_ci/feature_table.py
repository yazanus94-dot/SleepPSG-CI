"""Per-recording feature assembly with explicit missing-modality indicators.

Feature extraction processes the full available recording: a bounded chunk
reader yields globally aligned fixed-duration epochs, per-epoch features are
computed, and robust statistics (mean/median/std) are aggregated across epochs
and across all verified channels of a modality. Missing modalities are reported
with explicit indicators; no physiological value is fabricated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .channel_audit import ChannelRule, normalize_channel_name, resolve_modality
from .features_common import feature_summary
from .features_eeg import extract_eeg_features
from .features_ecg import extract_ecg_features
from .features_emg import extract_emg_features
from .features_eog import extract_eog_features
from .features_resp import extract_resp_features
from .features_spo2 import extract_spo2_features
from .preprocessing import epoch_signal
from .signal_reader import EdfChunkReader

EXTRACTORS = {
    "EEG": extract_eeg_features,
    "ECG": extract_ecg_features,
    "RESP": extract_resp_features,
    "AIRFLOW": extract_resp_features,
    "EMG": extract_emg_features,
    "EOG": extract_eog_features,
    "SPO2": extract_spo2_features,
}
EXPECTED_MODALITIES = ("EEG", "ECG", "RESP", "AIRFLOW", "EMG", "EOG", "SPO2")
SOURCE_PREFIX = {"RESP": "resp", "AIRFLOW": "resp"}
TARGET_PREFIX = {"RESP": "resp", "AIRFLOW": "airflow", "SPO2": "spo2"}
DEFAULT_CHUNK_SECONDS = 300.0
DEFAULT_EPOCH_SECONDS = 30.0


def _prefix_for(modality: str) -> str:
    return TARGET_PREFIX.get(modality, modality.lower())


def _reprefix(features: dict[str, float], modality: str) -> dict[str, float]:
    """Give shared extractors (resp/airflow) distinct feature namespaces."""
    source = SOURCE_PREFIX.get(modality)
    target = _prefix_for(modality)
    if source is None or source == target:
        return features
    return {
        (key.replace(f"{source}_", f"{target}_", 1) if key.startswith(f"{source}_") else key): value
        for key, value in features.items()
    }


def _channel_token(channel: str) -> str:
    token = normalize_channel_name(channel).lower().replace("-", "_").replace("+", "_")
    return token or "unknown"


def _summarize_epochs(
    rows: list[dict[str, float]], prefix: str, n_epochs: int, n_excluded: int
) -> dict[str, float]:
    """Aggregate per-epoch features into one robust night-level vector."""
    if not rows:
        return {f"{prefix}_features_valid": 0.0}
    result: dict[str, float] = {}
    for key in sorted({key for row in rows for key in row}):
        stats = feature_summary([row.get(key, float("nan")) for row in rows])
        result[key] = stats["mean"]
        if not key.endswith(("_n_gaps", "_n_segments", "_n_samples", "_features_valid",
                             "_nonfinite_fraction", "_n_beats")):
            result[f"{key}__median"] = stats["median"]
            result[f"{key}__std"] = stats["std"]
    result[f"{prefix}_n_epochs"] = float(n_epochs)
    result[f"{prefix}_n_excluded_epochs"] = float(n_excluded)
    return result


def _aggregate_channels(
    channel_features: dict[str, dict[str, float]], modality: str
) -> dict[str, float]:
    """Aggregate all verified channels of one modality deterministically."""
    prefix = _prefix_for(modality)
    if not channel_features:
        return {f"missing_{modality.lower()}": 1.0}
    output: dict[str, float] = {f"missing_{modality.lower()}": 0.0}
    output[f"n_channels_{modality.lower()}"] = float(len(channel_features))

    common_keys = sorted({key for features in channel_features.values() for key in features})
    for key in common_keys:
        values = [features[key] for features in channel_features.values() if key in features]
        if values:
            output[key] = float(np.mean(values))
    for channel, features in sorted(channel_features.items()):
        token = _channel_token(channel)
        for key, value in sorted(features.items()):
            output[f"{prefix}_ch_{token}_{key}"] = float(value)
    return output


def _night_features_from_arrays(
    samples: np.ndarray,
    fs: float,
    modality: str,
    epoch_seconds: float,
) -> tuple[dict[str, float], int, int]:
    """Compute and aggregate per-epoch features from one in-memory channel."""
    array = np.asarray(samples, dtype=float).reshape(-1)
    prefix = _prefix_for(modality)
    epoch_samples = int(round(fs * epoch_seconds))
    if array.size >= epoch_samples and epoch_samples > 0:
        epochs = epoch_signal(array, fs, epoch_seconds)
        n_excluded = int(array.size - epochs.size)
    else:
        # Short-recording fallback: one partial epoch, documented in the output.
        epochs = array.reshape(1, -1) if array.size else np.empty((0, 0))
        n_excluded = 0
    rows: list[dict[str, float]] = []
    for epoch in epochs:
        features = _reprefix(EXTRACTORS[modality](epoch, fs), modality)
        if features.get(f"{prefix}_features_valid", 0.0) == 1.0:
            rows.append(features)
    n_excluded += len(epochs) - len(rows)
    return _summarize_epochs(rows, prefix, len(epochs), n_excluded), len(epochs), n_excluded


def extract_from_signals(
    signals: dict[str, tuple[np.ndarray, float]],
    rules: tuple[ChannelRule, ...] = (),
    site: str = "*",
    epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
) -> dict[str, float]:
    """Extract night-level features from ``{channel: (samples, fs)}``.

    Every verified channel is retained: channel-specific features use stable
    ``<modality>_ch_<channel>_`` prefixes, and the unprefixed features are the
    documented mean across channels. Only verified mappings are used.
    """
    by_modality: dict[str, dict[str, tuple[np.ndarray, float]]] = {}
    for channel, (samples, fs) in sorted(signals.items()):
        modality = resolve_modality(channel, rules, site=site)
        if modality in EXTRACTORS:
            by_modality.setdefault(modality, {})[channel] = (np.asarray(samples), float(fs))

    features: dict[str, float] = {}
    for modality in EXPECTED_MODALITIES:
        channels = by_modality.get(modality, {})
        per_channel: dict[str, dict[str, float]] = {}
        for channel, (samples, fs) in channels.items():
            if not np.isfinite(fs) or fs <= 0 or samples.size == 0:
                continue
            night, _, _ = _night_features_from_arrays(samples, fs, modality, epoch_seconds)
            per_channel[channel] = night
        features.update(_aggregate_channels(per_channel, modality))
    features["n_verified_modalities"] = float(
        sum(1 for modality in EXPECTED_MODALITIES if by_modality.get(modality))
    )
    return features


def extract_from_edf(
    path: str | Path,
    rules: tuple[ChannelRule, ...] = (),
    site: str = "*",
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
) -> dict[str, float]:
    """Extract full-recording features using bounded partial EDF reads.

    ``get_data_slice`` reads only one chunk at a time; ``EdfSignal.data`` is
    never accessed. Chunk size affects I/O only: epochs are aligned to absolute
    sample offsets, so results are equivalent across chunk sizes.
    """
    edf_path = Path(path)
    if not edf_path.is_file():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    by_modality: dict[str, dict[str, dict[str, float]]] = {}
    with EdfChunkReader(edf_path) as reader:
        duration = reader.duration_seconds
        for channel in reader.channels():
            modality = resolve_modality(channel, rules, site=site)
            if modality not in EXTRACTORS:
                continue
            fs = reader.sampling_rate(channel)
            if not np.isfinite(fs) or fs <= 0:
                continue
            prefix = _prefix_for(modality)
            epochs, info = reader.iter_epochs(channel, chunk_seconds, epoch_seconds)
            if info.n_epochs == 0:
                # Whole (short) channel through the partial-read API.
                samples = reader.read_channel(channel)
                epochs = iter([samples])
            rows: list[dict[str, float]] = []
            for epoch in epochs:
                features = _reprefix(EXTRACTORS[modality](epoch, info.fs), modality)
                if features.get(f"{prefix}_features_valid", 0.0) == 1.0:
                    rows.append(features)
            n_excluded = int(info.n_epochs - len(rows))
            if info.n_epochs == 0:
                n_excluded = 0
            night = _summarize_epochs(rows, prefix, info.n_epochs, n_excluded)
            by_modality.setdefault(modality, {})[channel] = night

    features: dict[str, float] = {}
    for modality in EXPECTED_MODALITIES:
        features.update(_aggregate_channels(by_modality.get(modality, {}), modality))
    features["n_verified_modalities"] = float(
        sum(1 for modality in EXPECTED_MODALITIES if by_modality.get(modality))
    )
    features["signal_duration_seconds"] = float(duration)
    return features


def features_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a deterministic, column-sorted feature table from recording rows."""
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame
    return frame.reindex(sorted(frame.columns), axis=1)
