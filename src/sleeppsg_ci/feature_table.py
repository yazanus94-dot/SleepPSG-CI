"""Per-recording feature assembly with explicit missing-modality indicators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .channel_audit import ChannelRule, resolve_modality
from .features_eeg import extract_eeg_features
from .features_ecg import extract_ecg_features
from .features_emg import extract_emg_features
from .features_eog import extract_eog_features
from .features_resp import extract_resp_features

EXTRACTORS = {
    "EEG": extract_eeg_features,
    "ECG": extract_ecg_features,
    "RESP": extract_resp_features,
    "EMG": extract_emg_features,
    "EOG": extract_eog_features,
    "AIRFLOW": extract_resp_features,
}
EXPECTED_MODALITIES = ("EEG", "ECG", "RESP", "EMG", "EOG")


def extract_from_signals(
    signals: dict[str, tuple[np.ndarray, float]],
    rules: tuple[ChannelRule, ...] = (),
    site: str = "*",
) -> dict[str, float]:
    """Extract one feature vector from ``{channel: (samples, fs)}``.

    Missing modalities are represented with ``missing_<modality> = 1`` and no
    fabricated physiological values. Only verified channel mappings are used.
    """
    by_modality: dict[str, list[tuple[str, np.ndarray, float]]] = {}
    for channel, (samples, fs) in sorted(signals.items()):
        modality = resolve_modality(channel, rules, site=site)
        if modality in EXTRACTORS:
            by_modality.setdefault(modality, []).append((channel, np.asarray(samples), float(fs)))

    features: dict[str, float] = {}
    for modality in EXPECTED_MODALITIES:
        channels = by_modality.get(modality, [])
        features[f"missing_{modality.lower()}"] = 0.0 if channels else 1.0
        if not channels:
            continue
        _, samples, fs = channels[0]
        features.update(EXTRACTORS[modality](samples, fs))
    features["n_verified_modalities"] = float(
        sum(1 for modality in EXPECTED_MODALITIES if by_modality.get(modality))
    )
    return features


def extract_from_edf(
    path: str | Path,
    rules: tuple[ChannelRule, ...] = (),
    site: str = "*",
    max_segment_seconds: float = 600.0,
) -> dict[str, float]:
    """Extract a feature row from one EDF using a bounded leading segment.

    Only channels with verified mappings are read into memory, and at most one
    segment per channel, satisfying the full-night memory-efficiency rule.
    """
    try:
        import edfio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("edfio is required for EDF feature extraction") from exc

    edf_path = Path(path)
    if not edf_path.is_file():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    edf = edfio.read_edf(str(edf_path), lazy_load_data=True)
    signals: dict[str, tuple[np.ndarray, float]] = {}
    for edf_signal in edf.signals:
        label = str(edf_signal.label).strip()
        if resolve_modality(label, rules, site=site) not in EXTRACTORS:
            continue
        fs = float(edf_signal.sampling_frequency)
        if not np.isfinite(fs) or fs <= 0:
            continue
        data = np.asarray(edf_signal.data, dtype=float)
        limit = int(round(max_segment_seconds * fs))
        signals[label] = (data[:limit], fs)
    return extract_from_signals(signals, rules=rules, site=site)


def features_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Build a deterministic, column-sorted feature table from recording rows."""
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame
    return frame.reindex(sorted(frame.columns), axis=1)
