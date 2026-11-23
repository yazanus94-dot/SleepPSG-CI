import pandas as pd
import pytest

from sleeppsg_ci.channel_audit import (
    ChannelMappingError,
    ChannelRule,
    EdfHeader,
    RecordingAudit,
    audit_recording,
    channel_availability_table,
    default_channel_rules,
    inspect_edf_header,
    load_channel_mapping,
    normalize_channel_name,
    resolve_modality,
    sampling_rate_inventory,
)


def test_normalization_and_verified_resolution():
    assert normalize_channel_name(" EEG-C3 ") == "EEG-C3"
    rules = (ChannelRule("EEG C3", "EEG", site="S0001", verified=True),)
    assert resolve_modality("eeg c3", rules, site="S0001") == "EEG"
    assert resolve_modality("EEG C3", rules, site="I0002") == "UNVERIFIED"
    assert resolve_modality("C4", rules, site="S0001") == "UNVERIFIED"


def test_unverified_mapping_never_labels_modality():
    rules = (ChannelRule("ECG", "ECG", verified=False),)
    assert resolve_modality("ECG", rules) == "UNVERIFIED"
    assert default_channel_rules() == ()


def test_invalid_mapping_rejected_and_csv_loader(tmp_path):
    with pytest.raises(ChannelMappingError):
        ChannelRule("X", "UNKNOWN_MODALITY", verified=True)
    path = tmp_path / "map.csv"
    pd.DataFrame({
        "channel": ["EEG", "ECG", "C4"],
        "modality": ["EEG", "ECG", "EEG"],
        "site": ["*", "*", "*"],
        "verified": ["true", "1", "no"],
    }).to_csv(path, index=False)
    rules = load_channel_mapping(path)
    assert len(rules) == 2
    assert resolve_modality("C4", rules) == "UNVERIFIED"


def test_inventory_tables_from_headers():
    header_a = EdfHeader("a.edf", ("EEG", "ECG"), {"EEG": 256.0, "ECG": 512.0},
                         {"EEG": "uV", "ECG": "mV"},
                         {"EEG": (None, None), "ECG": (None, None)}, 3600.0)
    header_b = EdfHeader("b.edf", ("EEG", "EOG"), {"EEG": 128.0, "EOG": 256.0},
                         {"EEG": "uV", "EOG": "uV"},
                         {"EEG": (None, None), "EOG": (None, None)}, 1800.0)
    audits = [RecordingAudit(header_a, {}, ["EEG", "ECG"], ["EOG"], {}),
              RecordingAudit(header_b, {}, ["EEG", "EOG"], ["ECG"], {})]
    availability = channel_availability_table(audits)
    assert availability.loc["a.edf", "ECG"] == 1
    assert availability.loc["a.edf", "EOG"] == 0
    rates = sampling_rate_inventory(audits)
    assert rates.loc[rates["channel"] == "EEG", "n_distinct"].iloc[0] == 2


def test_missing_edf_raises():
    with pytest.raises(FileNotFoundError):
        inspect_edf_header("does_not_exist.edf")


def test_synthetic_edf_inventory_and_extraction(tmp_path):
    """Software-only EDF integration test; no real clinical data is involved."""
    import numpy as np
    import edfio

    from sleeppsg_ci.feature_table import extract_from_edf

    signals = [
        edfio.EdfSignal(np.sin(np.arange(1280) / 10), 128.0, label="EEG",
                        physical_dimension="uV", physical_range=(-100, 100)),
        edfio.EdfSignal(np.sin(np.arange(1280) / 20), 128.0, label="ECG",
                        physical_dimension="mV", physical_range=(-5, 5)),
        edfio.EdfSignal(np.zeros(1280), 128.0, label="UNKNOWN_CH", physical_dimension=""),
    ]
    edf_path = tmp_path / "synthetic.edf"
    edfio.Edf(signals).write(edf_path)

    header = inspect_edf_header(edf_path)
    assert header.channels == ("EEG", "ECG", "UNKNOWN_CH")
    assert header.sampling_rates["EEG"] == 128.0
    assert header.units["EEG"] == "uV"

    audit = audit_recording(edf_path, (ChannelRule("EEG", "EEG", verified=True),))
    assert "EEG" in audit.available_modalities
    assert "ECG" not in audit.available_modalities  # mapping not supplied
    assert audit.modality_by_channel["UNKNOWN_CH"] == "UNVERIFIED"

    rules = (ChannelRule("EEG", "EEG", verified=True), ChannelRule("ECG", "ECG", verified=True))
    row = extract_from_edf(edf_path, rules=rules)
    assert row["missing_eeg"] == 0.0
    assert row["missing_ecg"] == 0.0
    assert row["missing_eog"] == 1.0
