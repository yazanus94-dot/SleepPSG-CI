"""Regression tests for the real-data experiment guard.

``scripts/evaluate_models.py`` is the only path that can produce real-data
results. These tests pin its source-partition guard so a future change cannot
silently admit hidden validation/test sources or unlabelled partitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_models import validate_sources  # noqa: E402

from sleeppsg_ci.data_access import HIDDEN_SOURCES, TRAINING_SOURCES  # noqa: E402


def test_all_labels_come_from_the_documented_partition():
    assert TRAINING_SOURCES == ("S0001", "I0002", "I0006")
    assert HIDDEN_SOURCES == ("I0004", "I0007")


@pytest.mark.parametrize("source", TRAINING_SOURCES)
def test_permitted_training_sources_are_accepted(source):
    frame = pd.DataFrame({"SiteID": [source, source]})

    assert validate_sources(frame) == [source]


@pytest.mark.parametrize("source", HIDDEN_SOURCES)
def test_hidden_sources_are_rejected(source):
    frame = pd.DataFrame({"SiteID": ["S0001", source]})

    with pytest.raises(ValueError, match="Hidden validation/test sources"):
        validate_sources(frame)


def test_unknown_sources_are_rejected():
    frame = pd.DataFrame({"SiteID": ["S0001", "I9999"]})

    with pytest.raises(ValueError, match="not permitted"):
        validate_sources(frame)


def test_missing_site_column_is_rejected():
    with pytest.raises(ValueError, match="no SiteID column"):
        validate_sources(pd.DataFrame({"PatientID": ["a"]}))


def test_blank_sources_do_not_count_as_observations():
    frame = pd.DataFrame({"SiteID": ["", "   "]})

    with pytest.raises(ValueError, match="no SiteID values"):
        validate_sources(frame)


def test_sources_are_returned_sorted_and_deduplicated():
    frame = pd.DataFrame({"SiteID": ["I0006", "S0001", "I0006"]})

    assert validate_sources(frame) == ["I0006", "S0001"]
