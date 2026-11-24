"""Feature-schema consistency and fold-local missing-column handling."""

import warnings

import numpy as np
import pandas as pd

from sleeppsg_ci.modeling import _DropAllMissing, encode_demographics, make_model


def test_all_missing_training_column_dropped_without_imputer_warning():
    # Sex_male is entirely NaN in this training fold (no recorded sex).
    train = pd.DataFrame({"Age": [60, 61, 62, 63], "Sex_male": [np.nan] * 4})
    test = pd.DataFrame({"Age": [64, 65], "Sex_male": [np.nan, 1.0]})
    model = make_model("demo_lr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(train, [0, 1, 0, 1])
        model.predict_proba(test)
    messages = " ".join(str(w.message) for w in caught)
    assert "without any observed values" not in messages


def test_dropper_freezes_training_schema():
    transformer = _DropAllMissing().fit(
        pd.DataFrame({"present": [1.0, 2.0], "absent": [np.nan, np.nan]})
    )
    transformed = transformer.transform(
        pd.DataFrame({"present": [3.0], "absent": [np.nan]})
    )
    assert list(transformed.columns) == ["present"]
    assert transformer.dropped_ == ["absent"]


def test_demographic_schema_fixed():
    frame = pd.DataFrame({"Age": [60], "BMI": [25], "Sex": ["unknown"]})
    encoded = encode_demographics(frame)
    assert list(encoded.columns) == ["Age", "BMI", "Sex_male"]
    assert np.isnan(encoded["Sex_male"].iloc[0])
