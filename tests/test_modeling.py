import numpy as np
import pandas as pd
import pytest

from sleeppsg_ci.modeling import (
    build_baselines,
    feature_kind_for,
    fit_predict,
    make_model,
    model_config_hash,
    select_feature_matrix,
)


def frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SiteID": ["S0001"] * n,
        "BDSPPatientID": [f"p{i}" for i in range(n)],
        "Age": rng.integers(40, 90, n),
        "BMI": rng.normal(27, 4, n),
        "Sex": rng.choice(["Male", "Female"], n),
        "Cognitive_Impairment": rng.integers(0, 2, n),
        "eeg_delta_power": rng.normal(0, 1, n),
        "missing_ecg": rng.integers(0, 2, n),
    })


def test_all_six_baselines_build_and_fit():
    data = frame()
    for name, model in build_baselines().items():
        kind = feature_kind_for(name)
        x = select_feature_matrix(data, kind)
        y = data["Cognitive_Impairment"].to_numpy()
        probabilities = fit_predict(model, x, y, x)
        assert probabilities.shape == (len(data),)
        assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("not_a_model")


def test_seeded_models_are_reproducible():
    data = frame()
    x = select_feature_matrix(data, "combined")
    y = data["Cognitive_Impairment"].to_numpy()
    first = fit_predict(make_model("signal_rf", 42), x, y, x)
    second = fit_predict(make_model("signal_rf", 42), x, y, x)
    assert np.allclose(first, second)


def test_config_hash_stable():
    assert model_config_hash(make_model("combined_lr")) == model_config_hash(make_model("combined_lr"))
    assert model_config_hash(make_model("demo_lr")) != model_config_hash(make_model("signal_rf"))


def test_demographic_encoding():
    encoded = select_feature_matrix(frame(), "demographic")
    assert list(encoded.columns) == ["Age", "BMI", "Sex_male"]
    assert set(encoded["Sex_male"].unique()).issubset({0.0, 1.0})


def test_unknown_and_missing_sex_stay_missing_until_imputation():
    data = pd.DataFrame({
        "Age": [60, 61, 62, 63, 64],
        "BMI": [25.0, 26.0, 27.0, 28.0, 29.0],
        "Sex": ["Male", "M", "Female", "Unknown", None],
        "Cognitive_Impairment": [1, 0, 1, 0, 1],
        "eeg_delta_power": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    encoded = select_feature_matrix(data, "demographic")
    values = list(encoded["Sex_male"])
    assert values[:2] == [1.0, 1.0]
    assert values[2] == 0.0
    assert np.isnan(values[3]) and np.isnan(values[4])

    model = make_model("demo_lr")
    model.fit(encoded, data["Cognitive_Impairment"])
    assert not np.isnan(model.named_steps["imputer"].statistics_[2])


def test_eval_label_booleans_normalized_before_training():
    from sleeppsg_ci.evaluation import prepare_labelled

    data = pd.DataFrame({
        "SiteID": ["S0001", "S0001", "I0002"],
        "BDSPPatientID": ["p1", "p2", "p3"],
        "Age": [60, 61, 62],
        "Cognitive_Impairment": ["true", "no", "other"],
    })
    prepared = prepare_labelled(data)
    assert list(prepared["Cognitive_Impairment"]) == [1, 0]
    assert "patient_key" in prepared.columns
