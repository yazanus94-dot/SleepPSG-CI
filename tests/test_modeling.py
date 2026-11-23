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


def modality_fixture(n=12, seed=3):
    """Synthetic feature frame containing every extracted modality family."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SiteID": ["S0001"] * n,
        "BDSPPatientID": [f"p{i}" for i in range(n)],
        "recording": [f"sub-S0001111197789-ses{i + 1}" for i in range(n)],
        "Age": rng.integers(45, 85, n),
        "BMI": rng.normal(27, 3, n),
        "Sex": rng.choice(["Male", "Female"], n),
        "Cognitive_Impairment": rng.integers(0, 2, n),
        "eeg_delta_power": rng.normal(1, 0.2, n),
        "ecg_mean_hr": rng.normal(65, 6, n),
        "resp_rate_bpm": rng.normal(15, 2, n),
        "airflow_rate_bpm": rng.normal(15, 2, n),
        "airflow_amplitude_cv": rng.normal(0.3, 0.05, n),
        "spo2_mean": rng.normal(97, 1, n),
        "spo2_below_90_fraction": rng.uniform(0, 0.05, n),
        "emg_rms": rng.normal(2, 0.4, n),
        "eog_std": rng.normal(5, 1, n),
        "missing_airflow": 0.0,
        "missing_spo2": 0.0,
    })


def test_signal_and_combined_matrices_include_airflow_and_spo2():
    data = modality_fixture()
    for kind in ("signal", "combined"):
        matrix = select_feature_matrix(data, kind)
        for column in ("airflow_rate_bpm", "airflow_amplitude_cv",
                       "spo2_mean", "spo2_below_90_fraction"):
            assert column in matrix.columns, (kind, column)
        assert "missing_airflow" in matrix.columns
        assert "missing_spo2" in matrix.columns


def test_airflow_and_spo2_are_finite_or_correctly_imputed():
    data = modality_fixture()
    data.loc[0, "spo2_mean"] = np.nan
    data.loc[1, "airflow_rate_bpm"] = np.nan
    matrix = select_feature_matrix(data, "combined")
    assert np.isnan(matrix.loc[0, "spo2_mean"])
    assert np.isnan(matrix.loc[1, "airflow_rate_bpm"])

    model = make_model("combined_lr")
    probabilities = fit_predict(model, matrix, data["Cognitive_Impairment"], matrix)
    assert np.all(np.isfinite(probabilities))
    imputer = model.named_steps["imputer"]
    assert np.all(np.isfinite(imputer.statistics_))


def test_identifiers_and_labels_excluded_with_new_modalities():
    data = modality_fixture()
    for kind in ("signal", "combined"):
        columns = list(select_feature_matrix(data, kind).columns)
        for forbidden in ("SiteID", "BDSPPatientID", "recording", "Cognitive_Impairment"):
            assert forbidden not in columns
        assert any(column.startswith("airflow_") for column in columns)
        assert any(column.startswith("spo2_") for column in columns)


def test_signal_only_and_combined_baselines_fit_with_new_modalities():
    data = modality_fixture()
    y = data["Cognitive_Impairment"].to_numpy(dtype=int)
    for name in ("signal_lr", "signal_rf", "combined_lr", "combined_hgb"):
        model = make_model(name)
        matrix = select_feature_matrix(data, feature_kind_for(name))
        probabilities = fit_predict(model, matrix, y, matrix)
        assert probabilities.shape == (len(data),)
        assert np.all(np.isfinite(probabilities))
        assert np.all((probabilities >= 0) & (probabilities <= 1))
