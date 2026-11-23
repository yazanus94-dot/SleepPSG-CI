import numpy as np
import pandas as pd

from benchmark.evaluator import evaluate, evaluate_files, reproducibility_ok, validate_predictions
from benchmark.reference_solution import predict as reference_predict
from benchmark.starter_solution import predict as starter_predict


def labels():
    return pd.DataFrame({
        "SiteID": ["S0001", "S0001", "I0002", "I0002"],
        "BDSPPatientID": ["p1", "p2", "p3", "p4"],
        "Age": [60, 61, 62, 63],
        "BMI": [25.0, 26.0, 27.0, 28.0],
        "Cognitive_Impairment": [1, 0, 1, 0],
        "eeg_delta_power": [1.0, 2.0, 3.0, 4.0],
    })


def valid_predictions():
    return pd.DataFrame({
        "SiteID": ["S0001", "S0001", "I0002", "I0002"],
        "BDSPPatientID": ["p1", "p2", "p3", "p4"],
        "Cognitive_Impairment_Probability": [0.9, 0.2, 0.8, 0.1],
    })


def test_valid_prediction_passes_and_scores():
    result = evaluate(labels(), valid_predictions())
    assert result["valid"] is True
    assert result["metrics"]["auroc_age"] == 1.0


def test_malformed_predictions():
    good = valid_predictions()
    lab = labels()
    assert any("Row count" in error for error in validate_predictions(good.head(2), lab))
    assert any("NaN" in error for error in validate_predictions(good.assign(Cognitive_Impairment_Probability=np.nan), lab))
    assert any("[0, 1]" in error for error in validate_predictions(good.assign(Cognitive_Impairment_Probability=[1.2] * 4), lab))
    misaligned = good.copy()
    misaligned.loc[0, "BDSPPatientID"] = "wrong"
    assert any("alignment" in error for error in validate_predictions(misaligned, lab))
    duplicate = good.copy()
    duplicate.loc[1, "BDSPPatientID"] = "p1"
    assert any("Duplicate" in error for error in validate_predictions(duplicate, lab))
    missing = good.drop(columns=["Cognitive_Impairment_Probability"])
    assert any("Missing prediction column" in error for error in validate_predictions(missing, lab))


def test_reproducibility_check():
    good = valid_predictions()
    assert reproducibility_ok(good, good.copy())
    assert not reproducibility_ok(good, good.assign(Cognitive_Impairment_Probability=[0.1] * 4))


def test_starter_and_reference_contract():
    train = labels()
    evaluation = labels().drop(columns=["Cognitive_Impairment"])
    for solution in (starter_predict, reference_predict):
        predictions = solution(train, evaluation)
        assert validate_predictions(predictions, labels()) == []
        assert reproducibility_ok(predictions, solution(train, evaluation))


def test_evaluate_files(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    predictions_csv = tmp_path / "predictions.csv"
    labels().to_csv(labels_csv, index=False)
    valid_predictions().to_csv(predictions_csv, index=False)
    assert evaluate_files(labels_csv, predictions_csv)["valid"] is True
