import numpy as np
import pytest

from src.polymer_hsp.evaluation import RegressionMetrics, calculate_regression_metrics, evaluate_predictions, \
    create_experiment_result, ExperimentResult


def test_calculate_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = calculate_regression_metrics( y_true, y_pred)
    assert isinstance(metrics, RegressionMetrics)
    assert metrics.mse >= 0
    assert metrics.rmse >= 0
    assert metrics.r2 <= 1

def test_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = calculate_regression_metrics( y_true, y_pred)
    assert metrics.mse == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)

def test_mismatched_lengths_raise_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="y_true and y_pred must contain the same number of observations."):
        calculate_regression_metrics( y_true, y_pred)

def test_empty_arrays_raise_error():
    y_true = np.array([])
    y_pred = np.array([])
    with pytest.raises(ValueError, match="y_true and y_pred must not be empty."):
        calculate_regression_metrics( y_true, y_pred)

def test_non_finite_true_values_raise_error():
    y_true = np.array([1.0, np.nan, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        calculate_regression_metrics( y_true, y_pred)

def test_non_finite_predictions_raise_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.inf, 3.0])

    with pytest.raises(ValueError, match="finite"):
        calculate_regression_metrics( y_true, y_pred)


def test_evaluate_predictions_matches_metric_calculation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    direct_metrics = calculate_regression_metrics(y_true, y_pred)

    evaluated_metrics = evaluate_predictions(y_true, y_pred)

    assert evaluated_metrics == direct_metrics

def test_create_experiment_result():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.0, 2.9])
    sample_indices = np.array(["sample_a", "sample_b", "sample_c"])
    selected_features = [
        ["descriptor_1", "descriptor_2"],
        ["descriptor_1"],
        ["descriptor_1", "descriptor_3"],
    ]

    result = create_experiment_result(
        y_true=y_true,
        y_pred=y_pred,
        sample_indices=sample_indices,
        selected_features=selected_features,
    )

    assert isinstance(result, ExperimentResult)
    assert isinstance(result.metrics, RegressionMetrics)
    assert len(result.y_true) == 3
    assert len(result.y_pred) == 3
    assert len(result.sample_indices) == 3
    assert len(result.selected_features) == 3


def test_create_experiment_result_rejects_mismatched_predictions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    sample_indices = np.array(["sample_a", "sample_b", "sample_c"])
    selected_features = [
        ["descriptor_1"],
        ["descriptor_1"],
        ["descriptor_1"],
    ]

    with pytest.raises(ValueError, match="y_true and y_pred"):
        create_experiment_result(
            y_true=y_true,
            y_pred=y_pred,
            sample_indices=sample_indices,
            selected_features=selected_features,
        )


def test_create_experiment_result_rejects_mismatched_sample_indices():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    sample_indices = np.array(["sample_a", "sample_b"])
    selected_features = [["descriptor_1"], ["descriptor_1"], ["descriptor_1"]]

    with pytest.raises(ValueError, match="sample_indices"):
        create_experiment_result(y_true=y_true, y_pred=y_pred, sample_indices=sample_indices,
                                 selected_features=selected_features)


def test_create_experiment_result_rejects_mismatched_selected_features():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    sample_indices = np.array(["sample_a", "sample_b", "sample_c"])
    selected_features = [["descriptor_1"], ["descriptor_1"]]

    with pytest.raises(ValueError, match="selected_features"):
        create_experiment_result(y_true=y_true, y_pred=y_pred, sample_indices=sample_indices,
                                 selected_features=selected_features)

