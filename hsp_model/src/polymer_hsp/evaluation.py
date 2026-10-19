"""Utilities for evaluating regression experiments."""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegressionMetrics:
    """Regression performance metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float


@dataclass(frozen=True)
class ExperimentResult:
    """Results from a completed regression experiment."""

    metrics: RegressionMetrics
    y_true: np.ndarray
    y_pred: np.ndarray
    sample_indices: np.ndarray
    selected_features: list[list[str]]


def _validate_regression_arrays(y_true: np.ndarray, y_pred: np.ndarray ) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate actual and predicted target values."""

    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Targets and predictions must be numeric.") from exc

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must contain the same number of observations.")

    if not np.isfinite(y_true).all():
        raise ValueError("y_true must contain only finite values.")

    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred must contain only finite values.")

    return y_true, y_pred


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Calculate regression metrics from actual and predicted values."""

    y_true, y_pred = _validate_regression_arrays(y_true, y_pred)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    logger.info("Regression metrics: MSE=%.4f, RMSE=%.4f, MAE=%.4f, R2=%.4f", mse, rmse, mae, r2)

    return RegressionMetrics(mse=mse, rmse=rmse, mae=mae, r2=r2)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Evaluate model predictions."""

    return calculate_regression_metrics(y_true=y_true, y_pred=y_pred)


def create_experiment_result(y_true: np.ndarray, y_pred: np.ndarray, sample_indices: np.ndarray, selected_features: list[list[str]]) -> ExperimentResult:
    """Create a validated result object from model predictions."""

    y_true, y_pred = _validate_regression_arrays(y_true, y_pred)
    sample_indices = np.asarray(sample_indices)
    
    if sample_indices.ndim != 1:
        raise ValueError("sample_indices must be one-dimensional.")

    if len(y_true) != len(sample_indices):
        raise ValueError("y_true and sample_indices must contain the same number "
            "of observations.")

    if len(y_true) != len(selected_features):
        raise ValueError("y_true and selected_features must contain the same number "
            "of observations.")

    metrics = calculate_regression_metrics(y_true, y_pred)

    return ExperimentResult(
        metrics=metrics,
        y_true=y_true.copy(),
        y_pred=y_pred.copy(),
        sample_indices=sample_indices.copy(),
        selected_features=[features.copy() for features in selected_features])