"""
Model evaluation utilties
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RegressionMetrics:
    """
    Regression peformance metrics.
    """
    mse: float
    rmse: float
    r2: float

@dataclass(frozen=True)
class ExperimentResult:
    """
    Results from a completed regression experiment.
    """

    metrics: RegressionMetrics
    y_true: np.ndarray
    y_pred: np.ndarray
    sample_indices: np.ndarray
    selected_features: list[list[str]]


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """
    Calculate standard regression metrics using nested LOOCV.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of observations.")

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if not np.isfinite(y_true).all():
        raise ValueError("y_true must contain only finite values.")

    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred must contain only finite values.")


    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    logger.info(f"Regression metrics: {mse:.4f}, {rmse:.4f}, {r2:.4f}")

    return RegressionMetrics(mse=mse, rmse=rmse, r2=r2)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """
    Evaluate model predictions.
    """

    return calculate_regression_metrics(y_true=y_true, y_pred=y_pred)

def create_experiment_result(y_true: np.ndarray, y_pred: np.ndarray, sample_indices: np.ndarray,
                             selected_features: list[list[str]]) -> ExperimentResult:
    """
    Create a complete experiment result from model predictions.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of observations.")

    if len(y_true) != len(sample_indices):
        raise ValueError("y_true and sample_indices must contain the same number of observations.")

    if len(y_true) != len(selected_features):
        raise ValueError("y_true and selected_features must contain the same number of observations.")

    metrics = calculate_regression_metrics(y_true=y_true, y_pred=y_pred)

    return ExperimentResult(metrics=metrics, y_true=np.asarray(y_true), y_pred=np.asarray(y_pred),
                            sample_indices=np.asarray(sample_indices), selected_features=selected_features)



