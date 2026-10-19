"""
Machine-learning model training and cross-validation utilities.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

from polymer_hsp.feature_selection import select_features

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RandomForestConfig:
    """
    Configuration for the nested Random Forest model.
    """
    correlation_threshold: float  = 0.40
    vif_threshold: float = 5.0
    max_features: int | None = None
    random_state: int = 30
    use_correlation_filter: bool = True

@dataclass(frozen=True)
class RandomForestResult:
    """
    Results produced by nested Random Forest model.
    """
    y_true: np.ndarray
    y_pred: np.ndarray
    sample_indices: np.ndarray
    selected_features: list[list[str]]


def create_random_forest(random_state: int = 30) -> RandomForestRegressor:
    """Create a Random Forest regression model."""

    return RandomForestRegressor(random_state=random_state)

def run_random_forest_loocv(X: pd.DataFrame, y: pd.Series, config: RandomForestConfig | None = None) -> RandomForestResult:
    """Evaluate a Random Forest using outer LOOCV.

    Feature selection must be performed using only each outer
    training fold. Inner-CV feature selection should refit all
    selection steps independently within its own training folds.
    """

    if config is None:
        config = RandomForestConfig()

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if X.empty or X.shape[1] == 0:
        raise ValueError("X contains no features.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of observations.")

    if len(X) < 3:
        raise ValueError("At least 3 observations are required.")

    if not X.index.equals(y.index):
        raise ValueError("X and y indexes must match.")

    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("y must contain numeric targets.")

    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("y contains missing or infinite values.")

    if config.max_features is not None and config.max_features < 1:
        raise ValueError("max_features must be at least 1.")

    loo = LeaveOneOut()

    y_true: list[float] = []
    y_pred: list[float] = []
    sample_indices: list[object] = []
    selected_features: list[list[str]] = []

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Feature selection: training fold only.
        selection = select_features(
            X_train=X_train,
            y_train=y_train,
            correlation_threshold=config.correlation_threshold,
            vif_threshold=config.vif_threshold,
            max_features=config.max_features,
            use_correlation_filter=config.use_correlation_filter,
            estimator=create_random_forest(random_state=config.random_state))

        chosen_features = selection.features
        selected_features.append(chosen_features)

        X_train_selected = X_train.loc[:, chosen_features]
        X_test_selected = X_test.loc[:, chosen_features]

        # Fit the model on the outer training fold.
        model = create_random_forest(random_state=config.random_state)
        model.fit(X_train_selected, y_train)

        # Predict the held-out observation.
        prediction = model.predict(X_test_selected)

        y_true.append(float(y_test.iloc[0]))
        y_pred.append(float(prediction[0]))
        sample_indices.append(X.index[test_idx[0]])

    logger.info("Completed outer LOOCV with %d observations.", len(y_true))

    return RandomForestResult(
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        sample_indices=np.asarray(sample_indices),
        selected_features=selected_features)
