"""
Machine-learning model training and cross-validation utilities.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

from src.polymer_hsp.feature_selection import select_features

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

@dataclass(frozen=True)
class RandomForestResult:
    """
    Results produced by nested Random Forest model.
    """
    y_true: np.ndarray
    y_pred: np.ndarray
    sample_indices: np.ndarray
    selected_features: list[list[str]]

def run_random_forest_loocv(X: pd.DataFrame, y: pd.Series, config:RandomForestConfig| None = None) -> RandomForestResult:
    """
    Evaluate a Random Forest using nested LOOCV.

    Feature selectio is performed independently within each outer training fold.
    """
    sample_indices: list = []
    if config is None:
        config = RandomForestConfig()

    if X.empty:
        raise ValueError("X contains no features.")

    if y.empty:
        raise ValueError("y contains no targets.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of observations.")

    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("y must contain numeric solubility parameters.")

    if config.max_features is not None and config.max_features < 1:
        raise ValueError("max_features must be at least 1.")

    loo = LeaveOneOut()

    y_true: list[float] = []
    y_pred: list[float] = []
    selected_features: list[list[str]] = []


    for train_idx, test_idx in loo.split(X):
        # -------------------------
        # 1. Split outer CV fold
        # -------------------------

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]
        # -------------------------
        # 1. Feature Selection
        # -------------------------

        selection = select_features(X_train, y_train, correlation_threshold=config.correlation_threshold,
                                    vif_threshold=config.vif_threshold, max_features=config.max_features)
        selected_features.append(selection)
        X_train_selected = X_train[selection.features]
        X_test_selected = X_test[selection.features]

        # -------------------------
        # 1. Train model
        # -------------------------
        model =create_random_forest(random_state=config.random_state)
        model.fit(X_train_selected, y_train)

        # -------------------------
        # 4. Predict held-out sample
        # -------------------------

        prediction = model.predict(X_test_selected)

        # -------------------------
        # 5. Store prediction
        # -------------------------
        y_true.append(y_test.iloc[0])
        y_pred.append(prediction[0])

        sample_indices.append(X.index[test_idx[0]])

    logger.info(f"Completed nested LOOCV with {len(y_true)} samples.")

    return RandomForestResult(y_true=np.asarray(y_true), y_pred=np.asarray(y_pred), sample_indices=np.asarray(sample_indices), selected_features=selected_features)

def create_random_forest(random_state: int = 30) -> RandomForestRegressor:

    """
    Create the random forest regression model.
    """

    return RandomForestRegressor(random_state=random_state)





