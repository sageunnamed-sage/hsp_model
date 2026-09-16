"""
Machine-learning model training and cross-validation utilities.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut

from src.polymer_hsp.feature_selection import select_features

logger = logging.getLogger(__name__)

def run_random_forest_loocv(X: pd.DataFrame, y: pd.Series, max_features: int | None = None, random_state: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    Evalaute a Random Forest using nested LOOCV.

    Feature selectio is performed independently within each outer training fold.
    """

    loo = LeaveOneOut()

    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):


        # -------------------------
        # 1. Split outer CV fold
        # -------------------------

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        selection = select_features(X_train, y_train, correlation_threshold=0.40,
                                    vif_threshold=5.0, max_features=max_features)
        X_train_selected = X_train[selection.features]
        X_test_selected = X_test[selection.features]

        model = RandomForestRegressor(random_state=random_state)
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

    logger.info(f"Completed nested LOOCV with {len(y_true)} samples.")

    return np.asarray(y_true), np.asarray(y_pred)


