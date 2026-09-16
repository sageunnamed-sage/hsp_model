"""
Feature Selection and multicollinearity utilities
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.externals.array_api_extra import nunique
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FeatureSelectionResult:
    """
    Result from recursive feature elimination.
    """
    features: list[str]
    mse_by_feature_count: dict[int, float]
    best_feature_count: int

def select_uncorrelated_features(X_train: pd.DataFrame, threshold: float = 0.40) -> list[str]:
    """
    Select features whose absolute pairwise correlation does not exceed the specified threshold.
    :param X_train: Training features dataframe
    :param threshold: Aboslute correlation threshold
    :return: Feature names retained after correlation
    """

    correlation = X_train.corr(numeric_only=True).abs()

    mask = np.triu(np.ones(correlation.shape), k=1).astype(bool)
    upper_triangle = correlation.where(mask)

    features_to_drop = [column for column in upper_triangle.columns
                        if any(upper_triangle[column] > threshold) ]

    select_features = [column for column in upper_triangle.columns
                       if column not in features_to_drop]

    logger.info(f"Correlatin filtering removed {len(features_to_drop)} features.")

    return select_features

def selected_features_by_vif(X_train: pd.DataFrame, threshold: float = 5.0) -> list[str]:
    """
    Remove features with high variance inflation factors.
    VIF is calculated using only the training data
    :param X_train: Training features dataframe.
    :param threshold: Maximum acceptable VIF
    :return: Feature names retained after VIF filetering.
    """

    X = X_train.copy()

    # Keep only numeric columns
    X = X.select_dtypes(include="number")

    # Remove constant columns.
    nunique = X.nunique()
    constant_features = nunique[nunique <= 1].index.tolist()

    if constant_features:
        logger.info(f"Removing {len(constant_features)} features before VIF calculation.")

        X = X.drop(columns=constant_features)

    # Iteratively remove the featuere with the highest VIF
    while X.shape[1] > 1:
        vif_values = pd.Series([variance_inflation_factor(X.values, i)
                                for i in range(X.shape[1])], index=X.columns)
        max_vif = vif_values.max()
        if max_vif <= threshold:
            break
        feature_to_remove = vif_values.idxmax()
        logger.debug(f"Removing {feature_to_remove} with VIF {max_vif:.2f}.")

        X = X.drop(columns=[feature_to_remove])

    logger.info(f"VIF filtering retained {X.shape[1]} of {X_train.shape[1]} features.")

    return X.columns.tolist()

def selected_features_with_inner_cv(X_train: pd.DataFrame, y_train: pd.Series, estimator=None, max_features: int | None = None) -> FeatureSelectionResult:
    """
    Select the number of features using an inner cross-validation loop.
    :param X_train: Training features from the outer cross-validation loop.
    :param y_train: Training target from the outer cross-validation loop.
    :param estimator:  Estimator used by RFE
    :param max_features: Maximum number of features to evaluate.
    :return: FeatureSelectionResult. Selected feature names and inner-CV performance
    """

    if estimator is None:
        estimator = LinearRegression()

    if X_train.empty:
        raise ValueError("X_train contains no features.")

    max_features = min(max_features or X_train.shape[1], X_train.shape[1])

    inner_cv =LeaveOneOut()

    mse_by_feature_count: dict[int, float] = {}

    for n_features in range(1, max_features + 1):
        pipeline = Pipeline([(
            "feature_selection", RFE(estimator=estimator, n_features_to_select=n_features, step=1)),("model", estimator)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=inner_cv, scoring="neg_mean_squared_error")
        mse_by_feature_count[n_features] = -scores.mean()

    best_n = min(mse_by_feature_count, key=mse_by_feature_count.get)

    # Fit RFE once more on the complete outer training set.
    final_selector = RFE(estimator=estimator, n_features_to_select=best_n, step=1)
    final_selector.fit(X_train, y_train)
    selected_featurs = X_train.columns[final_selector.support_].tolist()

    logger.info(f"Selected {best_n} features using inner CV")

    return FeatureSelectionResult(features=selected_featurs, mse_by_feature_count=mse_by_feature_count, best_feature_count=best_n)

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a Pearson correlation matrix after removing empty rows/columns
    """

    corr = df.corr(numeric_only=True)

    corr = corr.dropna(axis=0, how='all')
    corr = corr.dropna(axis=1, how='all')

    logger.info(f"Correlation matrix computed {corr.shape} x {corr.shape}")

    return corr

def correlated_feature_pairs(corr : pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """
    Reutrn feature pairs whose absolute correlation is above the threshold.
    """

    pairs = []

    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1:]:
            value = corr.loc[col1, col2]

            if abs(value) >= threshold:
                pairs.append((col1, col2, value))

    logger.info(f"Found {len(pairs)} correlated feature pairs.")

    return pairs

def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate variance inflation factors for each feature.
    """

    numeric = df.select_dtypes(include="number").dropna().copy()

    if numeric.empty:
        raise ValueError("No numeric features available for VIF calculation.")

    # VIF is not meaningful for constant features
    nunique = numeric.nunique()
    constant_features = nunique[nunique <= 1].index.tolist()

    if constant_features:
        numeric = numeric.drop(columns=constant_features)

    if numeric.empty:
        raise ValueError("No non-constant numeric features available for VIF calculation.")

    # A single feature has no multicollinearity with another feature
    if numeric.shape[1] == 1:
        return pd.DataFrame(
            {"feature": numeric.columns,
             "vif": [1.0]}
        )

    vif = pd.DataFrame(
        {
            "feature": numeric.columns,
            "vif": [
                variance_inflation_factor(numeric.values, i)
                for i in range(numeric.shape[1])
            ]
        })

    return vif.sort_values("vif", ascending=False).reset_index(drop=True)

def find_optimal_feature_count(X: pd.DataFrame, y: pd.Series, estimator=None, max_features: int | None = None) -> dict[int, float]:
    """
    Evaluate feature counts using LOOCV mean squared error
    """
    if estimator is None:
        estimator = LinearRegression()

    if max_features is None:
        max_features = X.shape[1]

    loo = LeaveOneOut()
    scores = {}

    for n_features in range(1, max_features + 1):
        selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)

        selector.fit(X, y)
        selected = X.columns[selector.support_]

        mse = -cross_val_score(estimator, X[selected], y, cv=loo, scoring="neg_mean_squared_error").mean()

        scores[n_features] = mse

    logger.info(f"Evaluated {len(scores)} feature counts.")

    return scores

# def select_features_rfe(X: pd.DataFrame, y: pd.Series, estimator=None, max_features: int | None = None) -> list[str]:
#     """
#     Select an optimal subset of features using recursive feature elimination.
#     """
#     if estimator is None:
#         estimator = LinearRegression()
#
#     scores = find_optimal_feature_count(X, y, estimator, max_features)
#
#     best_n = min(scores, key=scores.get)
#
#     selector = RFE(estimator=estimator, n_features_to_select=best_n, step=1)
#
#     selector.fit(X, y)
#
#     selected = X.columns[selector.support_].tolist()
#     logger.info(f"Selected {len(selected)} features with LOOCV MSE {scores[best_n]:.3f}.")
#
#     return FeatureSelectionResult(features=selected, mse_by_feature=scores, best_feature_count=best_n)


def select_features_rfe(X_train: pd.DataFrame, y_train: pd.Series, estimator=None, max_features: int | None = None) -> FeatureSelectionResult:
    """
    Select features using RFE and inner cross-validation.

    This function should only receive training data. It should not have access
    to the outer test fold.
    """

    if estimator is None:
        estimator = LinearRegression()

    if max_features is None:
        max_features = X_train.shape[1]

    loo = LeaveOneOut()
    scores = {}

    for n_features in range(1, max_features + 1):
        selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)

        # Evaluate selection happens only on the training data
        selector.fit(X_train, y_train)

        selected_features = X_train.columns[selector.support_]

        # Evaluate selection happens on the training data
        mse = -cross_val_score(estimator, X_train[selected_features], y_train, cv=loo, scoring="neg_mean_squared_error").mean()
        scores[n_features] = mse

    best_n = min(scores, key=scores.get)

    # Refit RFE on the complete training set using the selected number of features
    selector = RFE(estimator=estimator, n_features_to_select=best_n, step=1)
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.support_].tolist()

    logger.info(f"Selected {best_n} features with inner-CV MSE {scores[best_n]:.4f}.")

    return FeatureSelectionResult(features=selected_features, mse_by_feature_count=scores,
                                  best_feature_count=best_n)

def select_features(X_train: pd.DataFrame, y_train: pd.Series, correlation_threshold: float = 0.40,
                    vif_threshold: float = 5.0,
                    estimator = None,
                    max_features: int | None = None) -> FeatureSelectionResult:

    """
    Perforn the complete feature-selection workflow.
    Feature selection is performed using training data only.

    The workflow is:
        1. Correlation Filtering
        2. VIF filtering
        3. RFE with inner cross-validation
    """

    if estimator is None:
        estimator = LinearRegression()

    # -------------------------
    # 1. Correlation filtering
    # -------------------------
    correlation_features = select_uncorrelated_features(X_train, threshold=correlation_threshold)

    X_filtered = X_train[correlation_features]

    max_features = min(max_features or X_filtered.shape[1], X_filtered.shape[1])
    if not correlation_features:
        raise ValueError("Correlation filtering removed all features")

    # -------------------------
    # 2. VIF filtering
    # -------------------------

    vif_features = selected_features_by_vif(X_filtered, threshold=vif_threshold)
    X_filtered = X_filtered[vif_features]

    max_features = min(max_features or X_filtered.shape[1], X_filtered.shape[1])


    if not vif_features:
        raise ValueError("VIF filtering removed all features")


    # -------------------------
    # 3. RFE
    # -------------------------

    result = selected_features_with_inner_cv(X_filtered, y_train, estimator=estimator, max_features=max_features)

    return result





