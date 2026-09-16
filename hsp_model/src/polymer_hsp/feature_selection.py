"""
Feature Selection and multicollinearity utilities
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
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
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    if threshold > 1.0:
        raise ValueError("threshold must be <= 1")

    correlation = X_train.corr(numeric_only=True).abs()

    mask = np.triu(np.ones(correlation.shape), k=1).astype(bool)
    upper_triangle = correlation.where(mask)

    features_to_drop = [column for column in upper_triangle.columns
                        if any(upper_triangle[column] > threshold)]

    selected_features = [column for column in upper_triangle.columns
                         if column not in features_to_drop]

    logger.info(f"Correlatin filtering removed {len(features_to_drop)} features.")

    return selected_features

def selected_features_by_vif(X_train: pd.DataFrame, threshold: float = 5.0) -> list[str]:
    """
    Remove features with high variance inflation factors.
    VIF is calculated using only the training data
    :param X_train: Training features dataframe.
    :param threshold: Maximum acceptable VIF
    :return: Feature names retained after VIF filetering.
    """

    if threshold < 1:
        raise ValueError("Threshold must be at least 1.")

    X = X_train.copy()

    # Keep only numeric columns
    X = X.select_dtypes(include="number")

    if X.empty:
        raise ValueError("X_train contains no numeric features.")

    # Remove constant columns.
    nunique = X.nunique()
    constant_features = nunique[nunique <= 1].index.tolist()

    if constant_features:
        logger.info(f"Removing {len(constant_features)} features before VIF calculation.")

        X = X.drop(columns=constant_features)

    if X.empty:
        raise ValueError("No non-constant numeric features remain for VIF calculation.")

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
    The function receives only the training data from an outer cross-validation split.
    RFE is fitted independently within each inner fold through the pipeline.
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

    if max_features is None:
        max_features = X_train.shape[1]

    if max_features < 1:
        raise ValueError("max_features must be at least 1.")

    max_features = min(max_features, X_train.shape[1])

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
    selected_features = X_train.columns[final_selector.support_].tolist()

    logger.info(f"Selected {best_n} features using inner CV")

    return FeatureSelectionResult(features=selected_features, mse_by_feature_count=mse_by_feature_count, best_feature_count=best_n)

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a Pearson correlation matrix after removing empty rows/columns
    """

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError("No numeric features available for correlation calculation.")

    corr = numeric.corr()

    corr = corr.dropna(axis=0, how='all')
    corr = corr.dropna(axis=1, how='all')

    logger.info(f"Correlation matrix computed {corr.shape} x {corr.shape}")

    return corr

def correlated_feature_pairs(corr : pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """
    Reutrn feature pairs whose absolute correlation is above the threshold.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    pairs: list[tuple[str, str, float]] = []

    for i, col1 in enumerate(corr.columns):
        for col2 in corr.columns[i + 1:]:
            value = corr.loc[col1, col2]

            if pd.notna(value) and abs(value) >= threshold:
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

def select_features(X_train: pd.DataFrame, y_train: pd.Series, correlation_threshold: float = 0.40,
                    vif_threshold: float = 5.0,
                    estimator = None,
                    max_features: int | None = None) -> FeatureSelectionResult:

    """
    Perform the complete feature-selection workflow.
    Feature selection is performed using training data only.

    The workflow is:
        1. Correlation Filtering
        2. VIF filtering
        3. RFE with inner cross-validation
    """
    if X_train.empty:
        raise ValueError("X_train contains no features.")

    if y_train.empty:
        raise ValueError("y_train contains no HSP parameter.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have number of observations.")

    if max_features is not None and max_features < 1:
        raise ValueError("max_features must be at least 1.")

    if estimator is None:
        estimator = LinearRegression()

    # -------------------------
    # 1. Correlation filtering
    # -------------------------
    correlation_features = select_uncorrelated_features(X_train, threshold=correlation_threshold)

    if not correlation_features:
        raise ValueError("Correlation filtering removed all features")

    X_filtered = X_train[correlation_features]

    if not correlation_features:
        raise ValueError("Correlation filtering removed all features")

    # -------------------------
    # 2. VIF filtering
    # -------------------------

    vif_features = selected_features_by_vif(X_filtered, threshold=vif_threshold)

    if not vif_features:
        raise ValueError("VIF filtering removed all features")

    X_filtered = X_filtered[vif_features]

    # The maximum number of features must be based on the feature set that remains after filtering
    if max_features is None:
        effective_max_features = X_filtered.shape[1]
    else:
        effective_max_features = min(max_features, X_filtered.shape[1])

    # -------------------------
    # 3. RFE with inner CV
    # -------------------------

    result = selected_features_with_inner_cv(X_filtered, y_train, estimator=estimator, max_features=effective_max_features)

    return result





