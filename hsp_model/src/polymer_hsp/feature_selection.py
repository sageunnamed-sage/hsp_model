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
from sklearn.pipeline import check_is_fitted, clone
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import qr

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FeatureSelectionResult:
    """
    Result from recursive feature elimination.
    """
    features: list[str]
    mse_by_feature_count: dict[int, float]
    best_feature_count: int

############
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Remove redundant features using pairwise absolute correlation.
    """

    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CorrelationFilter requires a pandas DataFrame.")

        if not 0 < self.threshold <= 1:
            raise ValueError("Threshold must be in (0 and 1].")
        self.features_name_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self._validate_numeric_data(X)

        correlation = X.corr().abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
        self.features_to_drop_ = [column for column in upper.columns if (upper[column] > self.threshold).any()]
        self.features_to_keep_ = [column for column in X.columns if column not in self.features_to_drop_]

        if not self.features_to_keep_:
            raise ValueError("Correlation filtering removed all features.")

        logger.info("Correlation filtering retained %d of %d features.", len(self.features_to_keep_), X.shape[1])
        return self

    @staticmethod
    def _validate_numeric_data(X):

        if X.shape[1] == 0:
            raise ValueError("X contains no features.")

        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise ValueError("All features must be numeric.")

        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Input contains missing or infinite values." \
            "Handle these in a training-fitted preprocessing step.")

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "features_to_keep_")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("CorrelationFilter requires a pandas DataFrame.")

        missing = set(self.features_to_keep_) - set(X.columns)
        if missing:
            raise ValueError(f"Input features are missing the following columns: {missing}")

        return X.loc[:, self.features_to_keep_].copy()

    def get_feature_names_out(self, input_features=None):

        check_is_fitted(self, "features_to_keep_")
        return np.asarray(self.features_to_keep_, dtype=object)


    ###
class VIFFilter(BaseEstimator, TransformerMixin):
        """
        Iteratively remove features with high variance inflation factors.
        """

        def __init__(self, threshold: float = 5.0):
            self.threshold = threshold

        def fit(self, X: pd.DataFrame, y=None):
            if not isinstance(X, pd.DataFrame):
                raise TypeError("VIFFilter requires a pandas DataFrame.")

            if self.threshold <= 0 :
                raise ValueError("Threshold must be greater than 0.")

            CorrelationFilter._validate_numeric_data(X)

            self.features_name_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = X.shape[1]

            # Renive constant columns before calculating VIF
            variances = X.var()
            keep = variances[variances > 0].index.tolist()

            if not keep:
                raise ValueError("All features are constant. VIF cannot be calculated.")

            remaining = X.loc[:, keep].copy()
            removed = []

            # Remove linearly dependent features before calculating VIF.
            # Center and scale features so differences in descriptor units
            # do not dominate the rank calculation.
            values = remaining.to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    "VIFFilter received NaN or infinite values. "
                    "Clean or impute the input data before feature selection."
                )

            centered = values - values.mean(axis=0)
            scales = centered.std(axis=0)

            # Exclude columns whose standard deviation is zero or non-finite.
            valid_scale = np.isfinite(scales) & (scales > 0)

            remaining = remaining.loc[:, valid_scale].copy()
            centered = centered[:, valid_scale]
            scales = scales[valid_scale]

            if remaining.empty:
                raise ValueError(
                    "No variable features remain for VIF calculation."
                )

            standardized = centered / scales

            if not np.isfinite(standardized).all():
                raise ValueError(
                    "Standardisation produced NaN or infinite values in VIFFilter."
                )

            _, r, pivots = qr(
                standardized,
                mode="economic",
                pivoting=True,
            )

            diagonal = np.abs(np.diag(r))
            if diagonal.size:
                tolerance = (
                    max(standardized.shape)
                    * np.finfo(float).eps
                    * diagonal[0]
                )
                rank = int(np.sum(diagonal > tolerance))
            else:
                rank = 0

            # Keep the independent columns in their original order.
            independent_positions = sorted(pivots[:rank].tolist())
            independent_columns = remaining.columns[independent_positions].tolist()

            dependent_columns = [
                col for col in remaining.columns
                if col not in independent_columns
            ]

            removed.extend(dependent_columns)
            remaining = remaining.loc[:, independent_columns].copy()

            if remaining.empty:
                raise ValueError(
                    "No linearly independent features remain after rank checking."
                )

            logger.info(
                "Rank check removed %d linearly dependent features.",
                len(dependent_columns),
            )

            while remaining.shape[1] > 1:
                values = remaining.to_numpy(dtype=float)

                # Standardise each feature to improve numerical conditioning.
                means = values.mean(axis=0)
                scales = values.std(axis=0)

                valid_scale = np.isfinite(scales) & (scales > 0)

                if not valid_scale.all():
                    invalid_columns = remaining.columns[~valid_scale].tolist()
                    removed.extend(invalid_columns)
                    remaining = remaining.loc[:, valid_scale].copy()

                    if remaining.shape[1] <= 1:
                        break

                    continue

                standardized = (values - means) / scales

                # Detect linearly dependent columns.
                _, r, pivots = qr(
                    standardized,
                    mode="economic",
                    pivoting=True,
                )

                diagonal = np.abs(np.diag(r))
                tolerance = (
                    max(standardized.shape)
                    * np.finfo(float).eps
                    * diagonal[0]
                ) if diagonal.size else 0.0

                rank = int(np.sum(diagonal > tolerance))

                if rank < remaining.shape[1]:
                    independent_positions = sorted(pivots[:rank].tolist())
                    keep_columns = remaining.columns[independent_positions].tolist()
                    removed.extend(
                        col for col in remaining.columns if col not in keep_columns
                    )
                    remaining = remaining.loc[:, keep_columns].copy()
                    continue

                # Standardised descriptors plus intercept.
                design = np.column_stack([
                    np.ones(standardized.shape[0]),
                    standardized,
                ])

                vif_values = [
                    variance_inflation_factor(design, i + 1)
                    for i in range(standardized.shape[1])
                ]

                vif_series = pd.Series(
                    vif_values,
                    index=remaining.columns,
                    dtype=float,
                )

                max_vif = vif_series.max()

                if not np.isfinite(max_vif) or max_vif > self.threshold:
                    feature_to_remove = vif_series.idxmax()
                    removed.append(feature_to_remove)
                    remaining = remaining.drop(columns=[feature_to_remove])
                else:
                    break

            self.features_to_keep_ = remaining.columns.tolist()
            self.features_to_remove_ = [col for col in X.columns if col not in self.features_to_keep_]

            if not self.features_to_keep_:
                raise ValueError("VIF filtering removed all features.")

            logger.info(
                "VIF filtering retained %d of %d input features.",
                len(self.features_to_keep_),
                X.shape[1],
            )
            return self

        def transform(self, X: pd.DataFrame):

            check_is_fitted(self, "features_to_keep_")

            if not isinstance(X, pd.DataFrame):
                raise TypeError("VIFFilter requires a pandas DataFrame.")

            missing = set(self.features_to_keep_) - set(X.columns)
            if missing:
                raise ValueError(f"Input is missing fitted features: {missing}")

            return X.loc[:, self.features_to_keep_].copy()

        def get_feature_names_out(self, input_features=None):
            check_is_fitted(self, "features_to_keep_")
            
            return np.asarray(self.features_to_keep_, dtype=object)


class FeatureFilter(BaseEstimator, TransformerMixin):
    """
    Optional correlation filter followed by mandatory VIF filtering.
    """
    def __init__(self, correlation_threshold: float = 0.40, vif_threshold: float = 5.0, use_correlation_filter: bool = True):
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.use_correlation_filter = use_correlation_filter


    def fit(self, X: pd.DataFrame, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureFilter requires a pandas DataFrame.")

        self.features_name_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        if self.use_correlation_filter:
            self.correlation_filter_ = CorrelationFilter(threshold=self.correlation_threshold)
            X_filtered = self.correlation_filter_.fit_transform(X)
        else:
            X_filtered = X.copy()

        self.vif_filter_ = VIFFilter(threshold=self.vif_threshold)
        X_filtered = self.vif_filter_.fit_transform(X_filtered)

        self.features_to_keep_ = X_filtered.columns.tolist()

        if not self.features_to_keep_:
            raise ValueError("Feature filtering removed all features.")
        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "features_to_keep_")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureFilter requires a pandas DataFrame.")

        missing = set(self.features_to_keep_) - set(X.columns)
        if missing:
            raise ValueError(f"Input is missing fitted features: {missing}")

        if self.use_correlation_filter:
            X = self.correlation_filter_.transform(X)

        X = self.vif_filter_.transform(X)

        return X.loc[:, self.features_to_keep_].copy() 

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "features_to_keep_")
        return np.asarray(self.features_to_keep_, dtype=object)

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

def selected_features_with_inner_cv(X_train: pd.DataFrame, y_train: pd.Series, 
                                    estimator=None, max_features: int | None = None,
                                    correlation_threshold: float = 0.40,
                                    vif_threshold: float = 5.0,
                                    use_correlation_filter: bool = True) -> FeatureSelectionResult:
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

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, (pd.Series, pd.DataFrame)):
        raise TypeError("y_train must be a pandas Series or DataFrame.")

    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] != 1:
        raise ValueError("y_train must contain exactly one target column.")

    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError("X_train contains no features.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of samples.")

    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train indexes must match.")

    if len(X_train) < 3:
        raise ValueError("At least 3 samples are required.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes):
        raise ValueError("All features in X_train must be numeric.")

    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("X_train contains non-finite values (NaN or Inf).")

    y_values = np.asarray(y_train).reshape(-1)
    if not np.isfinite(y_values.astype(float)).all():
        raise ValueError("y_train contains non-finite values (NaN or Inf).")

    if estimator is None:
        estimator = LinearRegression()

    if max_features is None:
        max_features = X_train.shape[1]

    if not isinstance(max_features, (int, np.integer)) or max_features < 1:
        raise ValueError("max_features must be a positive integer.")

    max_features = min(max_features, X_train.shape[1])

    cv =LeaveOneOut()

    mse_by_feature_count: dict[int, float] = {}

    min_avaiable_features = X_train.shape[1]

    for inner_train_idx, _ in cv.split(X_train):

        X_inner_train = X_train.iloc[inner_train_idx]
        feature_filter = FeatureFilter(correlation_threshold=correlation_threshold, vif_threshold=vif_threshold,
                                        use_correlation_filter=use_correlation_filter)

        X_filtered = feature_filter.fit_transform(X_inner_train)

        min_avaiable_features = min(min_avaiable_features, X_filtered.shape[1])

    max_candidates = min(max_features, min_avaiable_features)

    if max_candidates < 1:
        raise ValueError("No feasible feature counts remain after filtering.")


    for n_features in range(1, max_candidates + 1):

        squared_errors = []

        for inner_train_idx, inner_val_idx in cv.split(X_train):

            X_inner_train = X_train.iloc[inner_train_idx]
            X_inner_val = X_train.iloc[inner_val_idx]

            y_inner_train = y_train.iloc[inner_train_idx]
            y_inner_val = y_train.iloc[inner_val_idx]

            # Fit all filters on the inner training data only.
            feature_filter = FeatureFilter(correlation_threshold=correlation_threshold, vif_threshold=vif_threshold, 
                                           use_correlation_filter=use_correlation_filter)

            values = X_inner_train.to_numpy(dtype=float)

            if not np.isfinite(values).all():
                bad_columns = X_inner_train.columns[
                    ~np.isfinite(values).all(axis=0)
                ].tolist()

                raise ValueError(
                    f"Invalid values in inner training fold. "
                    f"Columns: {bad_columns}"
                )
            X_filtered_train = feature_filter.fit_transform(X_inner_train)
            X_filtered_val = feature_filter.transform(X_inner_val)

            selector = RFE(estimator=clone(estimator), n_features_to_select=n_features, step=1)
            selector.fit(X_filtered_train, y_inner_train)

            selected_columns = X_filtered_train.columns[selector.support_]

            model = clone(estimator)
            model.fit(X_filtered_train.loc[:, selected_columns], y_inner_train)

            prediction = model.predict(X_filtered_val.loc[:, selected_columns])[0]

            squared_errors.append((float(y_inner_val.iloc[0]) - float(prediction)) ** 2)

        mse_by_feature_count[n_features] = float(np.mean(squared_errors))

    best_n = min(mse_by_feature_count, key=mse_by_feature_count.get)

    # Refit filters and RFE on the complete outer training fold.

    final_filter = FeatureFilter(correlation_threshold=correlation_threshold, vif_threshold=vif_threshold,
                                 use_correlation_filter=use_correlation_filter)

    X_filtered = final_filter.fit_transform(X_train)

    final_n = min(best_n, X_filtered.shape[1])

    final_selector = RFE(estimator=clone(estimator), n_features_to_select=final_n, step=1)
    final_selector.fit(X_filtered, y_train)
    selected_features = X_filtered.columns[final_selector.support_].tolist()

    logger.info(f"Selected {final_n} features using inner CV: inner-CV MSE = {mse_by_feature_count[best_n]:.4f}")
    
    return FeatureSelectionResult(features=selected_features, mse_by_feature_count=mse_by_feature_count, best_feature_count=len(selected_features))

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


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    correlation_threshold: float = 0.40,
    vif_threshold: float = 5.0,
    estimator=None,
    max_features: int | None = None,
    use_correlation_filter: bool = True) -> FeatureSelectionResult:

    """Select features using optional correlation filtering, mandatory VIF,
    and RFE with inner cross-validation.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError("X_train contains no features.")

    if y_train.empty:
        raise ValueError("y_train contains no HSP parameter.")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have number of observations.")

    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train must have the same index.")

    if max_features is not None and max_features < 1:
        raise ValueError("max_features must be at least 1.")

    if not 0 < vif_threshold:
        raise ValueError("vif_threshold must be greater than 0.")

    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1].")

    X = X_train.select_dtypes(include="number").copy()

    if X.empty:
        raise ValueError("No numeric features are available.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X_train contains missing or infinite values.")

    if not np.isfinite(y_train.to_numpy(dtype=float)).all():
        raise ValueError("y_train contains missing or infinite targets.")

    result = selected_features_with_inner_cv(
        X_train=X,
        y_train=y_train,
        estimator=estimator,
        max_features=max_features,
        correlation_threshold=correlation_threshold,
        vif_threshold=vif_threshold,
        use_correlation_filter=use_correlation_filter)

    logger.info("Feature selection completed: %d features selected.", len(result.features))

    return result
