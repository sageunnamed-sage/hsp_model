from sklearn.pipeline import clone

import numpy as np
import pandas as pd
import pytest

from polymer_hsp.feature_selection import select_features, selected_features_by_vif, select_uncorrelated_features, \
    calculate_vif, correlation_matrix, correlated_feature_pairs, CorrelationFilter, VIFFilter


def make_test_data():
    rng = np.random.RandomState(30)

    n_samples = 20

    x1 = rng.normal(size=n_samples)
    x2 = x1 * 0.95 + rng.normal(scale=0.05, size=n_samples)
    x3 = rng.normal(size=n_samples)
    x4 = rng.normal(size=n_samples)

    y = 2 * x1 + 0.5 * x3 + rng.normal(scale=0.1, size=n_samples)

    X = pd.DataFrame({"descriptor_1": x1,
                      "descriptor_2": x2,
                      "descriptor_3": x3,
                      "descriptor_4": x4})

    y = pd.Series(y, name="target")

    return X, y

def test_select_uncorrelated_features():
    X, _ = make_test_data()

    selected = select_uncorrelated_features(X, threshold=0.40)

    assert "descriptor_1" in selected
    assert "descriptor_2" not in selected

def test_select_features_vif():
    X, _ = make_test_data()

    selected = selected_features_by_vif(X, threshold=5.0)

    assert isinstance(selected, list)
    assert len(selected) >= 1


def test_select_features():
    X, y = make_test_data()

    result = select_features(X, y, correlation_threshold=0.40, vif_threshold=5.0,
                             max_features=3, use_correlation_filter=True)

    assert isinstance(result.features, list)
    assert len(result.features) >= 1

    assert isinstance(result.mse_by_feature_count, dict)
    assert result.best_feature_count in result.mse_by_feature_count


def test_select_uncorrelated_features_rejects_invalid_threshold():
    X, _ = make_test_data()

    with pytest.raises(ValueError, match="threshold must be >= 0"):
        select_uncorrelated_features(X, threshold=-1)

    with pytest.raises(ValueError, match="threshold must be <= 1"):
        select_uncorrelated_features(X, threshold=1.1)

def test_select_uncorrelated_features_accepts_boundary_threshold():
    X, _ = make_test_data()

    assert isinstance(select_uncorrelated_features(X, threshold=0.0), list)
    assert isinstance(select_uncorrelated_features(X, threshold=1.0), list)

def test_select_features_by_vif_rejects_invalid_threshold():
    X, _ = make_test_data()

    with pytest.raises(ValueError, match="Threshold must be at least 1."):
        selected_features_by_vif(X, threshold=-1)

def test_select_features_by_vif_rejects_non_numeric_input():
    X = pd.DataFrame({"descriptor_1": ["a", "b", "c"], "descriptor_2": ["d", "e", "f"]})

    with pytest.raises(ValueError, match="X_train contains no numeric features."):
        selected_features_by_vif(X)

def test_calculate_vif_returns_expected_columns():
    X, _ = make_test_data()
    result = calculate_vif(X)

    assert list(result.columns) == ["feature", "vif"]
    assert len(result.vif) == X.shape[1]
    assert (result.vif >= 1).all()

def test_calcualte_vif_single_feature():
    X = pd.DataFrame({"descriptor_1": [1.0, 2.0, 3.0]})

    result = calculate_vif(X)

    assert result["feature"].tolist() == ["descriptor_1"]
    assert result["vif"].tolist() == [1.0]

def test_calcualte_vif_rejects_non_numeric_input():
    X = pd.DataFrame({"descriptor_1": ["a", "b", "c"],
                      "descriptor_2": ["d", "e", "f"]})
    with pytest.raises(ValueError, match="No numeric features available for VIF calculation."):
        calculate_vif(X)

def test_correlation_matrix():
    X, _ = make_test_data()

    result = correlation_matrix(X)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (X.shape[1], X.shape[1])
    assert list(result.index) == list(X.columns)
    assert list(result.columns) == list(X.columns)

def test_correlation_matrix_pairs():
    X, _ = make_test_data()
    corr = correlation_matrix(X)
    pairs = correlated_feature_pairs(corr, threshold=0.40)

    assert isinstance(pairs, list)
    assert len(pairs) >= 1
    for feature_1, feature_2, value in pairs:
        assert feature_1 in X.columns
        assert feature_2 in X.columns
        assert abs(value) >= 0.40

def test_correlated_feature_pairs_rejects_invalid_threshold():
    X, _ = make_test_data()
    corr = correlation_matrix(X)

    with pytest.raises(ValueError, match="Threshold must be between 0 and 1."):
         correlated_feature_pairs(corr, threshold=-0.1)
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1."):
         correlated_feature_pairs(corr, threshold=1.1)

def test_select_features_rejects_empty_features():
    X = pd.DataFrame()
    y = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="X_train contains no features."):
        select_features(X, y)

def test_select_features_rejects_empty_target():
    X, _ = make_test_data()
    y = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="y_train contains no HSP parameter."):
        select_features(X, y)

def test_select_features_rejects_mismatched_lengths():
    X, y = make_test_data()

    with pytest.raises(ValueError, match="X_train and y_train must have number of observations."):
        select_features(X, y.iloc[:-1])


def test_select_features_rejects_invalid_max_features():
    X, y = make_test_data()

    with pytest.raises(ValueError, match="max_features must be at least 1."):
        select_features(X, y, max_features=0)


def test_correlation_filter_removes_redundant_feature():
    X = pd.DataFrame({
        "a": [1., 2., 3., 4., 5.],
        "b": [2., 4., 6., 8., 10.],
        "c": [5., 1., 4., 2., 3.],
    })

    selector = CorrelationFilter(threshold=0.9)
    transformed = selector.fit_transform(X)

    assert "a" in transformed.columns
    assert "b" not in transformed.columns
    assert "c" in transformed.columns


def test_vif_filter_removes_constant_and_collinear_features():
    X = pd.DataFrame({
        "a": [1., 2., 3., 4., 5., 6.],
        "b": [2., 4., 6., 8., 10., 12.],
        "c": [1., 4., 2., 6., 3., 5.],
        "constant": [1., 1., 1., 1., 1., 1.],
    })

    selector = VIFFilter(threshold=5.0)
    transformed = selector.fit_transform(X)

    assert "constant" not in transformed.columns
    assert "a" in transformed.columns or "b" in transformed.columns
    assert transformed.shape[1] >= 1


def test_transformers_can_be_cloned():
    assert clone(CorrelationFilter(threshold=0.4)).threshold == 0.4
    assert clone(VIFFilter(threshold=5.0)).threshold == 5.0

