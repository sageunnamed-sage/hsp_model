import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.polymer_hsp.modelling import run_random_forest_loocv, create_random_forest, RandomForestConfig, RandomForestResult


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

def test_run_random_forest_loocv():

    X, y = make_test_data()

    config = RandomForestConfig(max_features=3, random_state=30)

    result = run_random_forest_loocv(X, y, config=config)

    assert isinstance(result, RandomForestResult)
    assert len(result.y_true) == len(X)
    assert len(result.y_pred) == len(X)
    assert len(result.selected_features) == len(X)
    assert np.isfinite(result.y_pred).all()
    assert len(result.sample_indices) == len(X)
    np.testing.assert_array_equal(result.sample_indices, X.index.to_numpy())

def test_create_random_forest():
    model = create_random_forest(random_state=30)

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 30

def test_run_random_forest_loocv_rejects_empty_featues():
    y = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="X contains no features."):
        run_random_forest_loocv(pd.DataFrame(), y)

def test_run_random_forest_loocv_rejects_mismatched_lengths():
    X, y = make_test_data()

    with pytest.raises(ValueError, match="X and y must contain the same number of observations."):
        run_random_forest_loocv(X, y.iloc[:-1])

def test_run_random_forest_loocv_rejects_non_numeric_features():
    X, _ = make_test_data()
    y = pd.Series(["a"] * len(X))

    with pytest.raises(ValueError, match="y must contain numeric solubility parameters."):
        run_random_forest_loocv(X, y)

def test_run_random_forest_loocv_rejects_invalid_max_features():
    X, y = make_test_data()

    with pytest.raises(ValueError, match="max_features must be at least 1."):
        run_random_forest_loocv(X, y, max_features=0)

def test_run_random_forest_loocv_is_reproducible():
    X, y = make_test_data()

    config = RandomForestConfig(max_features=3, random_state=30)

    result_1 = run_random_forest_loocv(X, y, config=config)
    result_2 = run_random_forest_loocv(X, y, config=config)

    np.testing.assert_array_equal(result_1.y_true, result_2.y_true)
    np.testing.assert_array_equal(result_1.y_pred, result_2.y_pred)
    np.testing.assert_array_equal(result_1.sample_indices, result_2.sample_indices)


def test_random_forest_config_defaults():
    config = RandomForestConfig()

    assert config.correlation_threshold == 0.40
    assert config.vif_threshold == 5.0
    assert config.max_features is None
    assert config.random_state == 30