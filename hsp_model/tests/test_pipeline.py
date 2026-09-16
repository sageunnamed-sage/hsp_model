import numpy as np
import pandas as pd

from src.polymer_hsp.evaluation import ExperimentResult
from src.polymer_hsp.modelling import RandomForestConfig
from src.polymer_hsp.pipeline import run_random_forest_experiment


def make_test_data():
    rng = np.random.RandomState(30)
    n_samples = 20

    x1 = rng.normal(size=n_samples)
    x2 = x1 * 0.95 + rng.normal(scale=0.05, size=n_samples)
    x3 = rng.normal(size=n_samples)
    x4 = rng.normal(size=n_samples)

    y = 2 * x1 + 0.5 * x3 + rng.normal(scale=0.1, size=n_samples)

    X = pd.DataFrame(
        {
            "descriptor_1": x1,
            "descriptor_2": x2,
            "descriptor_3": x3,
            "descriptor_4": x4,
        }
    )

    y = pd.Series(y, name="target")

    return X, y


def test_run_random_forest_experiment():
    X, y = make_test_data()

    config = RandomForestConfig(max_features=3, random_state=30)

    result = run_random_forest_experiment(X, y, config=config,
    )

    assert isinstance(result, ExperimentResult)
    assert len(result.y_true) == len(X)
    assert len(result.y_pred) == len(X)
    assert len(result.sample_indices) == len(X)
    assert len(result.selected_features) == len(X)
    assert np.isfinite(result.metrics.rmse)
    assert np.isfinite(result.metrics.r2)
