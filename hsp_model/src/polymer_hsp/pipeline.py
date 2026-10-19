
"""High-level orchestration for polymer HSP experiments."""

import logging

import pandas as pd

from polymer_hsp.evaluation import ExperimentResult, create_experiment_result
from polymer_hsp.modelling import RandomForestConfig, run_random_forest_loocv,

logger = logging.getLogger(__name__)


def run_random_forest_experiment(X: pd.DataFrame, y: pd.Series, config: RandomForestConfig | None = None ) -> ExperimentResult:
    """Run a complete Random Forest regression experiment."""

    model_result = run_random_forest_loocv(X=X, y=y, config=config)

    result = create_experiment_result(y_true=model_result.y_true, y_pred=model_result.y_pred, sample_indices=model_result.sample_indices, selected_features=model_result.selected_features)

    logger.info("Random Forest experiment completed: "
                "RMSE=%.4f, MAE=%.4f, R²=%.4f", result.metrics.rmse, result.metrics.mae, result.metrics.r2)

    return result