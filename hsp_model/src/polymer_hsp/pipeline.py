"""
High-level orchestration
"""

import logging
import pandas as pd

from src.polymer_hsp.evaluation import ExperimentResult, create_experiment_result

from src.polymer_hsp.modelling import RandomForestConfig, run_random_forest_loocv


logger = logging.getLogger(__name__)


def run_random_forest_experiment(X: pd.DataFrame, y: pd.Series, config: RandomForestConfig | None = None) -> ExperimentResult:
    """
    Run a complete Random Forest experiment.

    The pipeline coordinates model execution and evaluation while
    keeping those responsibilities in their respective modules.
    """

    model_result = run_random_forest_loocv(X=X, y=y, config=config)

    result = create_experiment_result(y_true=model_result.y_true, y_pred=model_result.y_pred,
                                      sample_indices=model_result.sample_indices,
                                      selected_features=model_result.selected_features)

    logger.info(f"Random Forest experiment completed with RMSE={result.metrics.rmse:.4f}.")

    return result
