"""
Run independent Random Forest experiments for the three HSP parameters
"""

import logging 
from pathlib import Path

import numpy as np
import pandas as pd

from polymer_hsp.modelling import RandomForestConfig
from polymer_hsp.pipeline import run_random_forest_experiment
from polymer_hsp.visualisation import plot_feature_selection_frequency, plot_predicted_vs_actual, plot_residuals

# Project configuration

WORKBOOK_PATH = Path("data/raw/increased_MW_data_all.xlsx")
OUTPUT_DIR = Path("outputs")

SHEET_CONFIG = {
    "DeltaD": {"target_column": "DeltaD", "label": "DeltaD"},
    "DeltaP": {"target_column": "DeltaP", "label": "DeltaP"},
    "DeltaH": {"target_column": "DeltaH", "label": "DeltaH"}}

RANDOM_FOREST_CONFIG = RandomForestConfig(
    correlation_threshold=0.40,
    vif_threshold=5.0,
    max_features=3,
    random_state=30,
    use_correlation_filter=True)

logger = logging.getLogger(__name__)

def run_experiment_for_sheet(workbook_path: Path, sheet_name: str, target_column: str, label: str, output_dir: Path, config: RandomForestConfig) -> dict:
    """Load one HSP sheet, run the experiment, and save its outputs."""

    logger.info("Starting experiment for %s", label)

    df = pd.read_excel(workbook_path, sheet_name=sheet_name)

    if df.empty:
        raise ValueError(f"Sheet '{sheet_name}' contains no data.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in sheet " f"'{sheet_name}'. Available columns: {df.columns.tolist()}")

    # Remove rows with missing targets; never impute the target.
    df = df.loc[df[target_column].notna()].copy()

    if df.empty:
        raise ValueError(f"Sheet '{sheet_name}' has no rows with a valid target.")

    
    if "filenames" not in df.columns:
        raise ValueError(
            f"Column 'filenames' not found in sheet '{sheet_name}'."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in "
            f"sheet '{sheet_name}'."
        )

    # Preserve polymer identifiers separately from model inputs.
    filenames = df["filenames"].copy()

    # Convert the target to numeric values.
    y = pd.to_numeric(df[target_column], errors="raise")

    # All columns except the identifier and target are descriptors.
    X = df.drop(columns=["filenames", target_column]).copy()

    # Require all descriptor columns to be numeric.
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric descriptor columns in '{sheet_name}': "
            f"{non_numeric}"
        )

    valid_rows = np.isfinite(y.to_numpy(dtype=float))
    valid_rows &= np.isfinite(X.to_numpy(dtype=float)).all(axis=1)

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()
    filenames = filenames.loc[valid_rows].copy()

    if len(X) < 3:
        raise ValueError(
            f"Sheet '{sheet_name}' has fewer than 3 valid observations."
        )

    if X.shape[1] == 0:
        raise ValueError(
            f"No descriptor columns found in '{sheet_name}'."
        )

    if X.empty:
        raise ValueError(f"No numeric descriptor columns found in sheet '{sheet_name}'.")

    valid_rows = np.isfinite(y.to_numpy(dtype=float))
    valid_rows &= np.isfinite(X.to_numpy(dtype=float)).all(axis=1)

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if len(X) < 3:
        raise ValueError(f"Sheet '{sheet_name}' has fewer than 3 valid observations.")

    if X.shape[1] == 0:
        raise ValueError(f"No numeric descriptors remain in sheet '{sheet_name}'.")

    logger.info("%s: using %d observations and %d descriptors", label, len(X), X.shape[1])

    # Run modelling and evaluation.
    result = run_random_forest_experiment(X=X, y=y, config=config)

    # Create a separate output directory for this parameter.
    parameter_dir = output_dir / label
    parameter_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions and original row indices.
    predictions = pd.DataFrame({"filename": filenames.to_numpy(), "sample_index": result.sample_indices, "actual": result.y_true, "predicted": result.y_pred, "residual": result.y_true - result.y_pred})
    predictions.to_csv(parameter_dir / "predictions.csv", index=False)

    # Save metrics.
    metrics = {"parameter": label, "n_samples": len(result.y_true), "mse": result.metrics.mse, "rmse": result.metrics.rmse, "r2": result.metrics.r2}

    # Include MAE if it is part of your RegressionMetrics class.
    if hasattr(result.metrics, "mae"):
        metrics["mae"] = result.metrics.mae

    pd.DataFrame([metrics]).to_csv(parameter_dir / "metrics.csv", index=False)

    # Save the selected descriptors for each LOOCV fold.
    feature_records = [{"sample_index": sample_index, "selected_features": ";".join(features)}
        for sample_index, features in zip(result.sample_indices, result.selected_features)]

    pd.DataFrame(feature_records).to_csv(parameter_dir / "selected_features.csv", index=False)

    # Generate plots from the completed experiment.
    plot_predicted_vs_actual(result, save_path=parameter_dir / "predicted_vs_actual.png")
    plot_residuals(result, save_path=parameter_dir / "residuals.png")
    plot_feature_selection_frequency(result, save_path=parameter_dir / "feature_selection_frequency.png")

    logger.info("%s complete: RMSE=%.4f, R2=%.4f", label, result.metrics.rmse, result.metrics.r2)

    return metrics


def main() -> None:
    """Run experiments for all configured HSP parameters."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not WORKBOOK_PATH.is_file():
        raise FileNotFoundError(f"Excel workbook not found: {WORKBOOK_PATH.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for sheet_name, settings in SHEET_CONFIG.items():
        metrics = run_experiment_for_sheet(workbook_path=WORKBOOK_PATH, sheet_name=sheet_name, target_column=settings["target_column"], label=settings["label"], output_dir=OUTPUT_DIR, config=RANDOM_FOREST_CONFIG)
        all_metrics.append(metrics)

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(OUTPUT_DIR / "hsp_summary.csv", index=False)

    logger.info("All HSP experiments completed.")
    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
