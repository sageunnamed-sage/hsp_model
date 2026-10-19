"""Visualisation utilities for polymer HSP regression experiments."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polymer_hsp.evaluation import ExperimentResult

logger = logging.getLogger(__name__)


def plot_predicted_vs_actual(result: ExperimentResult, save_path: str | Path | None = None, show: bool = False) -> None:
    """Plot LOOCV predictions against measured target values."""

    y_true = np.asarray(result.y_true, dtype=float)
    y_pred = np.asarray(result.y_pred, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional.")

    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("Actual and predicted values must be non-empty and equal in length.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Actual and predicted values must be finite.")

    lower = float(min(y_true.min(), y_pred.min()))
    upper = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.8, edgecolor="black")
    ax.plot([lower, upper], [lower, upper], linestyle="--", label="Ideal prediction")

    ax.set_xlabel("Actual HSP value")
    ax.set_ylabel("Predicted HSP value")
    ax.set_title("LOOCV: Predicted vs. Actual")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_residuals(result: ExperimentResult, save_path: str | Path | None = None, show: bool = False) -> None:
    """Plot prediction residuals against predicted values."""

    y_true = np.asarray(result.y_true, dtype=float)
    y_pred = np.asarray(result.y_pred, dtype=float)

    if (y_true.ndim != 1 or y_pred.ndim != 1 or y_true.size == 0 or y_true.size != y_pred.size):
        raise ValueError("Actual and predicted values must be non-empty, one-dimensional, and equal in length.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Actual and predicted values must be finite.")

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(y_pred, residuals, alpha=0.8, edgecolor="black")
    ax.axhline(0, linestyle="--")

    ax.set_xlabel("Predicted HSP value")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("LOOCV Residual Plot")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def plot_feature_selection_frequency(result: ExperimentResult, top_n: int | None = 20, save_path: str | Path | None = None, show: bool = False) -> None:
    """Plot how often each feature was selected across LOOCV folds."""

    if not result.selected_features:
        raise ValueError("No feature-selection results are available.")

    if top_n is not None and (not isinstance(top_n, (int, np.integer)) or isinstance(top_n, (bool, np.bool_)) or top_n < 1):
        raise ValueError("top_n must be a positive integer or None.")

    counts: dict[str, int] = {}

    for fold_features in result.selected_features:
        for feature in set(fold_features):
            counts[feature] = counts.get(feature, 0) + 1

    if not counts:
        raise ValueError("No selected features were recorded.")

    frequencies = pd.Series(counts, dtype=int).sort_values(ascending=True)

    if top_n is not None:
        frequencies = frequencies.tail(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(frequencies))))

    frequencies.plot(kind="barh", ax=ax)

    ax.set_xlabel("Number of LOOCV folds selecting feature")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Selection Frequency")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    _finish_figure(fig, save_path, show)


def _finish_figure(fig: plt.Figure, save_path: str | Path | None, show: bool) -> None:
    """Save, display, or close a completed figure."""

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved figure to %s", path)

    if show:
        plt.show()
    else:
        plt.close(fig)
