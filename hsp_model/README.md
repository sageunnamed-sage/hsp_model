# Polymer Hansen Solubility Parameter Modelling

A modular Python machine-learning workflow for predicting polymer Hansen solubility parameters (HSPs): dispersion (ΔD), polar (ΔP), and hydrogen-bonding (ΔH) components.

This project demonstrates how a research-oriented modelling workflow can be developed into maintainable scientific software. It combines descriptor preprocessing, feature selection, Random Forest regression, cross-validation, automated testing, and evaluation in a structured Python package.

## Project objectives

- Develop a modular and testable machine-learning pipeline for polymer property prediction.
- Apply feature selection to high-dimensional polymer descriptor data.
- Evaluate predictive performance using leave-one-out cross-validation (LOOCV).
- Reduce information leakage by performing feature selection within outer training folds.
- Support reproducible experiments through explicit configuration and random seeds.
- Retain out-of-fold predictions and feature-selection results for further analysis.
- Document methodological assumptions, limitations, and opportunities for improvement.

## Technology stack

- **Python** — package development and workflow orchestration
- **pandas and NumPy** — data manipulation and numerical computation
- **scikit-learn** — Random Forest regression, recursive feature elimination, and cross-validation
- **SciPy and statsmodels** — numerical linear algebra and variance inflation factor (VIF) calculations
- **pytest** — automated testing
- **Matplotlib** — visualisation and model diagnostics

## Project structure

```text
polymer-hsp/
├── data/
│   └── raw/
├── outputs/
├── src/
│   └── polymer_hsp/
│       ├── __init__.py
│       ├── io.py
│       ├── preprocessing.py
│       ├── feature_selection.py
│       ├── modelling.py
│       ├── evaluation.py
│       ├── visualisation.py
│       └── pipeline.py
├── scripts/
├── tests/
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

The project follows a `src` package layout, separating reusable package code from the main entry point, supporting scripts, tests, input data, and generated outputs.

## Installation

Create and activate a virtual environment from the project root.

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

If development dependencies are defined in a `dev` optional dependency group in `pyproject.toml`, install them with:

```bash
python -m pip install -e ".[dev]"
```

## Running the workflow

From the project root, run:

```bash
python -m main
```

The main script runs the configured experiments for the three Hansen solubility parameter targets:

- **ΔD (`DeltaD`)** — dispersion component
- **ΔP (`DeltaP`)** — polar component
- **ΔH (`DeltaH`)** — hydrogen-bonding component

The workflow loads the input data, prepares the descriptor matrices, performs feature selection and model evaluation, and saves results to the configured output directory.

The input workbook must contain the expected target columns and polymer identifiers. File paths and experiment settings are defined by the project configuration.

## Modelling workflow

The project uses Random Forest regression and outer leave-one-out cross-validation to generate out-of-fold predictions.

```text
Polymer descriptor data
          │
          ▼
 Outer Leave-One-Out CV
          │
          ▼
   Outer training fold
          │
          ├── Correlation filtering
          │
          ├── VIF-based filtering
          │
          └── RFE with inner CV
          │
          ▼
    Fit Random Forest
          │
          ▼
 Predict held-out polymer
          │
          ▼
 Aggregate out-of-fold predictions
          │
          ▼
 Calculate regression metrics
```

In each outer fold, one polymer is held out for testing. Feature selection and model fitting are performed using the remaining observations, and the fitted model predicts the held-out polymer. This process repeats until each observation has an out-of-fold prediction.

### Feature selection

The feature-selection module implements:

- **Correlation filtering** to reduce redundancy among highly correlated descriptors.
- **Variance inflation factor (VIF) filtering** to identify potential multicollinearity among candidate descriptors.
- **Recursive feature elimination (RFE) with inner cross-validation** to compare candidate feature subsets.

Feature selection is performed within the outer training fold rather than once on the complete dataset before evaluation. This prevents the outer test observation from directly influencing feature selection for its fold.

**Methodological consideration:** In the current workflow, correlation and VIF filtering are applied within each outer training fold before the inner-CV/RFE stage. These filters are not necessarily refitted independently within every inner fold. Fully nesting all data-dependent feature-selection steps inside inner cross-validation would provide a stricter evaluation of the complete selection procedure and remains an area for improvement.

## Evaluation

Predictive performance is evaluated using out-of-fold predictions from outer LOOCV.

The evaluation module supports regression metrics including:

- **Mean squared error (MSE)** — average squared prediction error.
- **Root mean squared error (RMSE)** — prediction error expressed in the target's original units.
- **Mean absolute error (MAE)** — average absolute prediction error.
- **Coefficient of determination (R²)** — compares predictions against a mean-based reference.

Inner-CV scores are used during feature selection and should not be interpreted as the final estimate of model performance. The outer-LOOCV predictions provide the primary evaluation of the configured modelling workflow.

Given the small number of polymers, performance estimates should be interpreted cautiously. Results may be sensitive to individual observations and should not be treated as definitive evidence of generalisation to new polymer families.

## Reproducibility and configuration

The modelling workflow exposes configuration options through `RandomForestConfig`, including a random seed and feature-selection settings.

For example, the package can be called programmatically from another Python script after the input data have been prepared:

```python
from polymer_hsp.modelling import RandomForestConfig
from polymer_hsp.pipeline import run_random_forest_experiment

config = RandomForestConfig(
    random_state=30,
    max_features=3,
)

result = run_random_forest_experiment(
    X,
    y,
    config=config,
)

print(result.metrics)
```

Here, `X` is a descriptor DataFrame and `y` is the corresponding target series. This example illustrates package usage independently of the `python -m main` entry point; the exact configuration arguments should match the implementation.

An explicit `random_state` supports reproducibility of stochastic model operations. Reproducing an experiment also requires consistent input data, dependency versions, and configuration.

## Software design and testing

The package separates responsibilities into dedicated modules:

| Module | Responsibility |
|---|---|
| `io.py` | Data loading and input/output operations |
| `preprocessing.py` | Descriptor cleaning and preprocessing |
| `feature_selection.py` | Correlation filtering, VIF filtering, and RFE-based selection |
| `modelling.py` | Model configuration and cross-validation |
| `evaluation.py` | Regression metrics and prediction evaluation |
| `visualisation.py` | Prediction diagnostics and feature-selection visualisations |
| `pipeline.py` | High-level experiment orchestration |

Automated tests are maintained using `pytest`.

Run the test suite with:

```bash
python -m pytest -v
```

The test suite is intended to validate component behaviour, input handling, feature selection, modelling, evaluation, and pipeline execution. Test coverage should be interpreted according to the tests actually present in the repository.

## Limitations

Several limitations affect the interpretation of the results:

- **Small sample size:** Approximately 20 polymers provide limited evidence about generalisation to unseen materials.
- **High dimensionality:** The number of descriptors is large relative to the number of observations, increasing the risk of unstable feature selection and overfitting.
- **Feature-selection stability:** Different training folds may select different descriptors.
- **Cross-validation uncertainty:** LOOCV uses nearly all observations for training in each fold, but estimates can still be sensitive to individual samples.
- **Partial nesting:** Correlation and VIF filtering are not necessarily repeated independently inside every inner-CV split.
- **External validation:** Evaluation on independent polymer data would provide additional evidence of generalisation.

## Future development

Potential improvements include:

- Fully nesting all data-dependent feature-selection operations inside inner cross-validation.
- Assessing feature-selection stability across outer folds.
- Comparing Random Forest against simpler regression baselines.
- Evaluating alternative feature-selection strategies and model configurations.
- Expanding diagnostic visualisations and experiment reporting.
- Adding continuous integration and dependency-version constraints.
- Evaluating predictions on an independent dataset when suitable data become available.

## Summary

This project combines scientific machine learning with practical software engineering. It demonstrates modular Python development, descriptor-based regression, cross-validation, feature selection, reproducible configuration, automated testing, and transparent documentation of methodological limitations.

The goal is to provide a maintainable foundation for investigating polymer solubility prediction while making the assumptions and limitations of the available data explicit.
