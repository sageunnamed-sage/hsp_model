# Polymer Hansen Solubility Parameter Modelling

A machine-learning workflow for predicting polymer Hansen solubility parameters (HSPs): **dispersion (ΔD), polar (ΔP), and hydrogen-bonding (ΔH)** components.

The project was originally developed as a research notebook and has been refactored into a modular Python package with automated tests, explicit configuration, cross-validation, and separation of modelling and evaluation concerns.

## Project goals

The project demonstrates how a research machine-learning workflow can be converted into maintainable scientific software.

The main goals are:

* separate data processing, feature selection, modelling, and evaluation;
* make modelling assumptions explicit and configurable;
* avoid information leakage during cross-validation;
* provide deterministic experiments through explicit random seeds;
* test individual components independently;
* retain enough information to inspect feature-selection behaviour across folds.

## Project structure

```text
polymer-hsp/
├── src/
│   └── polymer_hsp/
│       ├── preprocessing.py
│       ├── feature_selection.py
│       ├── modelling.py
│       ├── evaluation.py
│       └── pipeline.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_modelling.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
│
├── pyproject.toml
└── README.md
```

## Installation

The project uses a `src` package layout and can be installed in editable mode:

```bash
python -m pip install -e ".[dev]"
```

## Running the tests

Run the complete test suite with:

```bash
pytest -v
```

The tests cover:

* preprocessing behaviour;
* correlation and VIF feature selection;
* RFE-based feature selection;
* model construction;
* nested leave-one-out cross-validation;
* input validation;
* reproducibility;
* regression metrics;
* end-to-end pipeline execution.

## Modelling workflow

The current refactored workflow is:

```text
Input descriptors
       │
       ▼
Outer Leave-One-Out CV
       │
       ├── Training fold
       │      │
       │      ├── pairwise correlation filtering
       │      ├── VIF filtering
       │      └── RFE + inner CV
       │
       ├── fit Random Forest
       │
       └── predict held-out sample
       │
       ▼
Predictions
       │
       ▼
MSE / RMSE / R²
```

Feature selection is performed using the training portion of each outer cross-validation fold rather than using the complete dataset before evaluation.

This reduces information leakage compared with performing feature selection once on the full dataset before cross-validation.

## Leakage considerations

The outer test observation is excluded from feature selection and model fitting.

The current implementation performs correlation and VIF filtering on each outer training fold and performs RFE using an inner cross-validation loop.

A future methodological refinement would be to place **all data-dependent feature-selection operations**, including correlation and VIF filtering, inside the inner cross-validation procedure. This would provide stricter nesting of the entire feature-selection workflow.

This is documented as a methodological limitation rather than being treated as solved simply because an outer cross-validation loop is present.

## Reproducibility

Random Forest experiments accept an explicit `random_state` through `RandomForestConfig`.

For example:

```python
from polymer_hsp.modelling import RandomForestConfig
from polymer_hsp.pipeline import run_random_forest_experiment

config = RandomForestConfig(random_state=30, max_features=3)

result = run_random_forest_experiment(X, y, config=config)

print(result.metrics)
```

The experiment result contains:

* observed values;
* predictions;
* original sample indices;
* selected features for each outer fold;
* regression metrics.

## Software design

The package separates responsibilities into small modules:

### `preprocessing.py`

Handles deterministic data-cleaning operations such as missing-value replacement and removal of empty descriptor columns.

### `feature_selection.py`

Contains correlation filtering, VIF calculation/filtering, and RFE-based feature selection.

### `modelling.py`

Creates and evaluates the Random Forest model using outer leave-one-out cross-validation.

### `evaluation.py`

Calculates regression metrics independently of the model implementation.

### `pipeline.py`

Provides a thin high-level interface that coordinates modelling and evaluation.

This separation means that the evaluation code can be reused with alternative regression models without coupling it to Random Forest.

## Limitations and future work

Several areas remain intentionally open for further development:

* fully nest every data-dependent feature-selection step inside inner CV;
* add HSP-specific configuration for descriptor exclusions;
* make Random Forest hyperparameters explicit and configurable;
* add data-loading and schema validation;
* add prediction-vs-observation visualisations;
* investigate feature-selection stability across CV folds;
* add continuous integration for automated testing;
* pin or constrain dependency versions for stronger environment reproducibility.

## Research software engineering considerations

The refactoring prioritises:

* reproducibility;
* separation of concerns;
* explicit configuration;
* automated testing;
* traceable scientific assumptions;
* leakage-aware evaluation;
* maintainability over notebook-specific convenience.

The intention is to make the scientific workflow easier to inspect, test, extend, and reproduce without obscuring methodological assumptions.
```
```
