# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A collection of regression models that predict the compressive strength (CS, in MPa) of
concrete containing Palm Oil Fuel Ash (POFA), trained on `pofa_concrete_data.csv`. There is
no package structure — every module sits at the repo root and is imported by top-level name.

## Setup & running

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt   # includes optuna, streamlit, shap
```

There is no test suite, linter, or build step. You "run" the project by invoking one of the
three entry points below. Each loads and preprocesses the CSV, then trains/tunes and writes
artifacts under `out/`.

```bash
# v2 — current pipeline (Optuna tuning, SHAP/PDP, diagnostics) → writes out/v2/
python run_v2.py --model all --trials 100          # all 9 models
python run_v2.py --model rf --trials 20            # one model, quick
python run_v2.py --model gb --trials 100 --target 0.85

# v1 — legacy pipeline (sklearn GridSearchCV) → writes out/v1/
python main_enhanced.py --model all                # linear, rf, gb, svr, hybrid, bagging, gep
python main_enhanced.py --predict --model rf --cement 400 --pofa 100 --ca_fa 1.3 --w_b 0.35 --sp 5.0 --age 28
python main_enhanced.py --model compare            # also: correlation

# Streamlit GUI — loads already-trained .pkl models (v2 first, falls back to v1)
streamlit run gui.py
```

`main.py` is an older, smaller variant of `main_enhanced.py` (5 models, no GEP/bagging); prefer
`main_enhanced.py`. `pofa_concrete_prediction.py` is a standalone exploratory script that predates
the modular split and duplicates much of it — don't extend it; change the modules instead.

## Architecture

There are **two parallel pipeline generations** sharing the same data layer:

- **v1 (legacy)** — `main.py` / `main_enhanced.py`. Each model lives in its own module
  (`random_forest.py`, `svr.py`, `gep_model.py`, …) exposing `train_<model>(...)` and
  `tune_<model>(...)` functions that tune with sklearn `GridSearchCV`. The main script hand-wires
  each model's train→tune→`joblib.dump` sequence. Outputs go to the fixed `out/v1/` paths in
  `utils.OUTPUT_DIRS`.

- **v2 (current)** — `run_v2.py` orchestrates, `tuner.py` does the work. All 9 models tune via a
  single `tune(model_key, …)` dispatcher backed by `_TUNE_MAP` (`rf, gb, svr, bagging, ann, gep,
  mars, m5p, hybrid`), each using **Optuna** with an early-stop callback once `--target` R² is hit.
  `run_v2.py` then saves `<key>_model_v2.pkl` and every chart. Outputs go to `out/v2/` via
  `utils.get_output_dirs('v2')`.

When adding a model to v2, implement `tune_<key>(X_train, y_train, X_test, y_test, feature_names,
n_trials, r2_target)` returning `(model, best_params, r2, y_pred, study)`, register it in
`_TUNE_MAP` + `ALL_MODELS`/`MODEL_NAMES` in `run_v2.py`, and add an `elif` branch in
`get_feature_importance` (tuner.py) so its importances resolve.

### Shared layers (used by both generations)

- `data_preprocessing.py` — `load_and_explore_data()` (reads CSV, writes exploratory plots) and
  `preprocess_data()` (80/20 split at `random_state=42`, `StandardScaler`). Returns
  `X_train, X_test, y_train, y_test, scaler, feature_names`.
- `utils.py` — `OUTPUT_DIRS` (v1 paths), `get_output_dirs(version)` (parametrised paths),
  `predict_strength()`, `evaluate_model()`, model save/load helpers.
- `explainability.py` — `plot_pdp`, `plot_shap_beeswarm` (v2).
- `generate_diagnostics.py` — `plot_violin`, `plot_taylor` (v2 cross-model diagnostics).
- `correlation_analysis.py`, `model_comparison.py` — v1 cross-model reporting.

### Custom (non-sklearn) model classes

These are hand-rolled estimators you must understand before touching their modules:
- `gep_model.py` — Gene Expression Programming via **DEAP**. `GEPModel` is a picklable wrapper
  around the evolved symbolic expression; protected operators (`protectedDiv`, `protectedLog`, …)
  guard against NaN/inf. GEP predictions can still be NaN — callers fall back to RF (see
  `main_enhanced.predict`).
- `m5p.py` — `M5PModel` (sklearn `BaseEstimator`/`RegressorMixin`) wrapping a `DecisionTreeRegressor`
  with linear models at the leaves.
- `mars.py` — MARS via optional `pyearth`. `run_v2.py` silently drops `mars` from `--model all`
  when `pyearth` isn't installed; `--model mars` forces it and errors cleanly.

## Data & feature gotcha

`pofa_concrete_data.csv` columns: `DataPoint, C, POFA, CA/FA, W/B, SP, Age, CS`. Target is `CS`;
`DataPoint` is an arbitrary row id, not a feature. `preprocess_data` drops both
(`X = df.drop(['CS', 'DataPoint'], axis=1)`), so current models/scaler use **6 features** in the
order `[C, POFA, CA/FA, W/B, SP, Age]` — this is the order any input vector must follow.

Historically `preprocess_data` dropped only `CS`, so the **legacy v1 models in `out/v1/` were fit
on 7 features** (`DataPoint` leading). `utils.predict_strength` branches on `scaler.n_features_in_`
to handle both: a 6-feature scaler (current v2) gets `[C..Age]`; a 7-feature scaler (legacy v1)
gets a neutral `DataPoint` value prepended so the real inputs still line up. If you retrain v1
(`main_enhanced.py`), it now also produces 6-feature models. Keep input-vector construction
consistent with how the scaler was fit, or predictions silently misalign.

## Outputs

Artifacts are versioned under `out/v1/` and `out/v2/`, each split into `models/`, `shap/`, `pdp/`,
`feature_importance/`, `actual_vs_predicted/`, `correlation/`, `optuna/`, `misc/`. `__pycache__/`,
loose `*.png` at the root, and `sample/` are committed; the `venv/` is gitignored.
