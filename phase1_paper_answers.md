# Phase 1 — answers for the paper

All values below are read directly from the Phase 1 code and data
(`generate_phase1_figures.py`, `phase1_metrics.py`, `pofa_concrete_data.csv`),
**not** from the v1 pipeline or the GUI's Optuna-tuned `*_model_v2.pkl` files.

> **Important scope note.** The Phase 1 hyperparameters are *fixed, sensible defaults*
> chosen in `make_estimator()` for reproducible figure generation — they are **not**
> Optuna-tuned, and they differ from the GUI/v2 models. Trained on the 70/15/15 split.
> Where a value is **not present in the codebase**, that is stated explicitly rather than
> guessed.

---

## A. From the codebase

### Q1. Why were 5 records dropped (407 → 402)? Which rows, and why?

**Answer: There is no such step in this codebase.** `pofa_concrete_data.csv` already
contains **402 rows**, with **0 duplicate rows** and **0 missing values** (verified with
pandas). No module removes records — `load_split_scale()` and `preprocess_data()` only
drop the `DataPoint` id and the `CS` target column, never any rows.

So the "407 → 402" reduction did **not** happen in code and cannot be attributed to
duplicates/missing/outliers from these files. Either the source CSV was already cleaned to
402 before it entered the repo, or the 407 figure comes from an external source. **This
needs to be confirmed from wherever the raw 407-record database originated — it is not
recoverable from this project.**

### Q2. ANN settings (Table 3)

Defined in `make_estimator('ANN', ...)` — **scikit-learn `MLPRegressor`** (not TensorFlow/Keras):

| Setting | Value |
|---|---|
| Library | `sklearn.neural_network.MLPRegressor` |
| Hidden layers / neurons | `(128, 64, 32)` — 3 hidden layers |
| Activation | `relu` |
| Solver / optimiser | `adam` |
| L2 penalty (`alpha`) | `0.001` |
| Initial learning rate | `0.003` |
| Max iterations (epochs) | `5000` |
| Early stopping | `True`, `n_iter_no_change=40` |
| `random_state` | `42` |

⚠️ **Conflict to resolve:** your project report says the ANN uses TensorFlow. The Phase 1
code that produced the figures uses **scikit-learn `MLPRegressor`**. The paper should say
scikit-learn, or the ANN should be re-run in TensorFlow if the report must stand.

### Q3. Random Forest settings

`make_estimator('RF', ...)` — `sklearn.ensemble.RandomForestRegressor`:

| Setting | Value |
|---|---|
| `n_estimators` | `300` |
| `max_depth` | `20` |
| `min_samples_split` | `2` (sklearn default — not set) |
| `min_samples_leaf` | `1` (sklearn default — not set) |
| `max_features` | `1.0` (sklearn default for regression — not set) |
| `random_state` | `42` |
| `n_jobs` | `-1` |

### Q4. Gradient Boosting settings

`make_estimator('GB', ...)` — `sklearn.ensemble.GradientBoostingRegressor`:

| Setting | Value |
|---|---|
| `n_estimators` | `300` |
| `learning_rate` | `0.05` |
| `max_depth` | `3` |
| `subsample` | `0.9` |
| `min_samples_split` | `2` (default — not set) |
| `min_samples_leaf` | `1` (default — not set) |
| `loss` | `squared_error` (default — not set) |
| `random_state` | `42` |

### Q5. SVR settings

`make_estimator('SVR', ...)` — `sklearn.svm.SVR`:

| Setting | Value |
|---|---|
| `kernel` | `rbf` |
| `C` | `100.0` |
| `gamma` | `scale` (= 1 / (n_features · X.var())) |
| `epsilon` | `0.1` (default — not set) |

### Q6. Bagging settings

`make_estimator('Bagging', ...)` → `create_bagging_regressor(...)` in
`bagging_algorithm.py` — `sklearn.ensemble.BaggingRegressor`:

| Setting | Value |
|---|---|
| Base estimator | `DecisionTreeRegressor(max_depth=None, random_state=42)` (unpruned CART) |
| `n_estimators` | `100` |
| `max_samples` | `1.0` (default — not set) |
| `max_features` | `1.0` (default — not set) |
| `bootstrap` | `True` (default — samples **with** replacement) |
| `random_state` | `42` |

### Q7. M5P settings and library

M5P is **hand-implemented** in `m5p.py` (class `M5PModel`). There is **no external M5P
package** — it is a custom scikit-learn-style estimator (`BaseEstimator`, `RegressorMixin`)
that grows a `DecisionTreeRegressor` and fits a `LinearRegression` model at each leaf
(a "model tree").

| Setting | Value |
|---|---|
| Implementation | custom `M5PModel` in `m5p.py` (CART tree + per-leaf linear models) |
| `max_depth` | `5` |
| `min_samples_split` | `10` |
| `min_samples_leaf` | `5` |
| Pruning | None beyond the depth/leaf-size limits above |
| Smoothing | **Not implemented** (classic M5P leaf-to-root smoothing is absent) |

> Caveat for the write-up: because it lacks pruning/smoothing, this is an M5P-*style* model
> tree, not a faithful Quinlan M5′ implementation.

---

## B. Checks (numbers already in the paper)

### Q8. CPI formula

From `combined_performance_index()` in `phase1_metrics.py`. It is a **normalised,
lower-is-better composite** (there is no published formula in the source paper, so this one
was defined for this project):

```
CPI = ( RRMSE + (1 − R²) + (1 − VAF/100) ) / 3

where  RRMSE = RMSE / mean(y_actual)
       VAF   = (1 − Var(y − ŷ) / Var(y)) × 100     [percent]
```

⚠️ This is a project-specific definition, **not** the "cumulative prediction error index"
from any cited paper. Describe it as defined above, or replace it with a standard CPI if the
paper needs to match a reference.

### Q9. Preprocessing

From `load_split_scale()`:

- **Scaler:** `StandardScaler` (z-score), **fit on the training subset only** (`.fit(X_tr)`),
  then applied to VD and TS.
- **Split:** **random**, not stratified. Done in two `train_test_split` calls, both with
  `random_state=42`: first 70/30 (train vs temp), then the 30% split 50/50 into VD/TS.
- **Exact sizes (402 rows):** **TR = 281, VD = 60, TS = 61** (≈ 69.9 / 14.9 / 15.2 %).
- Features used (order matters): `[C, POFA, CA/FA, W/B, SP, Age]`; `DataPoint` and `CS` dropped.

### Q10. Hybrid cross-validation numbers — confirmed

`KFold(n_splits=10, shuffle=True, random_state=42)`, scaler re-fit per fold on that fold's
training portion. Hybrid = `StackingRegressor(RF(200) + GB(200, lr=0.05), final=SVR(RBF, C=100), cv=5)`.

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| R² (SVR meta) | 0.36 | 0.70 | 0.67 | 0.12 | 0.66 | 0.41 | 0.49 | 0.51 | 0.48 | 0.49 |

- **SVR meta:** median **0.493**, mean 0.489 — **confirmed** (matches the quoted values).
- **LinearRegression meta** (same bases/folds): per-fold
  `[0.76, 0.62, 0.70, 0.67, 0.59, 0.75, 0.72, 0.72, 0.78, 0.73]`,
  median **0.722**, mean 0.703 — **confirmed** (0.49 → 0.72).

### Q11. Test-set metrics for all 7 models — full table

Trained on TR (281), evaluated on the held-out TS (61). (GEP and MARS are in the wider
pipeline but are **not** among the paper's 7 models.)

| Model | r | R² | RMSE (MPa) | MAE (MPa) | MAPE (%) | MBE (MPa) | CPI | VAF (%) |
|---|---|---|---|---|---|---|---|---|
| ANN | 0.9599 | 0.9210 | 8.608 | 6.653 | 14.39 | 0.295 | 0.1047 | 92.11 |
| RF | 0.9901 | 0.9773 | 4.616 | 3.267 | 9.12 | 0.597 | 0.0429 | 97.76 |
| GB | 0.9915 | 0.9823 | 4.074 | 3.114 | 7.37 | −0.153 | 0.0364 | 98.23 |
| SVR | 0.9626 | 0.9182 | 8.758 | 6.018 | 11.99 | −1.342 | 0.1068 | 92.01 |
| Bagging | 0.9899 | 0.9768 | 4.666 | 3.296 | 9.25 | 0.531 | 0.0436 | 97.71 |
| M5P | 0.9095 | 0.8183 | 13.051 | 5.913 | 21.81 | 1.990 | 0.1986 | 82.25 |
| Hybrid | 0.9906 | 0.9810 | 4.217 | 2.961 | 7.89 | 0.311 | 0.0381 | 98.11 |

Best test-set performer: **GB (R² 0.982)**, closely followed by **Hybrid (0.981)** and RF (0.977).

---

## C. Not from the codebase (cannot be answered from these files)

These require external sources, not this project:

- **Fig. 1 — methodology flowchart.** Not code-derived; can be drawn from the paper's steps.
- **Hasan et al. (2023), Table 1** — number of mixes, best model, R²/RMSE/MAE. External lookup.
- **Al-Mughanam et al. (2020), Table 1** — testing-stage R/R² and RMSE. External lookup.
- **Kellouche et al.** — journal, year, volume, pages. Not in repo; needs the PDF / citation record.

---

## Reproduce these numbers

```bash
cd /Users/salami/Desktop/ML-MODEL && source venv/bin/activate
python generate_phase1_figures.py          # figures to out/v2/figures/
```

Hyperparameters: `make_estimator()` in `generate_phase1_figures.py`.
Split/scaler: `load_split_scale()` in the same file.
Metrics (incl. CPI, VAF): `phase1_metrics.py`.
