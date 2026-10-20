# GEP fix — diagnosis and results (Phase 1)

## 1. Diagnosis (why GEP predicted a constant ~51.76)

Reproduced the old Phase 1 GEP exactly. Findings:

- **Evolved "best" expression was a pure constant** using no features:
  `sub(sub(sub(sub(sub(sub(9.551, -6.921), -6.921), -6.921), -6.921), -6.921), -7.602) = 51.76`.
  Raw-output variance ≈ 5e-29 (one unique value) → R² ≈ 0 on TR/VD/TS. The constant ≈ the
  training mean (53.7).
- It was **retrained each run** (not a reused v1 model), but collapsed every time.
- The `lr_corrector` was fit (slope 0.895) but **never applied** — `GEPModel.predict_single`
  only applies `final_corrector`, which the Phase 1 path leaves `None`. So predictions were
  the raw constant.
- **Root cause:** in `evaluate_gep_individual` (`gep_model.py`), invalid predictions were
  replaced with `np.mean(y)`. A constant ≈ mean therefore scored a *good* MSE — the fitness
  function **rewarded** the trivial constant. No zero-variance penalty, plus a weak search
  (pop 150, gen 20) and NaN-prone operators, drove evolution straight to it.

## 2. Fix

New module **`gep_phase1.py`** (v1 `gep_model.py` and all other models untouched; split,
seed 42, and scaler unchanged). The Phase 1 `GEPAdapter` now calls it.

- **Numerically-safe operators:** `exp(clip(x,-10,10))`, `log(|x|+eps)`, `sqrt(|x|)`,
  safe division (denominator floored at 1e-6). Vectorised over numpy.
- **No constant substitution:** non-finite predictions **penalise** the individual's fitness
  (1e12), they are not replaced by a constant.
- **Constant penalty:** if an expression's output variance on the training data is ~0, it gets
  the 1e12 penalty.
- **Least-squares linear scaling** `a·f(x)+b` fit on **training data only** is part of the
  fitness (standard GP practice — this is *not* the removed noise "corrector"; no noise, never
  fit on test).
- **Selection by validation R²:** an internal 80/20 fit/val split of the training set (seed 42)
  selects the expression; 3 restarts with seeds 42/43/44.
- **Scaled vs unscaled:** scaled inputs won (internal val R² 0.67 vs 0.56; TS 0.79 vs 0.75), so
  GEP uses the same scaled features as every other model.

## 3. Results

**Settings:** DEAP GP. Population 400, 120 generations, 3 restarts (seeds 42–44),
tournament size 3, one-point crossover (cxpb 0.7), uniform mutation (mutpb 0.3),
tree-height limit 12, init genHalfAndHalf depth 1–3.
**Function set:** add, sub, mul, neg, safe div, safe log, safe sqrt, safe exp, sin, cos.
**Terminals:** the 6 scaled features `[C, POFA, CA/FA, W/B, SP, Age]` + ephemeral constants
`U(-2, 2)`. **Linking:** single expression tree. **Output:** `a·f(x) + b` with
**a = 3.987, b = 16.575** (least squares on all training rows). Inputs **scaled** (z-score).

**Final expression** (cleaned names; `f(x)` before the `a·f+b` scaling):

```
f = cos(sqrt(SP)) + cos(Age) + (Age − 0.728) + C + cos(CA_FA)
    + sin(Age + POFA + C + (C − CA_FA)) + cos(CA_FA)
    + cos(sqrt(sin(CA_FA))) + cos(SP·1.854) + SP + C + POFA + POFA + CA_FA
    + sin(CA_FA·CA_FA) + cos(CA_FA) + (Age − 0.728) + C
    + C + (C − CA_FA) + cos(SP·1.854) + Age + SP + cos(SP·1.854)
    + cos(Age) + (W_B − Age) + cos(CA_FA) + (Age − 0.728)
    + POFA + sin(POFA + POFA + C + (C − CA_FA))
    + POFA + sqrt(CA_FA + POFA + C + CA_FA) + cos(CA_FA) + C − log(Age)
    + cos(CA_FA) − log(Age) + cos(CA_FA)
CS ≈ 3.987 · f + 16.575
```

**Metrics (scaled inputs):**

| Stage | r | R² | RMSE (MPa) | MAE (MPa) | MAPE (%) | MBE (MPa) | CPI | VAF (%) |
|---|---|---|---|---|---|---|---|---|
| TR | 0.9201 | 0.8467 | 10.948 | 8.458 | 19.24 | 0.000 | 0.1702 | 84.67 |
| VD | 0.9415 | 0.8858 | 11.119 | 8.439 | 26.85 | −0.106 | 0.1488 | 88.58 |
| TS | 0.9342 | 0.8657 | 11.220 | 8.775 | 17.66 | −1.720 | 0.1563 | 86.89 |

**10-fold CV R²** (same `KFold(n_splits=10, shuffle=True, random_state=42)`; CV uses the
reduced GEP budget pop 250 / gen 80 / 1 restart):

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| R² | 0.856 | 0.835 | 0.747 | 0.793 | 0.521 | 0.765 | 0.861 | 0.838 | 0.794 | 0.647 |

Median **0.794**, mean 0.766, min 0.521 — GEP is no longer excluded from the radar.

## 4. Figures regenerated

All figures in `out/v2/figures/` were regenerated with the fixed GEP: GEP TR/VD scatter
(`actual_vs_predicted/gep_trvd.png`), the test scatter grid, violin plots, Taylor diagrams,
the radar (GEP now included; only Hybrid excluded), and the rank table. The subtitle
"(literature rows supplied by user — verify against sources)" was removed from
`comparison/fig18_literature.png`.

## 5. Other six models unchanged

Only GEP code changed; split/seed/scaler/other models untouched. Test-set R² is identical to
before the fix: ANN 0.921, RF 0.977, GB 0.982, SVR 0.918, Bagging 0.977, M5P 0.818,
Hybrid 0.981. GEP went from ≈ 0 to **0.866**.
