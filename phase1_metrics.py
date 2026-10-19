"""
Performance metrics and rank scoring used by the Phase 1 figures.

Metric conventions (y = actual, p = predicted):
  r     Pearson correlation coefficient        (higher better)
  R2    coefficient of determination           (higher better)
  RMSE  root mean squared error, MPa           (lower better)
  MAE   mean absolute error, MPa               (lower better)
  MAPE  mean absolute percentage error, %      (lower better)
  MBE   mean bias error = mean(p - y), MPa     (closer to 0 better)
  VAF   variance accounted for, %              (higher better)
  CPI   combined performance index             (lower better)

CPI definition: the source paper does not publish its formula, so we use a
transparent, normalised lower-is-better composite:
    CPI = mean( RMSE/mean(y),  1 - R2,  1 - VAF/100 )
Adjust `combined_performance_index` if your reference uses another definition.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def _clean(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    return y, p


def rmse(y, p):
    y, p = _clean(y, p)
    return float(np.sqrt(mean_squared_error(y, p)))


def mape(y, p):
    y, p = _clean(y, p)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100.0)


def mbe(y, p):
    y, p = _clean(y, p)
    return float(np.mean(p - y))


def vaf(y, p):
    y, p = _clean(y, p)
    denom = np.var(y)
    if denom == 0:
        return 0.0
    return float((1.0 - np.var(y - p) / denom) * 100.0)


def pearson_r(y, p):
    y, p = _clean(y, p)
    if np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    return float(np.corrcoef(y, p)[0, 1])


def combined_performance_index(y, p):
    y, p = _clean(y, p)
    r2 = r2_score(y, p)
    mean_y = np.mean(y)
    rrmse = rmse(y, p) / mean_y if mean_y != 0 else np.nan
    return float(np.mean([rrmse, 1.0 - r2, 1.0 - vaf(y, p) / 100.0]))


def all_metrics(y, p):
    """Return every Phase 1 metric for one (actual, predicted) pair."""
    y, p = _clean(y, p)
    return {
        'r':    pearson_r(y, p),
        'R2':   float(r2_score(y, p)),
        'RMSE': rmse(y, p),
        'MAE':  float(mean_absolute_error(y, p)),
        'MAPE': mape(y, p),
        'MBE':  mbe(y, p),
        'CPI':  combined_performance_index(y, p),
        'VAF':  vaf(y, p),
    }


# Direction each metric is ranked in: True = higher is better.
HIGHER_IS_BETTER = {
    'r': True, 'R2': True, 'VAF': True,
    'RMSE': False, 'MAE': False, 'MAPE': False, 'CPI': False,
    'MBE': None,   # ranked by absolute value (closer to 0 better)
}


def rank_models(metric_rows, metric_keys):
    """
    Rank models for each metric (best = 1), ties share the lower rank.

    metric_rows: {model_name: {metric_key: value}}
    metric_keys: ordered list of metric keys to rank on
    Returns {model_name: {metric_key: rank}} plus a 'total' key per model.
    """
    names = list(metric_rows.keys())
    ranks = {n: {} for n in names}

    for key in metric_keys:
        hib = HIGHER_IS_BETTER.get(key, False)
        if hib is None:
            vals = {n: abs(metric_rows[n][key]) for n in names}
            ordered = sorted(names, key=lambda n: vals[n])          # smaller |MBE| first
        elif hib:
            ordered = sorted(names, key=lambda n: -metric_rows[n][key])  # larger first
        else:
            ordered = sorted(names, key=lambda n: metric_rows[n][key])   # smaller first

        prev_val, prev_rank = None, 0
        for i, n in enumerate(ordered, start=1):
            cur = abs(metric_rows[n][key]) if hib is None else metric_rows[n][key]
            rank = prev_rank if (prev_val is not None and cur == prev_val) else i
            ranks[n][key] = rank
            prev_val, prev_rank = cur, rank

    for n in names:
        ranks[n]['total'] = sum(ranks[n][k] for k in metric_keys)
    return ranks
