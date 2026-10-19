"""
Generate the Phase 1 figures and tables (paper: Hamed, Elshaarawy & Alsaadawi,
"Stacked-based machine learning to predict the uniaxial compressive strength of
concrete materials", Computers and Structures 308 (2025) 107644), reproduced for
the POFA dataset and all eight trained models.

Outputs go to out/v2/figures/, organised into per-measurement subfolders
(like out/v2/shap, pdp, actual_vs_predicted):
  kde/                KDE distributions (Fig 2)
  correlation/        heatmap (Fig 3)
  hexbin/             hexbin input-vs-CS (Fig 4)
  actual_vs_predicted/  TR+VD scatter per model (Fig 8-10 + 5 others) and
                        the test-stage scatter grid (Fig 13)
  violin/             TR/VD violins (Fig 11) + test violin (Fig 14a)
  taylor/             TR/VD Taylor (Fig 12) + test Taylor (Fig 14b)
  radar/              10-fold CV radar (Fig 15)
  shap/               SHAP dot+bar (Fig 16)
  pdp/                partial dependence (Fig 17)
  comparison/         literature comparison (Fig 18)
  tables/             dataset stats (Table 2) + rank analysis (Table 4)
All eight models are covered in the per-model scatter folder.

Hand-drawn diagrams (Fig 1 flowchart, Fig 5/6/7 architectures) are out of scope.

Models are trained fresh on a 70/15/15 TR/VD/TS split (and 10-fold CV for Fig 15)
so every stage is consistent; this is independent of the Optuna-tuned models the
GUI loads from out/v2/models/.

Usage:
    python generate_phase1_figures.py [--data pofa_concrete_data.csv]
                                      [--folds 10] [--skip-cv]
                                      [--gep-pop 150] [--gep-gen 20]
"""

import argparse
import os
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde, skew, kurtosis

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              StackingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from phase1_metrics import all_metrics, rank_models
from utils import get_output_dirs
from explainability import plot_pdp

import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES = ['C', 'POFA', 'CA/FA', 'W/B', 'SP', 'Age']
TARGET = 'CS'

AXIS_LABELS = {
    'C':     'Cement (kg/m³)',
    'POFA':  'POFA (kg/m³)',
    'CA/FA': 'CA/FA ratio',
    'W/B':   'W/B ratio',
    'SP':    'Superplasticizer (kg/m³)',
    'Age':   'Age (days)',
    'CS':    'Compressive Strength (MPa)',
}

# Display order of the eight models and a stable colour per model.
MODEL_ORDER = ['ANN', 'RF', 'GB', 'SVR', 'Bagging', 'GEP', 'M5P', 'Hybrid']
MODEL_KEYS = {
    'ANN': 'ann', 'RF': 'rf', 'GB': 'gb', 'SVR': 'svr',
    'Bagging': 'bagging', 'GEP': 'gep', 'M5P': 'm5p', 'Hybrid': 'hybrid',
}
BASE_MODELS = ['ANN', 'RF', 'GB', 'SVR', 'Bagging']   # candidates for "best base model"
MODEL_COLORS = dict(zip(MODEL_ORDER, sns.color_palette('tab10', n_colors=len(MODEL_ORDER))))

METRIC_KEYS = ['r', 'R2', 'RMSE', 'MAE', 'MAPE', 'MBE', 'CPI', 'VAF']

# Literature rows for Fig 18 (supplied by the user; verify against sources before use).
# RMSE/MAE left as None where the source reports a normalised (non-MPa) value.
LITERATURE = [
    # (label, R2, RMSE_MPa, MAE_MPa)
    ('Kazemi et al. (2026) | ANN-BBO',   0.984, 3.167, 2.28),
    ('Ali et al. (2025) | XGB-LGBM',     0.976, None,  3.11),
    ('Ahmad et al. (2024) | ANN',        0.890, 4.40,  2.99),
    ('Kellouche et al. (2024) | ANN',    0.989, None,  None),
    ('Safiuddin et al. (2016) | ANN',    0.949, None,  1.74),
]


# ---------------------------------------------------------------------------
# GEP adapter — gives the GEP model a uniform fit/predict interface
# ---------------------------------------------------------------------------
class GEPAdapter:
    def __init__(self, feature_names, pop_size=150, n_gen=20):
        self.feature_names = list(feature_names)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.model = None

    def fit(self, X, y):
        import gep_model
        _, _, func = gep_model.train_gep(
            np.asarray(X), np.asarray(y), np.asarray(X), np.asarray(y),
            self.feature_names, pop_size=self.pop_size, n_gen=self.n_gen, plot=False,
        )
        self.model = func
        return self

    def predict(self, X):
        preds = np.asarray(self.model.predict(np.asarray(X)), dtype=float)
        return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)


def make_estimator(name, feature_names, gep_pop=150, gep_gen=20):
    """Return a fresh, unfitted estimator for the given model name."""
    if name == 'RF':
        return RandomForestRegressor(n_estimators=300, max_depth=20,
                                     random_state=42, n_jobs=-1)
    if name == 'GB':
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                         max_depth=3, subsample=0.9, random_state=42)
    if name == 'SVR':
        return SVR(C=100.0, gamma='scale', kernel='rbf')
    if name == 'Bagging':
        from bagging_algorithm import create_bagging_regressor
        return create_bagging_regressor(
            DecisionTreeRegressor(max_depth=None, random_state=42),
            n_estimators=100, random_state=42, n_jobs=-1)
    if name == 'ANN':
        return MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                            solver='adam', alpha=1e-3, learning_rate_init=3e-3,
                            max_iter=5000, early_stopping=True, n_iter_no_change=40,
                            random_state=42)
    if name == 'M5P':
        from m5p import M5PModel
        return M5PModel(max_depth=5, min_samples_split=10, min_samples_leaf=5)
    if name == 'Hybrid':
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                             random_state=42)),
        ]
        return StackingRegressor(estimators=estimators, final_estimator=SVR(C=100.0))
    if name == 'GEP':
        return GEPAdapter(feature_names, pop_size=gep_pop, n_gen=gep_gen)
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split_scale(data_file):
    df = pd.read_csv(data_file)
    X = df[FEATURES].values
    y = df[TARGET].values

    # 70 / 15 / 15 train / validation / test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_vd, X_ts, y_vd, y_ts = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    data = {
        'TR': (scaler.transform(X_tr), y_tr),
        'VD': (scaler.transform(X_vd), y_vd),
        'TS': (scaler.transform(X_ts), y_ts),
    }
    return df, data, scaler


def train_all(data, feature_names, gep_pop, gep_gen):
    """Fit each model on TR; return predictions for TR/VD/TS and the fitted model."""
    X_tr, y_tr = data['TR']
    results = {}
    for name in MODEL_ORDER:
        try:
            est = make_estimator(name, feature_names, gep_pop, gep_gen)
            est.fit(X_tr, y_tr)
            preds = {stage: np.asarray(est.predict(data[stage][0])) for stage in ('TR', 'VD', 'TS')}
            results[name] = {'model': est, 'pred': preds}
            r2_ts = all_metrics(data['TS'][1], preds['TS'])['R2']
            print(f"  trained {name:8}  TS R2 = {r2_ts:.3f}")
        except Exception as e:
            print(f"  FAILED {name}: {e}")
    return results


# ---------------------------------------------------------------------------
# Fig 2 — KDE plots
# ---------------------------------------------------------------------------
def fig_kde(df, out_path):
    cols = FEATURES + [TARGET]
    n = len(cols)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    palette = sns.color_palette('husl', n)

    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        vals = df[col].dropna().values.astype(float)
        kde = gaussian_kde(vals)
        pad = (vals.max() - vals.min()) * 0.12
        grid = np.linspace(vals.min() - pad, vals.max() + pad, 400)
        dens = kde(grid)
        ax.fill_between(grid, dens, color=palette[i], alpha=0.6, edgecolor='black', lw=1)
        ax.set_xlabel(AXIS_LABELS.get(col, col), fontsize=10, fontweight='bold')
        ax.set_ylabel('Density', fontsize=9)
        stats = (f"Mean: {np.mean(vals):.2f}\nStd Dev: {np.std(vals):.2f}\n"
                 f"Skewness: {skew(vals):.2f}\nKurtosis: {kurtosis(vals):.2f}")
        ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=8,
                va='top', ha='right', style='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))
        ax.grid(True, alpha=0.2)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('KDE plots of input and output variables', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig 3 — correlation heatmap
# ---------------------------------------------------------------------------
def fig_heatmap(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Heatmap correlation matrix for all studied parameters', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig 4 — hexbin input vs CS
# ---------------------------------------------------------------------------
def fig_hexbin(df, out_path):
    ncol = 3
    nrow = int(np.ceil(len(FEATURES) / ncol))
    cmaps = ['viridis', 'cividis', 'magma', 'plasma', 'inferno', 'turbo']
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).flatten()

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        hb = ax.hexbin(df[feat], df[TARGET], gridsize=22, cmap=cmaps[i % len(cmaps)],
                       mincnt=1)
        ax.set_xlabel(AXIS_LABELS[feat], fontsize=10)
        ax.set_ylabel('Compressive Strength (MPa)', fontsize=9)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Counts', fontsize=8)

    for j in range(len(FEATURES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Hexbin graphs: input variables vs CS of concrete', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Table 2 — dataset statistics (rendered as an image)
# ---------------------------------------------------------------------------
def table_stats(df, out_path):
    cols = FEATURES + [TARGET]
    stats = pd.DataFrame({
        'Mean':   df[cols].mean(),
        'Std Dev': df[cols].std(),
        'Minimum': df[cols].min(),
        'Median': df[cols].median(),
        'Maximum': df[cols].max(),
        'Count':  df[cols].count(),
    }).T
    stats = stats.round(2)

    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 2, 0.5 * len(stats) + 1.5))
    ax.axis('off')
    tbl = ax.table(cellText=stats.values,
                   rowLabels=stats.index,
                   colLabels=[AXIS_LABELS.get(c, c) for c in cols],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_text_props(fontweight='bold')
    ax.set_title('Table 2: Summary of the statistical information for the dataset',
                 fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Scatter TR + VD (Fig 8/9/10 and the other five models)
# ---------------------------------------------------------------------------
def _metric_text(tag, m):
    return (f"{tag}: r = {m['r']:.3f}, R² = {m['R2']:.3f}, RMSE = {m['RMSE']:.3f} MPa, "
            f"MAE = {m['MAE']:.3f} MPa,\n       MAPE = {m['MAPE']:.2f}%, "
            f"MBE = {m['MBE']:.3f} MPa, CPI = {m['CPI']:.3f}, VAF = {m['VAF']:.1f}")


def scatter_trvd(name, data, pred, out_path):
    y_tr, p_tr = data['TR'][1], pred['TR']
    y_vd, p_vd = data['VD'][1], pred['VD']
    m_tr = all_metrics(y_tr, p_tr)
    m_vd = all_metrics(y_vd, p_vd)

    allv = np.concatenate([y_tr, p_tr, y_vd, p_vd])
    lo, hi = float(allv.min()), float(allv.max())
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_tr, p_tr, marker='p', s=70, c='royalblue', edgecolors='k',
               lw=0.4, alpha=0.8, label=f'{name} (TR)')
    ax.scatter(y_vd, p_vd, marker='p', s=70, c='gold', edgecolors='k',
               lw=0.4, alpha=0.8, label=f'{name} (VD)')
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Equality line')
    ax.plot([lo, hi], [lo * 1.1, hi * 1.1], 'r--', lw=1.2, label='+10% Deviation')
    ax.plot([lo, hi], [lo * 0.9, hi * 0.9], 'g--', lw=1.2, label='-10% Deviation')

    box = _metric_text('Train', m_tr) + '\n' + _metric_text('Valid', m_vd)
    ax.text(0.03, 0.97, box, transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', alpha=0.9))

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_title(f'Actual vs Predicted CS — {name} (TR & VD)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Violin panel (reused for Fig 11 a/b and Fig 14 a)
# ---------------------------------------------------------------------------
def _violin_panel(ax, data_dict, title):
    order = list(data_dict.keys())
    colors = sns.color_palette('Set2', n_colors=len(order))
    max_w = 0.40
    all_vals = np.concatenate([data_dict[n] for n in order])
    pad = (all_vals.max() - all_vals.min()) * 0.15
    y_lo, y_hi = float(all_vals.min() - pad), float(all_vals.max() + pad)
    grid = np.linspace(y_lo, y_hi, 400)

    for i, name in enumerate(order):
        vals = np.asarray(data_dict[name], dtype=float)
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        # Degenerate (near-constant) predictions break gaussian_kde; draw a thin
        # marker at the median instead so the model still appears on the axis.
        if np.std(vals) < 1e-6:
            ax.plot([i, i], [vals.min() - 0.5, vals.max() + 0.5],
                    color=colors[i], lw=3, zorder=2)
            ax.scatter([i], [med], s=40, color='white', edgecolors='black', lw=1.0, zorder=4)
            ax.text(i + max_w + 0.05, med, f'{med:.2f}', ha='left', va='center',
                    fontsize=7.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='gray', alpha=0.85))
            continue
        kde = gaussian_kde(vals)
        dens = kde(grid)
        half = dens / dens.max() * max_w
        ax.fill_betweenx(grid, i - half, i + half, facecolor=colors[i], alpha=0.45,
                         edgecolor='black', linewidth=0.8, zorder=2)
        w_med = float(kde(med)[0] / dens.max() * max_w)
        ax.add_patch(Polygon([(i, q75), (i + w_med, med), (i, q25), (i - w_med, med)],
                             closed=True, facecolor=colors[i], alpha=0.95,
                             edgecolor='black', linewidth=1.0, zorder=3))
        ax.scatter([i], [med], s=40, color='white', edgecolors='black', lw=1.0, zorder=4)
        for y, txt, bold in [(q75, f'{q75:.2f}', False), (med, f'{med:.2f}', True),
                             (q25, f'{q25:.2f}', False)]:
            ax.text(i + max_w + 0.05, y, txt, ha='left', va='center', fontsize=7.5,
                    fontweight='bold' if bold else 'normal',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='gray', alpha=0.85),
                    zorder=5)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, fontsize=9)
    ax.set_xlim(-0.7, len(order) - 0.1)
    ax.set_ylim(y_lo, y_hi)
    ax.set_ylabel('CS (MPa)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)


def _stage_pred_dict(data, results, stage, actual_label):
    d = {actual_label: np.asarray(data[stage][1], dtype=float)}
    for name in MODEL_ORDER:
        if name in results:
            d[name] = np.asarray(results[name]['pred'][stage], dtype=float)
    return d


def fig_violin_trvd(data, results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    _violin_panel(axes[0], _stage_pred_dict(data, results, 'TR', 'Actual (TR)'),
                  '(a) Training stage')
    _violin_panel(axes[1], _stage_pred_dict(data, results, 'VD', 'Actual (VD)'),
                  '(b) Validation stage')
    fig.suptitle('Violin plots: predicted CS from the models (TR and VD stages)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Taylor panel (reused for Fig 12 a/b and Fig 14 b)
# ---------------------------------------------------------------------------
def _taylor_panel(ax, pred_dict, actual, title):
    actual = np.asarray(actual, dtype=float)
    sd_ref = float(np.std(actual, ddof=1))
    stats = []
    for name, preds in pred_dict.items():
        preds = np.asarray(preds, dtype=float)
        sd = float(np.std(preds, ddof=1))
        cc = float(np.corrcoef(preds, actual)[0, 1]) if sd > 0 else np.nan
        if not np.isfinite(cc):
            print(f"    Taylor: skipping {name} (degenerate predictions)")
            continue
        stats.append((name, sd, cc))
    sd_max = 1.5 * max([sd_ref] + [s[1] for s in stats])
    ax.set_aspect('equal')

    for r in np.linspace(0, sd_max, 6)[1:]:
        ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=0, theta2=90,
                         color='gray', ls='--', lw=0.7))
        ax.text(r, 0, f'{r:.1f}', fontsize=7, color='gray', ha='center', va='top')
    for cc in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]:
        ang = np.arccos(cc)
        ax.plot([0, sd_max * np.cos(ang)], [0, sd_max * np.sin(ang)],
                color='lightgray', ls=':', lw=0.7)
        ax.text(sd_max * np.cos(ang) * 1.02, sd_max * np.sin(ang) * 1.02,
                f'{cc:.2f}', fontsize=7, color='gray', ha='left', va='bottom')
    for r in np.linspace(0, sd_max, 5)[1:]:
        ax.add_patch(Arc((sd_ref, 0), 2 * r, 2 * r, theta1=30, theta2=150,
                         color='green', ls='--', lw=0.6, alpha=0.5))
    ax.scatter([sd_ref], [0], s=110, c='red', marker='s', edgecolors='k', zorder=5)
    ax.text(sd_ref, -sd_max * 0.04, 'Actual', fontsize=9, color='red',
            ha='center', va='top', fontweight='bold')

    for name, sd, cc in stats:
        ang = np.arccos(max(min(cc, 1.0), -1.0))
        x, y = sd * np.cos(ang), sd * np.sin(ang)
        col = MODEL_COLORS.get(name, 'gray')
        ax.scatter([x], [y], s=70, color=col, edgecolors='k', zorder=5,
                   label=f'{name} (r={cc:.3f})')
    ax.set_xlim(0, sd_max); ax.set_ylim(0, sd_max)
    ax.set_xlabel('Standard Deviation (MPa)', fontsize=10)
    ax.set_ylabel('Standard Deviation (MPa)', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)


def fig_taylor_trvd(data, results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    _taylor_panel(axes[0],
                  {n: results[n]['pred']['TR'] for n in MODEL_ORDER if n in results},
                  data['TR'][1], '(a) Training stage')
    _taylor_panel(axes[1],
                  {n: results[n]['pred']['VD'] for n in MODEL_ORDER if n in results},
                  data['VD'][1], '(b) Validation stage')
    fig.suptitle('Taylor diagrams: predicted CS from the models (TR and VD stages)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig 13 — test-stage scatter grid (all models)
# ---------------------------------------------------------------------------
def fig_test_scatter_grid(data, results, out_path):
    names = [n for n in MODEL_ORDER if n in results]
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    y_ts = np.asarray(data['TS'][1], dtype=float)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.3 * nrow))
    axes = np.atleast_1d(axes).flatten()

    for idx, name in enumerate(names):
        ax = axes[idx]
        p = np.asarray(results[name]['pred']['TS'], dtype=float)
        m = all_metrics(y_ts, p)
        allv = np.concatenate([y_ts, p])
        lo, hi = float(allv.min()), float(allv.max())
        pad = (hi - lo) * 0.05; lo, hi = lo - pad, hi + pad
        ax.scatter(y_ts, p, s=45, color=MODEL_COLORS[name], edgecolors='k', lw=0.4, alpha=0.8,
                   label=f'{name}_Test')
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2)
        ax.plot([lo, hi], [lo * 1.1, hi * 1.1], 'r--', lw=1.0)
        ax.plot([lo, hi], [lo * 0.9, hi * 0.9], 'g--', lw=1.0)
        ax.text(0.04, 0.96, f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.3f} MPa",
                transform=ax.transAxes, fontsize=9, va='top', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('Actual', fontsize=10); ax.set_ylabel('Predicted', fontsize=10)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.25)

    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Test-stage scatter plots (Actual vs Predicted CS)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_test_violin(data, results, out_path):
    fig, ax = plt.subplots(figsize=(13, 7))
    _violin_panel(ax, _stage_pred_dict(data, results, 'TS', 'Actual (TS)'),
                  'Test stage — Violin plot of predicted CS')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_test_taylor(data, results, out_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    _taylor_panel(ax, {n: results[n]['pred']['TS'] for n in MODEL_ORDER if n in results},
                  data['TS'][1], 'Test stage — Taylor diagram')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig 15 — radar across 10-fold CV
# ---------------------------------------------------------------------------
def fig_radar_cv(df, scaler, feature_names, out_path, n_folds=10, gep_pop=80, gep_gen=12):
    X = df[FEATURES].values
    y = df[TARGET].values
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    metrics_tracked = ['R2', 'RMSE', 'MAE', 'MAPE', 'MBE']
    # fold_scores[model][metric] = list over folds
    fold_scores = {n: {m: [] for m in metrics_tracked} for n in MODEL_ORDER}

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
        sc = StandardScaler().fit(X[tr_idx])
        Xtr, Xte = sc.transform(X[tr_idx]), sc.transform(X[te_idx])
        ytr, yte = y[tr_idx], y[te_idx]
        print(f"    CV fold {fold}/{n_folds}")
        for name in MODEL_ORDER:
            try:
                est = make_estimator(name, feature_names, gep_pop, gep_gen)
                est.fit(Xtr, ytr)
                m = all_metrics(yte, est.predict(Xte))
                for mk in metrics_tracked:
                    fold_scores[name][mk].append(m[mk])
            except Exception as e:
                for mk in metrics_tracked:
                    fold_scores[name][mk].append(np.nan)
                print(f"      {name} failed on fold {fold}: {e}")

    titles = {'R2': '(a) R²', 'RMSE': '(b) RMSE (MPa)', 'MAE': '(c) MAE (MPa)',
              'MAPE': '(d) MAPE (%)', 'MBE': '(e) MBE (MPa)'}
    angles = np.linspace(0, 2 * np.pi, n_folds, endpoint=False).tolist()
    angles += angles[:1]

    # Exclude degenerate models (e.g. GEP collapse) so one bad model doesn't
    # blow up every radial scale and flatten the rest to the centre.
    usable = []
    for name in MODEL_ORDER:
        r2s = np.array(fold_scores[name]['R2'], dtype=float)
        if np.all(np.isnan(r2s)):
            continue
        if np.nanmedian(r2s) < 0.5:
            print(f"    radar: excluding {name} (median CV R² = {np.nanmedian(r2s):.2f})")
            continue
        usable.append(name)

    def _limits(mk):
        # Percentile-based so a single catastrophic fold (M5P/Hybrid can spike)
        # does not set the scale and flatten the stable models.
        vals = np.concatenate([np.array(fold_scores[n][mk], dtype=float) for n in usable])
        vals = vals[np.isfinite(vals)]
        if mk == 'R2':
            return 0.6, 1.0                      # fixed, paper-like discriminating window
        hi = float(np.percentile(vals, 90)) * 1.15
        if mk == 'MBE':
            lo = float(np.percentile(vals, 10))
            return min(0.0, lo) * 1.15, hi
        return 0.0, hi

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    for ai, mk in enumerate(metrics_tracked):
        ax = axes[ai]
        lo, hi = _limits(mk)
        for name in usable:
            vals = np.array(fold_scores[name][mk], dtype=float)
            vals = np.nan_to_num(vals, nan=hi)
            series = np.clip(np.concatenate([vals, vals[:1]]), lo, hi)
            ax.plot(angles, series, lw=1.8, color=MODEL_COLORS[name], label=name)
        ax.set_ylim(lo, hi)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'Fold {i}' for i in range(1, n_folds + 1)], fontsize=8)
        ax.set_title(titles[mk], fontsize=12, pad=14)
    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', fontsize=11, title='Model')
    excluded = [n for n in MODEL_ORDER if n not in usable and
                not np.all(np.isnan(fold_scores[n]['R2']))]
    if excluded:
        axes[5].text(0.5, 0.05, 'Excluded (degenerate): ' + ', '.join(excluded),
                     ha='center', fontsize=9, style='italic', transform=axes[5].transAxes)

    fig.suptitle('Radar plots across 10-fold CV', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return fold_scores


# ---------------------------------------------------------------------------
# Fig 16 — SHAP dot + bar for the best base model
# ---------------------------------------------------------------------------
def fig_shap(best_name, model, data, feature_names, out_path):
    import shap
    key = MODEL_KEYS[best_name]
    X_tr = data['TR'][0]
    X_ts = data['TS'][0][:120]
    feature_names = list(feature_names)

    if key in ('rf', 'gb', 'bagging'):
        sv = shap.TreeExplainer(model).shap_values(X_ts)
    else:
        bg = shap.sample(X_tr, 50) if len(X_tr) > 50 else X_tr
        sv = shap.Explainer(model.predict, bg)(X_ts).values

    fig = plt.figure(figsize=(10, 11))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.sca(ax1)
    shap.summary_plot(sv, X_ts, feature_names=feature_names, plot_type='dot', show=False)
    ax1.set_title(f'(a) SHAP summary — {best_name}', fontsize=12)

    ax2 = fig.add_subplot(2, 1, 2)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)
    ax2.barh([feature_names[i] for i in order], mean_abs[order],
             color=sns.color_palette('viridis', len(feature_names)))
    ax2.set_xlabel('mean(|SHAP value|)', fontsize=10)
    ax2.set_title(f'(b) Feature importance — {best_name}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Fig 18 — comparison vs literature
# ---------------------------------------------------------------------------
def fig_literature(present_label, present_metrics, out_path):
    rows = [(f'Present study | {present_label}',
             present_metrics['R2'], present_metrics['RMSE'], present_metrics['MAE'])]
    rows += LITERATURE

    labels = [r[0] for r in rows]
    cols = ['R²', 'RMSE (MPa)', 'MAE (MPa)']
    values = [[r[1], r[2], r[3]] for r in rows]

    # Colour each error cell on a shared scale; R2 cells on their own scale.
    err_vals = [v for row in values for v in (row[1], row[2]) if v is not None]
    err_norm = plt.Normalize(min(err_vals), max(err_vals))
    r2_vals = [row[0] for row in values if row[0] is not None]
    r2_norm = plt.Normalize(min(r2_vals), max(r2_vals))

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(rows) + 1.5))
    ax.set_xlim(0, 3); ax.set_ylim(0, len(rows))
    ax.axis('off')

    for ri, (label, row) in enumerate(zip(labels, values)):
        y = len(rows) - ri - 1
        ax.text(-0.1, y + 0.5, label, ha='right', va='center', fontsize=9, fontweight='bold')
        for ci, val in enumerate(row):
            if val is None:
                color, txt = (0.85, 0.85, 0.85, 1.0), '--'
            elif ci == 0:
                color, txt = plt.cm.viridis(r2_norm(val)), f'{val:.3f}'
            else:
                color, txt = plt.cm.RdYlGn_r(err_norm(val)), f'{val:.3f}'
            ax.add_patch(plt.Rectangle((ci, y), 1, 1, facecolor=color, edgecolor='white'))
            # Pick readable text colour from the cell's luminance (Rec. 601).
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            ax.text(ci + 0.5, y + 0.5, txt, ha='center', va='center',
                    fontsize=10, color='black' if lum > 0.6 else 'white',
                    fontweight='bold')
    for ci, c in enumerate(cols):
        ax.text(ci + 0.5, -0.3, c, ha='center', va='top', fontsize=10, fontweight='bold')

    ax.set_title('Comparative analysis for prediction of the concrete CS\n'
                 '(literature rows supplied by user — verify against sources)',
                 fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Table 4 — rank analysis
# ---------------------------------------------------------------------------
def table_rank(data, results, out_path):
    rank_keys = ['R2', 'RMSE', 'MAE', 'MAPE', 'MBE', 'CPI', 'VAF']
    stage_ranks = {}
    for stage in ('TR', 'VD'):
        rows = {n: all_metrics(data[stage][1], results[n]['pred'][stage])
                for n in MODEL_ORDER if n in results}
        stage_ranks[stage] = rank_models(rows, rank_keys)

    overall = {n: stage_ranks['TR'][n]['total'] + stage_ranks['VD'][n]['total']
               for n in stage_ranks['TR']}
    overall_rank_order = sorted(overall, key=lambda n: overall[n])
    overall_rank = {n: i for i, n in enumerate(overall_rank_order, start=1)}

    header = ['Model', 'Stage'] + rank_keys + ['Total', 'Overall Rank']
    cell_text = []
    for n in [m for m in MODEL_ORDER if m in stage_ranks['TR']]:
        for si, stage in enumerate(('TR', 'VD')):
            r = stage_ranks[stage][n]
            row = ([n if si == 0 else '', stage]
                   + [r[k] for k in rank_keys]
                   + [r['total'], overall_rank[n] if si == 0 else ''])
            cell_text.append(row)

    fig, ax = plt.subplots(figsize=(1.0 * len(header) + 2, 0.45 * len(cell_text) + 1.5))
    ax.axis('off')
    tbl = ax.table(cellText=cell_text, colLabels=header, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight='bold')
    ax.set_title('Table 4: Rank analysis for the developed models (lower is better)',
                 fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Generate Phase 1 figures for the POFA dataset')
    p.add_argument('--data', type=str, default='pofa_concrete_data.csv')
    p.add_argument('--folds', type=int, default=10)
    p.add_argument('--skip-cv', action='store_true', help='Skip Fig 15 (10-fold CV radar)')
    p.add_argument('--gep-pop', type=int, default=150)
    p.add_argument('--gep-gen', type=int, default=20)
    return p.parse_args()


def _run(label, fn):
    try:
        fn()
    except Exception as e:
        print(f"  {label} FAILED: {e}")
        traceback.print_exc()


# Each figure lives in a per-measurement subfolder (like out/v2/shap, pdp, ...).
def _fp(base, category, filename):
    d = os.path.join(base, category)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def main():
    args = parse_args()
    out = os.path.join(get_output_dirs('v2')['models'], '..', 'figures')
    out = os.path.normpath(out)
    os.makedirs(out, exist_ok=True)
    print(f"Output directory: {out}")

    print("\nLoading + splitting data (70/15/15)...")
    df, data, scaler = load_split_scale(args.data)
    feature_names = FEATURES

    print("\nTraining 8 models on TR split...")
    results = train_all(data, feature_names, args.gep_pop, args.gep_gen)

    # Best model (by test R2) overall and among base models
    ts_r2 = {n: all_metrics(data['TS'][1], results[n]['pred']['TS'])['R2'] for n in results}
    best_overall = max(ts_r2, key=ts_r2.get)
    base_available = [n for n in BASE_MODELS if n in results]
    best_base = max(base_available, key=lambda n: ts_r2[n]) if base_available else best_overall
    print(f"\nBest overall (TS R2): {best_overall} ({ts_r2[best_overall]:.3f}); "
          f"best base: {best_base} ({ts_r2[best_base]:.3f})")

    # ---- Data-derived figures ----
    print("\nData-derived figures:")
    _run('fig02 KDE',      lambda: fig_kde(df, _fp(out, 'kde', 'fig02_kde.png')))
    _run('fig03 heatmap',  lambda: fig_heatmap(df, _fp(out, 'correlation', 'fig03_heatmap.png')))
    _run('fig04 hexbin',   lambda: fig_hexbin(df, _fp(out, 'hexbin', 'fig04_hexbin.png')))
    _run('table02 stats',  lambda: table_stats(df, _fp(out, 'tables', 'table02_stats.png')))

    # ---- Per-model scatter (TR+VD) ----
    print("\nScatter (TR + VD):")
    fig_name = {'ANN': 'fig08', 'RF': 'fig09', 'GB': 'fig10'}
    for name in MODEL_ORDER:
        if name not in results:
            continue
        prefix = fig_name.get(name, '')
        fname = (f'{prefix}_{MODEL_KEYS[name]}_trvd.png' if prefix
                 else f'{MODEL_KEYS[name]}_trvd.png')
        _run(f'{fname} ({name})',
             lambda n=name, f=fname: scatter_trvd(n, data, results[n]['pred'],
                                                  _fp(out, 'actual_vs_predicted', f)))

    # ---- Comparison figures ----
    print("\nComparison figures:")
    _run('fig11 violin TR/VD',
         lambda: fig_violin_trvd(data, results, _fp(out, 'violin', 'fig11_trvd_violin.png')))
    _run('fig12 Taylor TR/VD',
         lambda: fig_taylor_trvd(data, results, _fp(out, 'taylor', 'fig12_trvd_taylor.png')))
    _run('fig13 test scatter',
         lambda: fig_test_scatter_grid(data, results,
                                       _fp(out, 'actual_vs_predicted', 'fig13_test_scatter_grid.png')))
    _run('fig14 test violin',
         lambda: fig_test_violin(data, results, _fp(out, 'violin', 'fig14_test_violin.png')))
    _run('fig14 test Taylor',
         lambda: fig_test_taylor(data, results, _fp(out, 'taylor', 'fig14_test_taylor.png')))

    # ---- Explainability ----
    print("\nExplainability:")
    _run(f'fig16 SHAP ({best_base})',
         lambda: fig_shap(best_base, results[best_base]['model'], data, feature_names,
                          _fp(out, 'shap', 'fig16_shap.png')))
    _run(f'fig17 PDP ({best_overall})',
         lambda: plot_pdp(results[best_overall]['model'], MODEL_KEYS[best_overall],
                          best_overall, data['TS'][0], feature_names,
                          _fp(out, 'pdp', 'fig17_pdp.png')))

    # ---- Literature comparison + rank table ----
    print("\nComparison table + ranks:")
    _run('fig18 literature',
         lambda: fig_literature(best_overall,
                                all_metrics(data['TS'][1], results[best_overall]['pred']['TS']),
                                _fp(out, 'comparison', 'fig18_literature.png')))
    _run('table04 rank',
         lambda: table_rank(data, results, _fp(out, 'tables', 'table04_rank.png')))

    # ---- 10-fold CV radar (slow; last) ----
    if not args.skip_cv:
        print("\nFig 15 — 10-fold CV radar (slow)...")
        _run('fig15 radar',
             lambda: fig_radar_cv(df, scaler, feature_names,
                                  _fp(out, 'radar', 'fig15_radar_cv.png'),
                                  n_folds=args.folds,
                                  gep_pop=max(60, args.gep_pop // 2),
                                  gep_gen=max(10, args.gep_gen // 2)))
    else:
        print("\nSkipping Fig 15 (--skip-cv).")

    print(f"\nDone. Figures in: {out}")


if __name__ == '__main__':
    main()
