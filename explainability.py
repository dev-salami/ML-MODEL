"""
Explainability plots for v2 models:
  - plot_pdp:            2x3 grid of partial dependence curves (one per feature)
  - plot_shap_beeswarm:  SHAP beeswarm summary plot
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.inspection import partial_dependence
import shap

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# PDP — 2x3 grid (one subplot per feature)
# ---------------------------------------------------------------------------
def plot_pdp(model, model_key, model_name, X_test, feature_names, out_path):
    feature_names = list(feature_names)
    n = len(feature_names)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    for i, fname in enumerate(feature_names[:6]):
        ax = axes_flat[i]
        try:
            pd_result = partial_dependence(
                model, X_test, features=[i],
                grid_resolution=50, kind='average',
            )
            # sklearn ≥1.5 uses 'grid_values'; older releases use 'values'
            grid = pd_result.get('grid_values', pd_result.get('values'))
            xs = grid[0]
            ys = pd_result['average'][0]

            ax.plot(xs, ys, lw=2)
            ax.set_xlabel(fname, fontsize=11, fontweight='bold')
            ax.set_ylabel('Partial dependence', fontsize=10)
            ax.grid(True, alpha=0.3)

            # rug ticks of observed feature values along the bottom edge
            x_obs = X_test[:, i]
            ymin = ax.get_ylim()[0]
            ax.plot(x_obs, np.full_like(x_obs, ymin), '|',
                    markersize=8, alpha=0.4, color='black')
        except Exception as e:
            ax.text(0.5, 0.5, f'PDP failed:\n{e}',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_xlabel(fname, fontsize=11, fontweight='bold')

    # Hide unused axes if fewer than 6 features
    for j in range(min(n, 6), 6):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'{model_name}: Partial Dependence Plots', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  PDP:  {out_path}")


# ---------------------------------------------------------------------------
# SHAP beeswarm
# ---------------------------------------------------------------------------
def plot_shap_beeswarm(model, model_key, model_name,
                       X_train, X_test, feature_names, out_path,
                       max_test=100, bg_size=50):
    feature_names = list(feature_names)
    X_test_sub = X_test[:max_test]

    try:
        if model_key in ('rf', 'gb', 'bagging'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sub)
        elif model_key == 'm5p':
            # M5PModel wraps a DecisionTreeRegressor in .tree
            tree_model = getattr(model, 'tree', None)
            if tree_model is not None:
                explainer = shap.TreeExplainer(tree_model)
                shap_values = explainer.shap_values(X_test_sub)
            else:
                bg = shap.sample(X_train, bg_size) if len(X_train) > bg_size else X_train
                explainer = shap.Explainer(model.predict, bg)
                shap_values = explainer(X_test_sub).values
        else:
            # SVR / ANN / GEP / MARS / Hybrid — generic permutation-based explainer
            bg = shap.sample(X_train, bg_size) if len(X_train) > bg_size else X_train
            explainer = shap.Explainer(model.predict, bg)
            shap_values = explainer(X_test_sub).values

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_test_sub,
            feature_names=feature_names,
            plot_type='dot',     # beeswarm
            show=False,
        )
        plt.title(f'{model_name}: SHAP beeswarm', fontsize=13)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  SHAP: {out_path}")

    except Exception as e:
        print(f"  SHAP failed for {model_key}: {e}")
