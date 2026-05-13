"""
Paper-style diagnostic plots for v2 models:
  - Violin plot of actual vs each model's predicted CS values (test set)
  - Taylor diagram comparing SD / CC / RMSD of each model vs actual

Usage:
    python generate_diagnostics.py [--version v2] [--data pofa_concrete_data.csv]
"""

import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from data_preprocessing import load_and_explore_data, preprocess_data
from utils import get_output_dirs


MODEL_FILES = {
    'RF':      'rf_model_v2.pkl',
    'GB':      'gb_model_v2.pkl',
    'SVR':     'svr_model_v2.pkl',
    'Bagging': 'bagging_model_v2.pkl',
    'ANN':     'ann_model_v2.pkl',
    'GEP':     'gep_model_v2.pkl',
    'MARS':    'mars_model_v2.pkl',
    'M5P':     'm5p_model_v2.pkl',
    'Hybrid':  'hybrid_model_v2.pkl',
}


# ---------------------------------------------------------------------------
# Violin plot — custom paper-style (outer violin + colored inner IQR diamond)
# ---------------------------------------------------------------------------
def plot_violin(predictions, y_test, out_path):
    data = {'Actual': np.asarray(y_test, dtype=float)}
    for name, preds in predictions.items():
        data[name] = np.asarray(preds, dtype=float)
    order = list(data.keys())

    fig, ax = plt.subplots(figsize=(max(11, 1.4 * len(order)), 7))
    colors = sns.color_palette('Set2', n_colors=len(order))

    max_w = 0.40  # max half-width of a violin at its widest point

    # Shared global y range — KDE evaluated beyond data extremes tapers
    # to ~0 density, producing the paper's pointed tip tails.
    all_vals = np.concatenate([data[name] for name in order])
    y_pad = (all_vals.max() - all_vals.min()) * 0.15
    y_lo  = float(all_vals.min() - y_pad)
    y_hi  = float(all_vals.max() + y_pad)
    y_grid = np.linspace(y_lo, y_hi, 400)

    for i, name in enumerate(order):
        vals = data[name]
        q25, med, q75 = np.percentile(vals, [25, 50, 75])

        # KDE outline evaluated on shared global grid
        kde = gaussian_kde(vals)
        dens = kde(y_grid)
        dens_max = dens.max()
        dens_norm = dens / dens_max
        half_w = dens_norm * max_w

        # Outer violin (light shade) — tapers to pointed tips at y_lo / y_hi
        ax.fill_betweenx(y_grid, i - half_w, i + half_w,
                         facecolor=colors[i], alpha=0.45,
                         edgecolor='black', linewidth=0.8, zorder=2)

        # Inner IQR diamond — width follows KDE width at the median
        w_med = float(kde(med)[0] / dens_max * max_w)
        diamond = Polygon([
            (i,         q75),
            (i + w_med, med),
            (i,         q25),
            (i - w_med, med),
        ], closed=True,
           facecolor=colors[i], alpha=0.95,
           edgecolor='black', linewidth=1.0, zorder=3)
        ax.add_patch(diamond)

        # White median dot
        ax.scatter([i], [med], s=45, color='white',
                   edgecolors='black', linewidth=1.0, zorder=4)

        # Q75 / median / Q25 labels — to the right of the violin
        x_off = max_w + 0.05
        for y, txt, bold in [(q75, f'{q75:.2f}', False),
                             (med, f'{med:.2f}', True),
                             (q25, f'{q25:.2f}', False)]:
            ax.text(i + x_off, y, txt, ha='left', va='center',
                    fontsize=9,
                    fontweight='bold' if bold else 'normal',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='gray', alpha=0.85),
                    zorder=5)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=15)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title('Violin plot — Predicted vs Actual CS (Test set)', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('CS (MPa)', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Taylor diagram (Cartesian)
# ---------------------------------------------------------------------------
def _taylor_xy(sd, cc):
    cc_clipped = max(min(cc, 1.0), -1.0)
    angle = np.arccos(cc_clipped)
    return sd * np.cos(angle), sd * np.sin(angle)


def plot_taylor(predictions, y_test, out_path):
    sd_ref = float(np.std(y_test, ddof=1))

    stats = []
    for name, preds in predictions.items():
        sd = float(np.std(preds, ddof=1))
        cc = float(np.corrcoef(preds, y_test)[0, 1])
        stats.append((name, sd, cc))

    sd_max = 1.5 * max([sd_ref] + [s[1] for s in stats])

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_aspect('equal')

    # --- SD arcs (concentric quarter-circles from origin) -------------------
    sd_ticks = np.linspace(0, sd_max, 6)[1:]
    for r in sd_ticks:
        arc = Arc((0, 0), 2 * r, 2 * r, angle=0, theta1=0, theta2=90,
                  color='gray', linestyle='--', linewidth=0.7)
        ax.add_patch(arc)
        ax.text(r, 0, f'{r:.1f}', fontsize=8, color='gray',
                ha='center', va='top')

    # --- CC rays (radial lines from origin) ---------------------------------
    cc_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
    for cc in cc_ticks:
        angle = np.arccos(cc)
        x_end = sd_max * np.cos(angle)
        y_end = sd_max * np.sin(angle)
        ax.plot([0, x_end], [0, y_end], color='lightgray',
                linestyle=':', linewidth=0.7)
        ax.text(x_end * 1.02, y_end * 1.02, f'{cc:.2f}',
                fontsize=8, color='gray',
                ha='left' if cc < 0.99 else 'center',
                va='bottom' if cc < 0.99 else 'bottom')

    # --- RMSD contours (arcs centered at reference) -------------------------
    rmsd_ticks = np.linspace(0, sd_max, 5)[1:]
    for r in rmsd_ticks:
        arc = Arc((sd_ref, 0), 2 * r, 2 * r, angle=0, theta1=30, theta2=150,
                  color='green', linestyle='--', linewidth=0.6, alpha=0.6)
        ax.add_patch(arc)
        lbl_x = sd_ref + r * np.cos(np.radians(135))
        lbl_y = r * np.sin(np.radians(135))
        if 0 <= lbl_x <= sd_max and 0 <= lbl_y <= sd_max:
            ax.text(lbl_x, lbl_y, f'RMSD={r:.1f}', fontsize=7,
                    color='green', alpha=0.7, ha='center')

    # --- Reference point (Actual) -------------------------------------------
    ax.scatter([sd_ref], [0], s=120, c='red', marker='s',
               edgecolors='k', zorder=5, label='Actual (reference)')
    ax.text(sd_ref, -sd_max * 0.04, 'Actual', fontsize=10,
            color='red', ha='center', va='top', fontweight='bold')

    # --- Model points -------------------------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats)))
    for (name, sd, cc), color in zip(stats, colors):
        x, y = _taylor_xy(sd, cc)
        ax.scatter([x], [y], s=80, color=color, edgecolors='k', zorder=5,
                   label=f'{name} (r={cc:.3f})')
        ax.annotate(name, (x, y), textcoords='offset points',
                    xytext=(6, 6), fontsize=10, fontweight='bold')

    # --- Axes / labels ------------------------------------------------------
    ax.set_xlim(0, sd_max)
    ax.set_ylim(0, sd_max)
    ax.set_xlabel('Standard Deviation (MPa)', fontsize=12)
    ax.set_ylabel('Standard Deviation (MPa)', fontsize=12)
    ax.set_title('Taylor diagram — Test set predictions', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(sd_max * 0.5, sd_max * 0.95, 'Correlation Coefficient',
            fontsize=10, color='gray', ha='center', style='italic')

    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Generate violin plot + Taylor diagram for v2 model predictions'
    )
    p.add_argument('--version', type=str, default='v2',
                   help='Model version directory under out/ (default: v2)')
    p.add_argument('--data', type=str, default='pofa_concrete_data.csv',
                   help='CSV data file (default: pofa_concrete_data.csv)')
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading and preprocessing data...")
    df = load_and_explore_data(args.data)
    _, X_test, _, y_test, _, _ = preprocess_data(df)

    dirs = get_output_dirs(args.version)

    predictions = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(dirs['models'], fname)
        if not os.path.exists(path):
            print(f"  Skipping {name}: missing {path}")
            continue
        try:
            model = joblib.load(path)
            predictions[name] = np.asarray(model.predict(X_test))
        except Exception as e:
            print(f"  Failed to load/predict {name}: {e}")

    print(f"\nLoaded {len(predictions)} models for diagnostics.")
    if not predictions:
        print("No models available. Run run_v2.py first.")
        return

    os.makedirs(dirs['misc'], exist_ok=True)

    print("\nGenerating violin plot...")
    plot_violin(predictions, np.asarray(y_test),
                os.path.join(dirs['misc'], 'violin_plot.png'))

    print("Generating Taylor diagram...")
    plot_taylor(predictions, np.asarray(y_test),
                os.path.join(dirs['misc'], 'taylor_diagram.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
