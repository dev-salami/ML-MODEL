"""
v2 pipeline: Optuna fine-tuning + chart generation for all 9 models.

Usage:
    python run_v2.py [--model all|rf|gb|svr|bagging|ann|gep|mars|m5p|hybrid]
                     [--trials 100]
                     [--target 0.82]
                     [--data pofa_concrete_data.csv]
"""

import argparse
import os
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib

from data_preprocessing import load_and_explore_data, preprocess_data
from utils import get_output_dirs
from tuner import tune, get_feature_importance

ALL_MODELS = ['rf', 'gb', 'svr', 'bagging', 'ann', 'gep', 'mars', 'm5p', 'hybrid']

MODEL_NAMES = {
    'rf':      'Random Forest',
    'gb':      'Gradient Boosting',
    'svr':     'SVR',
    'bagging': 'Bagging',
    'ann':     'ANN',
    'gep':     'GEP',
    'mars':    'MARS',
    'm5p':     'M5P',
    'hybrid':  'Hybrid',
}


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def save_actual_vs_predicted(y_test, y_pred, model_key, dirs):
    from sklearn.metrics import r2_score
    os.makedirs(dirs['actual_vs_predicted'], exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k', s=60)

    lo = min(float(y_test.min()), float(y_pred.min()))
    hi = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect prediction')

    r2 = r2_score(y_test, y_pred)
    ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.92), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), fontsize=12)

    ax.set_title(f'{MODEL_NAMES[model_key]}: Actual vs Predicted Compressive Strength')
    ax.set_xlabel('Actual Strength (MPa)')
    ax.set_ylabel('Predicted Strength (MPa)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()

    out = os.path.join(dirs['actual_vs_predicted'], f'{model_key}_actual_vs_predicted.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart: {out}")


def save_feature_importance_chart(importances, feature_names, model_key, dirs):
    os.makedirs(dirs['feature_importance'], exist_ok=True)

    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_names, sorted_vals)
    ax.set_title(f'{MODEL_NAMES[model_key]}: Feature Importance')
    ax.set_xlabel('Importance')
    plt.tight_layout()

    out = os.path.join(dirs['feature_importance'], f'{model_key}_feature_importance.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart: {out}")


def save_optuna_history(study, model_key, dirs):
    os.makedirs(dirs['optuna'], exist_ok=True)

    try:
        import optuna.visualization.matplotlib as optuna_mpl
        ax = optuna_mpl.plot_optimization_history(study)
        fig = ax.figure
        ax.set_title(f'{MODEL_NAMES[model_key]}: Optuna Optimization History')
    except Exception:
        trial_values = [t.value for t in study.trials if t.value is not None]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(trial_values)), trial_values, 'b-o', markersize=4)
        ax.set_title(f'{MODEL_NAMES[model_key]}: Optuna Optimization History')
        ax.set_xlabel('Trial')
        ax.set_ylabel('R²')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    out = os.path.join(dirs['optuna'], f'optuna_{model_key}.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Chart: {out}")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(model_key, X_train, y_train, X_test, y_test, feature_names,
              dirs, n_trials, r2_target):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print(f"\n{'='*60}")
    print(f"Tuning: {MODEL_NAMES[model_key]}  (key={model_key})")
    print(f"{'='*60}")

    model, best_params, r2, y_pred, study = tune(
        model_key, X_train, y_train, X_test, y_test, feature_names,
        n_trials=n_trials, r2_target=r2_target
    )

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    target_met = r2 >= r2_target

    if target_met:
        print(f"  R² = {r2:.4f}  (target met)")
    else:
        print(f"  WARNING: R² = {r2:.4f}  (below target {r2_target})")

    # Save model
    os.makedirs(dirs['models'], exist_ok=True)
    model_path = os.path.join(dirs['models'], f'{model_key}_model_v2.pkl')
    joblib.dump(model, model_path)
    print(f"  Model: {model_path}")

    # Charts
    save_actual_vs_predicted(y_test, y_pred, model_key, dirs)

    importances = get_feature_importance(model_key, model, X_test, y_test, feature_names)
    save_feature_importance_chart(importances, feature_names, model_key, dirs)

    save_optuna_history(study, model_key, dirs)

    # ---- Explainability: PDP + SHAP beeswarm ----
    from explainability import plot_pdp, plot_shap_beeswarm

    try:
        os.makedirs(dirs['pdp'], exist_ok=True)
        pdp_path = os.path.join(dirs['pdp'], f'{model_key}_pdp.png')
        plot_pdp(model, model_key, MODEL_NAMES[model_key],
                 X_test, feature_names, pdp_path)
    except Exception as e:
        print(f"  PDP failed for {model_key}: {e}")

    try:
        os.makedirs(dirs['shap'], exist_ok=True)
        shap_path = os.path.join(dirs['shap'], f'{model_key}_shap.png')
        plot_shap_beeswarm(model, model_key, MODEL_NAMES[model_key],
                           X_train, X_test, feature_names, shap_path)
    except Exception as e:
        print(f"  SHAP failed for {model_key}: {e}")

    return {
        'model':       model_key,
        'name':        MODEL_NAMES[model_key],
        'r2':          r2,
        'rmse':        rmse,
        'mae':         mae,
        'target_met':  target_met,
        'best_params': best_params,
        'y_pred':      np.asarray(y_pred),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results, r2_target):
    print(f"\n{'='*68}")
    print(f"{'Model':<22} {'R²':>8} {'RMSE':>10} {'MAE':>10}  {'Target Met'}")
    print(f"{'-'*68}")
    for r in results:
        flag = 'YES' if r['target_met'] else 'NO  <-- below target'
        r2_s   = f"{r['r2']:.4f}"   if not np.isnan(r['r2'])   else 'ERROR'
        rmse_s = f"{r['rmse']:.4f}" if not np.isnan(r['rmse']) else 'ERROR'
        mae_s  = f"{r['mae']:.4f}"  if not np.isnan(r['mae'])  else 'ERROR'
        print(f"{r['name']:<22} {r2_s:>8} {rmse_s:>10} {mae_s:>10}  {flag}")
    print(f"{'='*68}")
    n_met = sum(1 for r in results if r['target_met'])
    print(f"Models meeting R² ≥ {r2_target}: {n_met}/{len(results)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='v2 Pipeline: Optuna fine-tuning for all POFA concrete models'
    )
    parser.add_argument('--model', type=str, default='all',
                        choices=['all'] + ALL_MODELS,
                        help='Model to tune (default: all)')
    parser.add_argument('--trials', type=int, default=100,
                        help='Max Optuna trials per model (default: 100)')
    parser.add_argument('--target', type=float, default=0.82,
                        help='R² target threshold (default: 0.82)')
    parser.add_argument('--data', type=str, default='pofa_concrete_data.csv',
                        help='Path to CSV data file')
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading and preprocessing data...")
    df = load_and_explore_data(args.data)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    feature_names = list(feature_names)

    dirs = get_output_dirs('v2')

    # Persist the scaler so the GUI (and any downstream tool) can transform new inputs.
    os.makedirs(dirs['models'], exist_ok=True)
    joblib.dump(scaler, os.path.join(dirs['models'], 'pofa_concrete_strength_scaler.pkl'))

    # Drop MARS from the default 'all' set if pyearth isn't installed.
    # Users can still explicitly opt-in via `--model mars` and get a clean error.
    pyearth_available = True
    try:
        import pyearth  # noqa: F401
    except ImportError:
        pyearth_available = False

    if args.model == 'all':
        models_to_run = [m for m in ALL_MODELS
                         if m != 'mars' or pyearth_available]
        if not pyearth_available and 'mars' in ALL_MODELS:
            print("Note: pyearth not installed — skipping MARS. "
                  "Install with `pip install sklearn-contrib-py-earth` "
                  "to enable it (or use --model mars to force).")
    else:
        models_to_run = [args.model]

    print(f"\nModels to tune: {models_to_run}")
    print(f"Max trials per model: {args.trials}")
    print(f"R² target: {args.target}")

    results = []
    for key in models_to_run:
        try:
            result = run_model(
                key, X_train, y_train, X_test, y_test, feature_names,
                dirs, n_trials=args.trials, r2_target=args.target
            )
            results.append(result)
        except ImportError as e:
            # Missing optional dep (e.g. pyearth) — clean one-liner, no traceback
            print(f"\nSkipping {key}: {e}")
            results.append({
                'model':      key,
                'name':       MODEL_NAMES.get(key, key),
                'r2':         float('nan'),
                'rmse':       float('nan'),
                'mae':        float('nan'),
                'target_met': False,
                'best_params': {},
                'y_pred':     None,
            })
        except Exception as e:
            print(f"\nERROR tuning {key}: {e}")
            traceback.print_exc()
            results.append({
                'model':      key,
                'name':       MODEL_NAMES.get(key, key),
                'r2':         float('nan'),
                'rmse':       float('nan'),
                'mae':        float('nan'),
                'target_met': False,
                'best_params': {},
                'y_pred':     None,
            })

    print_summary_table(results, args.target)

    # ---- Diagnostics: violin plot + Taylor diagram ----
    from generate_diagnostics import plot_violin, plot_taylor

    predictions = {
        r['name']: r['y_pred']
        for r in results
        if r.get('y_pred') is not None and not np.isnan(r['r2'])
    }
    if predictions:
        print("\nGenerating diagnostic plots...")
        os.makedirs(dirs['misc'], exist_ok=True)
        plot_violin(predictions, np.asarray(y_test),
                    os.path.join(dirs['misc'], 'violin_plot.png'))
        plot_taylor(predictions, np.asarray(y_test),
                    os.path.join(dirs['misc'], 'taylor_diagram.png'))
    else:
        print("\nNo successful models — skipping diagnostic plots.")


if __name__ == '__main__':
    main()
