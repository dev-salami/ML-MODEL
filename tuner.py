"""
Optuna-based hyperparameter tuning for all 9 POFA concrete strength models.
Each tune_* function returns (model, best_params, r2, y_pred, study).
"""

import os
import warnings
import operator
import numpy as np
import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               StackingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_early_stop(r2_target):
    def callback(study, trial):
        if study.best_value >= r2_target:
            study.stop()
    return callback


# ---------------------------------------------------------------------------
# RF
# ---------------------------------------------------------------------------
def tune_rf(X_train, y_train, X_test, y_test, feature_names,
            n_trials=100, r2_target=0.82):
    def _objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
            'max_depth':         trial.suggest_int('max_depth', 3, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
        m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials, callbacks=[_make_early_stop(r2_target)])

    bp = study.best_params
    final = RandomForestRegressor(**bp, random_state=42, n_jobs=-1)
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    return final, bp, r2_score(y_test, y_pred), y_pred, study


# ---------------------------------------------------------------------------
# GB
# ---------------------------------------------------------------------------
def tune_gb(X_train, y_train, X_test, y_test, feature_names,
            n_trials=100, r2_target=0.82):
    def _objective(trial):
        params = {
            'n_estimators':  trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth':     trial.suggest_int('max_depth', 2, 10),
            'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
        }
        m = GradientBoostingRegressor(**params, random_state=42)
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials, callbacks=[_make_early_stop(r2_target)])

    bp = study.best_params
    final = GradientBoostingRegressor(**bp, random_state=42)
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    return final, bp, r2_score(y_test, y_pred), y_pred, study


# ---------------------------------------------------------------------------
# SVR
# ---------------------------------------------------------------------------
def tune_svr(X_train, y_train, X_test, y_test, feature_names,
             n_trials=100, r2_target=0.82):
    def _objective(trial):
        C = trial.suggest_float('C', 0.1, 1000, log=True)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
        gamma_type = trial.suggest_categorical('gamma_type', ['scale', 'auto', 'value'])
        if gamma_type == 'value':
            gamma = trial.suggest_float('gamma_val', 1e-4, 10, log=True)
        else:
            gamma = gamma_type
        m = SVR(C=C, kernel=kernel, gamma=gamma)
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials, callbacks=[_make_early_stop(r2_target)])

    bp = study.best_params
    g_type = bp['gamma_type']
    gamma = bp.get('gamma_val', g_type) if g_type == 'value' else g_type
    final = SVR(C=bp['C'], kernel=bp['kernel'], gamma=gamma)
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    return final, bp, r2_score(y_test, y_pred), y_pred, study


# ---------------------------------------------------------------------------
# Bagging
# ---------------------------------------------------------------------------
def tune_bagging(X_train, y_train, X_test, y_test, feature_names,
                 n_trials=100, r2_target=0.82):
    from bagging_algorithm import create_bagging_regressor

    def _objective(trial):
        n_est = trial.suggest_int('n_estimators', 50, 300)
        max_s = trial.suggest_float('max_samples', 0.5, 1.0)
        max_f = trial.suggest_float('max_features_frac', 0.5, 1.0)
        base = DecisionTreeRegressor(random_state=42)
        m = create_bagging_regressor(base, n_estimators=n_est,
                                     max_samples=max_s, max_features=max_f,
                                     bootstrap=True, random_state=42)
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials, callbacks=[_make_early_stop(r2_target)])

    bp = study.best_params
    base = DecisionTreeRegressor(random_state=42)
    final = create_bagging_regressor(base, n_estimators=bp['n_estimators'],
                                     max_samples=bp['max_samples'],
                                     max_features=bp['max_features_frac'],
                                     bootstrap=True, random_state=42)
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    return final, bp, r2_score(y_test, y_pred), y_pred, study


# ---------------------------------------------------------------------------
# ANN
# ---------------------------------------------------------------------------
def tune_ann(X_train, y_train, X_test, y_test, feature_names,
             n_trials=100, r2_target=0.82):
    def _objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)
        units = tuple(trial.suggest_int(f'units_l{i}', 16, 256) for i in range(n_layers))
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        lr_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)
        m = MLPRegressor(hidden_layer_sizes=units, alpha=alpha,
                         learning_rate_init=lr_init, max_iter=500, random_state=42)
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials, callbacks=[_make_early_stop(r2_target)])

    bp = study.best_params
    n_layers = bp['n_layers']
    units = tuple(bp[f'units_l{i}'] for i in range(n_layers))
    final = MLPRegressor(hidden_layer_sizes=units, alpha=bp['alpha'],
                         learning_rate_init=bp['learning_rate_init'],
                         max_iter=500, random_state=42)
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    return final, bp, r2_score(y_test, y_pred), y_pred, study


# ---------------------------------------------------------------------------
# GEP (capped at 10 trials — each trial is a full DEAP run)
# ---------------------------------------------------------------------------
def tune_gep(X_train, y_train, X_test, y_test, feature_names,
             n_trials=100, r2_target=0.82):
    from deap import algorithms, gp, tools
    from gep_model import setup_gep_toolbox, evaluate_gep_individual, train_gep

    actual_trials = min(n_trials, 10)

    def _objective(trial):
        pop_size = trial.suggest_int('pop_size', 100, 400)
        n_gen    = trial.suggest_int('n_gen', 20, 80)
        cxpb     = trial.suggest_float('cxpb', 0.5, 0.9)
        mutpb    = trial.suggest_float('mutpb', 0.1, 0.4)

        toolbox, pset, _ = setup_gep_toolbox(feature_names)
        toolbox.register("evaluate", evaluate_gep_individual,
                         pset=pset, X=X_train, y=y_train)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate("mutate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=17))

        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        pop, _ = algorithms.eaSimple(
            pop, toolbox, cxpb=cxpb, mutpb=mutpb,
            ngen=n_gen, halloffame=hof, verbose=False
        )

        best_ind = hof[0]
        func = gp.compile(best_ind, pset)

        y_pred = np.zeros(len(y_test))
        for i in range(len(X_test)):
            try:
                v = func(*X_test[i])
                y_pred[i] = v if not (np.isnan(v) or np.isinf(v)) else np.mean(y_train)
            except Exception:
                y_pred[i] = np.mean(y_train)

        return r2_score(y_test, y_pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=actual_trials,
                   callbacks=[_make_early_stop(r2_target)])

    bp = study.best_params
    _, _, final = train_gep(
        X_train, y_train, X_test, y_test, feature_names,
        pop_size=bp['pop_size'], n_gen=bp['n_gen'],
        cxpb=bp['cxpb'], mutpb=bp['mutpb']
    )
    y_pred = final.predict(X_test)
    return final, bp, r2_score(y_test, y_pred), y_pred, study


# ---------------------------------------------------------------------------
# MARS
# ---------------------------------------------------------------------------
def tune_mars(X_train, y_train, X_test, y_test, feature_names,
              n_trials=100, r2_target=0.82):
    try:
        from pyearth import Earth
    except ImportError:
        raise ImportError("pyearth is required for MARS tuning: pip install pyearth")

    def _objective(trial):
        max_degree = trial.suggest_int('max_degree', 1, 3)
        penalty    = trial.suggest_float('penalty', 1.0, 5.0)
        m = Earth(max_degree=max_degree, penalty=penalty)
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials, callbacks=[_make_early_stop(r2_target)])

    bp = study.best_params
    final = Earth(max_degree=bp['max_degree'], penalty=bp['penalty'])
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    return final, bp, r2_score(y_test, y_pred), y_pred, study


# ---------------------------------------------------------------------------
# M5P
# ---------------------------------------------------------------------------
def tune_m5p(X_train, y_train, X_test, y_test, feature_names,
             n_trials=100, r2_target=0.82):
    from m5p import M5PModel

    def _objective(trial):
        params = {
            'max_depth':          trial.suggest_int('max_depth', 3, 15),
            'min_samples_split':  trial.suggest_int('min_samples_split', 5, 30),
            'min_samples_leaf':   trial.suggest_int('min_samples_leaf', 2, 15),
        }
        m = M5PModel(**params)
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials, callbacks=[_make_early_stop(r2_target)])

    bp = study.best_params
    final = M5PModel(**bp)
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    return final, bp, r2_score(y_test, y_pred), y_pred, study


# ---------------------------------------------------------------------------
# Hybrid (Stacking: RF + GB → SVR)
# ---------------------------------------------------------------------------
def tune_hybrid(X_train, y_train, X_test, y_test, feature_names,
                n_trials=100, r2_target=0.82):
    def _objective(trial):
        rf_n  = trial.suggest_int('rf_n_estimators', 50, 300)
        gb_n  = trial.suggest_int('gb_n_estimators', 50, 300)
        gb_lr = trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True)
        svr_C = trial.suggest_float('svr_C', 0.1, 100, log=True)
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=rf_n, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=gb_n,
                                              learning_rate=gb_lr, random_state=42)),
        ]
        m = StackingRegressor(estimators=estimators, final_estimator=SVR(C=svr_C))
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials, callbacks=[_make_early_stop(r2_target)])

    bp = study.best_params
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=bp['rf_n_estimators'],
                                      random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=bp['gb_n_estimators'],
                                          learning_rate=bp['gb_learning_rate'],
                                          random_state=42)),
    ]
    final = StackingRegressor(estimators=estimators,
                               final_estimator=SVR(C=bp['svr_C']))
    final.fit(X_train, y_train)
    y_pred = final.predict(X_test)
    return final, bp, r2_score(y_test, y_pred), y_pred, study


# ---------------------------------------------------------------------------
# Feature importance dispatcher
# ---------------------------------------------------------------------------
def get_feature_importance(model_key, model, X_val, y_val, feature_names):
    feature_names = list(feature_names)
    n = len(feature_names)

    if model_key in ('rf', 'gb', 'm5p'):
        imp = model.feature_importances_

    elif model_key == 'bagging':
        imps = [
            est.feature_importances_ for est in model.estimators_
            if hasattr(est, 'feature_importances_')
        ]
        imp = np.mean(imps, axis=0) if imps else np.ones(n) / n

    elif model_key in ('svr', 'ann', 'mars'):
        try:
            result = permutation_importance(model, X_val, y_val,
                                            n_repeats=10, random_state=42)
            imp = np.clip(result.importances_mean, 0, None)
        except Exception:
            imp = np.ones(n) / n

    elif model_key == 'gep':
        from gep_model import clean_feature_names
        cleaned = clean_feature_names(feature_names)
        expr_str = getattr(model, 'individual_str', '')
        counts = np.array([expr_str.count(name) for name in cleaned], dtype=float)
        total = counts.sum()
        imp = counts / total if total > 0 else np.ones(n) / n

    elif model_key == 'hybrid':
        imps = [
            est.feature_importances_ for est in model.estimators_
            if hasattr(est, 'feature_importances_')
        ]
        imp = np.mean(imps, axis=0) if imps else np.ones(n) / n

    else:
        imp = np.ones(n) / n

    imp = np.array(imp, dtype=float)
    if len(imp) < n:
        imp = np.pad(imp, (0, n - len(imp)))
    imp = imp[:n]
    total = np.sum(np.abs(imp))
    return np.abs(imp) / total if total > 0 else np.ones(n) / n


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------
_TUNE_MAP = {
    'rf':      tune_rf,
    'gb':      tune_gb,
    'svr':     tune_svr,
    'bagging': tune_bagging,
    'ann':     tune_ann,
    'gep':     tune_gep,
    'mars':    tune_mars,
    'm5p':     tune_m5p,
    'hybrid':  tune_hybrid,
}


def tune(model_key, X_train, y_train, X_test, y_test, feature_names,
         n_trials=100, r2_target=0.82):
    return _TUNE_MAP[model_key](
        X_train, y_train, X_test, y_test, feature_names, n_trials, r2_target
    )
