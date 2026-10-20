"""
Clean, numerically-safe GEP (symbolic regression via DEAP) for the Phase 1 pipeline.

Why this exists: the original gep_model.py collapsed to a constant because its fitness
replaced NaN/inf predictions with mean(y), so a constant ≈ mean scored well. This module:
  - uses numerically-safe protected operators (clipped exp, log(|x|+eps), sqrt(|x|), safe div),
  - PENALISES an individual whose predictions are non-finite or (near-)constant, instead of
    substituting a constant,
  - applies least-squares LINEAR SCALING (a·expr + b, fit on training data only) so evolution
    optimises the expression's shape, not its magnitude — this is standard GP practice, NOT
    the noise-adding "corrector" that was removed,
  - selects the final expression by VALIDATION R² (internal split of the training data),
  - runs several restarts with different seeds.

Vectorised over numpy arrays for speed. The returned GEPPredictor is picklable (stores the
expression string, the cleaned feature names, and the scaling coefficients a, b).
"""

import operator
import random

import numpy as np
from deap import base, creator, gp, tools, algorithms

EPS = 1e-9
_BIG = 1e12  # fitness penalty


# --------------------------------------------------------------------------- #
# Numerically-safe, vectorised operators
# --------------------------------------------------------------------------- #
def _safe_div(a, b):
    b = np.where(np.abs(b) < 1e-6, 1.0, b)
    return a / b

def _safe_log(x):
    return np.log(np.abs(x) + EPS)

def _safe_sqrt(x):
    return np.sqrt(np.abs(x))

def _safe_exp(x):
    return np.exp(np.clip(x, -10.0, 10.0))

def _safe_sin(x):
    return np.sin(x)

def _safe_cos(x):
    return np.cos(x)

# name -> callable, used to evaluate a stored expression string at predict time
_EVAL_CTX = {
    'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'neg': np.negative,
    '_safe_div': _safe_div, '_safe_log': _safe_log, '_safe_sqrt': _safe_sqrt,
    '_safe_exp': _safe_exp, '_safe_sin': _safe_sin, '_safe_cos': _safe_cos,
}


def clean_feature_names(feature_names):
    out = []
    for name in feature_names:
        c = (name.replace('/', '_').replace('-', '_').replace(' ', '_')
                 .replace('(', '').replace(')', '').replace('.', '_'))
        if not c[0].isalpha():
            c = 'feat_' + c
        out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Picklable predictor
# --------------------------------------------------------------------------- #
class GEPPredictor:
    """Evaluates a·f(x)+b, where f is the evolved expression (vectorised)."""

    def __init__(self, expr_str, feature_names, a, b):
        self.expr_str = expr_str
        self.feature_names = clean_feature_names(feature_names)
        self.a = float(a)
        self.b = float(b)

    def _raw(self, X):
        X = np.asarray(X, dtype=float)
        ctx = dict(_EVAL_CTX)
        for i, name in enumerate(self.feature_names):
            ctx[name] = X[:, i]
        out = eval(self.expr_str, {"__builtins__": {}}, ctx)  # noqa: S307 (sandboxed ctx)
        out = np.asarray(out, dtype=float)
        if out.ndim == 0:                      # constant expression
            out = np.full(X.shape[0], float(out))
        return out

    def predict(self, X):
        pred = self.a * self._raw(X) + self.b
        return np.nan_to_num(pred, nan=self.b, posinf=self.b, neginf=self.b)


# --------------------------------------------------------------------------- #
# Primitive set + fitness
# --------------------------------------------------------------------------- #
def _build_pset(feature_names):
    cleaned = clean_feature_names(feature_names)
    pset = gp.PrimitiveSet("MAIN", len(cleaned))
    for i, name in enumerate(cleaned):
        pset.renameArguments(**{f'ARG{i}': name})
    pset.addPrimitive(np.add, 2, name='add')
    pset.addPrimitive(np.subtract, 2, name='sub')
    pset.addPrimitive(np.multiply, 2, name='mul')
    pset.addPrimitive(np.negative, 1, name='neg')
    pset.addPrimitive(_safe_div, 2)
    pset.addPrimitive(_safe_log, 1)
    pset.addPrimitive(_safe_sqrt, 1)
    pset.addPrimitive(_safe_exp, 1)
    pset.addPrimitive(_safe_sin, 1)
    pset.addPrimitive(_safe_cos, 1)
    pset.addEphemeralConstant('randP1', lambda: round(random.uniform(-2.0, 2.0), 3))
    return pset, cleaned


def _linear_scale(raw, y):
    """Least-squares a, b for a*raw + b ≈ y (training data only)."""
    a, b = np.polyfit(raw, y, 1)
    return float(a), float(b)


def _eval_individual(individual, pset, X, y):
    try:
        func = gp.compile(individual, pset)
        out = func(*[X[:, i] for i in range(X.shape[1])])
        out = np.asarray(out, dtype=float)
        if out.ndim == 0:                       # constant tree → penalise
            return (_BIG,)
        if out.shape[0] != len(y) or not np.all(np.isfinite(out)):
            return (_BIG,)                       # non-finite → penalise (no substitution)
        if np.std(out) < 1e-6:                   # (near-)constant → penalise
            return (_BIG,)
        a, b = _linear_scale(out, y)
        pred = a * out + b
        if not np.all(np.isfinite(pred)):
            return (_BIG,)
        return (float(np.mean((y - pred) ** 2)),)
    except Exception:
        return (_BIG,)


def _setup_creator():
    if not hasattr(creator, 'FitnessMinP1'):
        creator.create('FitnessMinP1', base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'IndividualP1'):
        creator.create('IndividualP1', gp.PrimitiveTree, fitness=creator.FitnessMinP1)


def _r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_gep_phase1(X_train, y_train, feature_names,
                     pop_size=400, n_gen=120, n_restarts=3,
                     cxpb=0.7, mutpb=0.3, val_frac=0.2, verbose=False):
    """
    Fit a GEP expression on X_train only. An internal fit/validation split
    (val_frac, seed 42) is used to SELECT the expression by validation R².
    Returns (GEPPredictor, info_dict).
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    _setup_creator()

    # internal fit/val split (does not touch the global 70/15/15 split)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y_train))
    n_val = max(10, int(round(val_frac * len(y_train))))
    val_idx, fit_idx = idx[:n_val], idx[n_val:]
    Xf, yf = X_train[fit_idx], y_train[fit_idx]
    Xv, yv = X_train[val_idx], y_train[val_idx]

    pset, cleaned = _build_pset(feature_names)

    toolbox = base.Toolbox()
    toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register('individual', tools.initIterate, creator.IndividualP1, toolbox.expr)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gp.compile, pset=pset)
    toolbox.register('evaluate', _eval_individual, pset=pset, X=Xf, y=yf)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', gp.cxOnePoint)
    toolbox.register('expr_mut', gp.genFull, min_=0, max_=2)
    toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=12))
    toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=12))

    best = {'val_r2': -np.inf, 'expr': None, 'a': 0.0, 'b': float(np.mean(y_train))}

    def _val_r2_of(ind):
        """Validation R² using a,b fit on the FIT part (honest selection)."""
        func = gp.compile(ind, pset)
        rf = np.asarray(func(*[Xf[:, i] for i in range(Xf.shape[1])]), dtype=float)
        if rf.ndim == 0 or np.std(rf) < 1e-6 or not np.all(np.isfinite(rf)):
            return -np.inf, None
        a, b = _linear_scale(rf, yf)
        rv = np.asarray(func(*[Xv[:, i] for i in range(Xv.shape[1])]), dtype=float)
        if rv.ndim == 0 or not np.all(np.isfinite(rv)):
            return -np.inf, None
        return _r2(yv, a * rv + b), (a, b)

    for r in range(n_restarts):
        random.seed(42 + r)
        np.random.seed(42 + r)
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(10)
        algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
                            halloffame=hof, verbose=verbose)
        for ind in hof:
            vr2, ab = _val_r2_of(ind)
            if ab is not None and vr2 > best['val_r2']:
                best.update(val_r2=vr2, expr=str(ind), ab=ab)

    if best['expr'] is None:
        # Should not happen; fall back to mean predictor (flagged by a=0)
        return GEPPredictor('0.0', feature_names, 0.0, float(np.mean(y_train))), \
            {'val_r2': -np.inf, 'degenerate': True}

    # Refit the linear-scaling coefficients on ALL training data for the final model.
    tmp = GEPPredictor(best['expr'], feature_names, 1.0, 0.0)
    raw_all = tmp._raw(X_train)
    a, b = _linear_scale(raw_all, y_train)
    predictor = GEPPredictor(best['expr'], feature_names, a, b)

    info = {'val_r2': best['val_r2'], 'expr': best['expr'], 'a': a, 'b': b,
            'pop_size': pop_size, 'n_gen': n_gen, 'n_restarts': n_restarts,
            'degenerate': False}
    return predictor, info
