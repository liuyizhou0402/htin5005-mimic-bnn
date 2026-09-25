"""
The three non-Bayesian comparators, exactly as Appendix B specifies them.

Appendix B: "For classification tasks we use the LogisticRegression model with
l1 regularization (equivalent to Lasso), the LinearSVC model with l1
regularization and the RandomForestClassifier). For all models the
regularization strength is optimized to minimize the test error."

That last sentence selects the hyperparameter on the *test* fold.  We
reproduce it verbatim as `tune="test"` because that is what the paper did and
the reproduction has to match its protocol to be comparable -- but it is
optimistically biased, so `tune="nested"` runs the same sweep inside the
training fold instead.  run_experiments.py reports both, and the difference is
a result in its own right.

LinearSVC has no predict_proba, which is why Table 2 shows no AUROC or NLL for
SVM.  RandomForest does expose probabilities, but the paper reports none for it
either ("it does not provide a principled probabilistic estimation"), so we
follow suit in the Table 2 reproduction and report them separately instead.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold

# Regularisation grids swept for "the regularization strength is optimized".
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
RF_GRID = [None, 10, 20]          # max_depth


def _lasso(C):
    # l1-penalised logistic regression == Lasso for classification.
    #
    # scikit-learn 1.9 deprecates `penalty=` in favour of `l1_ratio=` and warns
    # "Inconsistent values: penalty=l1 with l1_ratio=0.0", which reads as though
    # the L1 penalty were being ignored.  It is not: penalty="l1" with
    # liblinear, penalty="l1" with saga, and l1_ratio=1.0 with saga all return
    # the same 9-of-17 non-zero coefficients on a fixed test problem.  Keeping
    # liblinear because saga is markedly slower here for no change in result.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Inconsistent values")
        return LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                  max_iter=5000)


def _svm(C):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Inconsistent values")
        return LinearSVC(penalty="l1", C=C, dual=False, max_iter=20000)


def _rf(max_depth):
    return RandomForestClassifier(n_estimators=300, max_depth=max_depth,
                                  n_jobs=-1, random_state=0)


BASELINES = {
    "Lasso":        (_lasso, C_GRID,  True),    # has predict_proba
    "SVM":          (_svm,   C_GRID,  False),
    "RandomForest": (_rf,    RF_GRID, True),
}


def _scores(model, X):
    """Probabilities where available, decision values otherwise."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return (d >= 0).astype(float)      # used only for the error rate


def fit_predict(name, Xtr, ytr, Xte, yte, tune="test", seed=0):
    """
    Fit one baseline, sweeping its regularisation strength.

    tune="test"   -> pick the setting with the lowest error on Xte (the paper)
    tune="nested" -> pick it by 3-fold CV inside Xtr (unbiased)
    """
    build, grid, has_proba = BASELINES[name]

    if tune == "test":
        best, best_err = None, np.inf
        for g in grid:
            m = build(g).fit(Xtr, ytr)
            err = float(np.mean(m.predict(Xte) != yte))
            if err < best_err:
                best, best_err = m, err
        model = best
    elif tune == "nested":
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        best_g, best_err = None, np.inf
        for g in grid:
            errs = []
            for a, b in skf.split(Xtr, ytr):
                m = build(g).fit(Xtr[a], ytr[a])
                errs.append(float(np.mean(m.predict(Xtr[b]) != ytr[b])))
            e = float(np.mean(errs))
            if e < best_err:
                best_g, best_err = g, e
        model = build(best_g).fit(Xtr, ytr)
    else:
        raise ValueError(tune)

    return _scores(model, Xte), has_proba, model
