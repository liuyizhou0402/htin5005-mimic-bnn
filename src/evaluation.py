"""
The three metrics of Table 2, the confusion matrix of Table 3, and the
10-fold cross-validation harness that produces "mean +/- standard error".

The paper reports "the mean value and standard error of each metric over
10-fold cross-validation", so the reported spread is std / sqrt(10), not std.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from paper_spec import CV_FOLDS

EPS = 1e-7


def error_rate(y_true, p, threshold=0.5):
    return float(np.mean((p >= threshold).astype(int) != y_true))


def auroc(y_true, p):
    return float(roc_auc_score(y_true, p))


def nll(y_true, p):
    """Mean negative log-likelihood of the predictive distribution."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def confusion_row_normalised(y_true, p, threshold=0.5):
    """
    Table 3's layout: rows = true label, columns = predicted, each row sums
    to 1.  Returns {"survived": {...}, "deceased": {...}}.
    """
    pred = (p >= threshold).astype(int)
    out = {}
    for label, name in ((0, "survived"), (1, "deceased")):
        m = y_true == label
        n = int(m.sum())
        if n == 0:
            out[name] = {"survived": float("nan"), "deceased": float("nan")}
            continue
        out[name] = {
            "survived": float(np.mean(pred[m] == 0)),
            "deceased": float(np.mean(pred[m] == 1)),
        }
    return out


def summarise(values):
    """mean and standard error, matching the paper's '+/-' convention."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if v.size == 0:
        return None, None
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def cv_splits(X, y, folds=CV_FOLDS, seed=0):
    """Stratified so every fold keeps the 13.5% mortality rate."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def evaluate_fold(y_true, p, has_proba=True):
    """All Table 2 metrics for one fold.  SVM has no probabilities."""
    res = {"error_rate": error_rate(y_true, p)}
    if has_proba:
        res["auroc"] = auroc(y_true, p)
        res["nll"] = nll(y_true, p)
    else:
        res["auroc"] = None
        res["nll"] = None
    return res


def aggregate(fold_results):
    """List of per-fold dicts -> {metric: (mean, standard_error)}."""
    out = {}
    for metric in ("error_rate", "auroc", "nll"):
        out[metric] = summarise([r.get(metric) for r in fold_results])
    return out


def compare_to_paper(name, agg, table2):
    """Signed gap against the published value, in units of the paper's SE."""
    rows = []
    for metric in ("error_rate", "auroc", "nll"):
        target = table2[name][metric]
        ours = agg[metric]
        if target is None or ours[0] is None:
            rows.append((metric, ours[0], None, None, None))
            continue
        delta = ours[0] - target[0]
        z = delta / target[1] if target[1] else float("nan")
        rows.append((metric, ours[0], ours[1], target[0], (delta, z)))
    return rows
