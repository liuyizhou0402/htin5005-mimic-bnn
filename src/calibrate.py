"""
Tune the synthetic generator's two signal knobs so that fitted models land on
the paper's Table 2 AUROCs.

The paper reports a 0.024 AUROC gap between linear models (0.807) and the BNNs
(0.830/0.831).  A synthetic cohort is only a fair test bed for this paper if it
reproduces *that gap*, not just the absolute level -- the whole claim under
test is "the non-linear model buys you something".  So we tune two knobs:

  signal_scale -> how much linearly-usable signal there is  -> drives 0.807
  quad_scale   -> how much of the signal is U-shaped only   -> drives the gap

Proxies are cheap stand-ins for the real models (logistic regression for the
linear family, gradient boosting for the non-linear family) so the search is
seconds rather than hours.  run_experiments.py then fits the actual models.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from paper_spec import TABLE2
from preprocessing import Preprocessor
from synthetic_cohort import generate

TARGET_LINEAR = TABLE2["LinearGaussian"]["auroc"][0]      # 0.807
TARGET_NONLIN = TABLE2["HorseshoeBNN"]["auroc"][0]        # 0.831


def _cv_auroc(X, y, model_fn, folds=2, seed=0):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        pre = Preprocessor().fit(X[tr])
        Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
        m = model_fn()
        m.fit(Xtr, y[tr])
        scores.append(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]))
    return float(np.mean(scores))


def _linear_proxy():
    return LogisticRegression(max_iter=2000, C=1.0)


def _nonlinear_proxy():
    # Deliberately modest: the proxy is here to detect the U-shaped signal the
    # BNN will exploit, not to squeeze out maximum accuracy.
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=60, l2_regularization=1.0, random_state=0)


def probe_linear(signal_scale, quad_scale, seed=0, n=None):
    c = generate(n=n, seed=seed, signal_scale=signal_scale, quad_scale=quad_scale)
    return _cv_auroc(c.X, c.y, _linear_proxy)


def probe(signal_scale, quad_scale, seed=0, n=None):
    c = generate(n=n, seed=seed, signal_scale=signal_scale, quad_scale=quad_scale)
    lin = _cv_auroc(c.X, c.y, _linear_proxy)
    nl = _cv_auroc(c.X, c.y, _nonlinear_proxy)
    return lin, nl


def _fit_linear_level(quad_scale, seed=0, n=None, iters=11, verbose=False):
    """Inner loop: pick signal_scale so the linear proxy sits on 0.807."""
    lo, hi = 0.05, 4.0
    mid = 0.5 * (lo + hi)
    lin = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lin = probe_linear(mid, quad_scale, seed=seed, n=n)
        if lin > TARGET_LINEAR:
            hi = mid
        else:
            lo = mid
    if verbose:
        print(f"      inner: signal_scale={mid:.4f} -> linear {lin:.4f}")
    return mid, lin


def search(seed=0, n=None, verbose=True):
    """
    Nested search.

    The two knobs are not independent: U-shaped signal is *noise* to a linear
    model, so raising quad_scale pushes the linear AUROC down.  Tuning them
    one after the other (as a naive coordinate search does) therefore lands
    both metrics below target.  So the inner loop re-pins the linear AUROC to
    0.807 for every candidate quad_scale, and the outer loop moves quad_scale
    until the non-linear proxy clears 0.831 on top of that.
    """
    target_gap = TARGET_NONLIN - TARGET_LINEAR
    lo, hi = 0.0, 3.0
    best = None
    for it in range(9):
        q = 0.5 * (lo + hi)
        s_scale, lin = _fit_linear_level(q, seed=seed, n=n, verbose=verbose)
        c = generate(n=n, seed=seed, signal_scale=s_scale, quad_scale=q)
        nl = _cv_auroc(c.X, c.y, _nonlinear_proxy)
        gap = nl - lin
        if verbose:
            print(f"  [outer {it}] quad={q:.4f} signal={s_scale:.4f} -> "
                  f"linear {lin:.4f}  nonlinear {nl:.4f}  gap {gap:+.4f}")
        if best is None or abs(gap - target_gap) < abs(best[4] - target_gap):
            best = (s_scale, q, lin, nl, gap)
        if gap > target_gap:
            hi = q
        else:
            lo = q
    return best[:4]


if __name__ == "__main__":
    import json, pathlib
    print(f"targets: linear {TARGET_LINEAR:.3f}, non-linear {TARGET_NONLIN:.3f}, "
          f"gap {TARGET_NONLIN - TARGET_LINEAR:+.3f}\n")
    s, q, lin, nl = search(seed=0)
    print(f"\nchosen: signal_scale={s:.4f}  quad_scale={q:.4f}")
    print(f"        linear AUROC {lin:.4f} (target {TARGET_LINEAR:.3f})")
    print(f"        non-lin AUROC {nl:.4f} (target {TARGET_NONLIN:.3f})")
    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "calibration.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"signal_scale": s, "quad_scale": q,
         "proxy_linear_auroc": lin, "proxy_nonlinear_auroc": nl,
         "target_linear_auroc": TARGET_LINEAR,
         "target_nonlinear_auroc": TARGET_NONLIN}, indent=2))
    print(f"\nwrote {out}")
