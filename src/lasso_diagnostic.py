"""
Why does our Lasso not underperform, when the paper's does?

Table 2 reports Lasso at AUROC 0.795 against 0.807 for the Bayesian linear
models -- a 0.012 deficit between two models that are both linear.  Our
reproduction shows no such gap (-0.0001).  This is one of only two qualitative
claims that did not reproduce, so it is worth a real attempt to explain rather
than a shrug.

Hypothesis: Appendix B selects the regularisation strength "to minimize the
test error".  Tuning L1 on the error rate can discard features that carry no
weight at a 0.5 threshold but still help ranking, which is what AUROC measures.
If so, the paper's Lasso deficit would be an artefact of its tuning objective
rather than a property of Lasso.

Result: not supported on this cohort.  Choosing C by error rate instead of by
AUROC costs 0.0001 AUROC, two orders of magnitude short of the reported gap,
and at the selected C the L1 penalty keeps all 17 features -- so there is
nothing being discarded for the hypothesis to bite on.

What is left is that the real MIMIC-III cohort has structure this synthetic
one does not: more collinear and noisier measurements, over which the L1 path
actually starts dropping useful features at the C that minimises error.  A
synthetic cohort built from 17 clean marginals cannot show that, and this is a
concrete example of what the synthetic stand-in cannot test.  Re-run this
script once the real cohort is available.
"""
from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import evaluation as ev
from baselines import C_GRID
from preprocessing import Preprocessor
from run_experiments import build_cohort

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
PAPER_LASSO_DEFICIT = 0.795 - 0.807      # Table 2


def sweep(folds=5, seed=0):
    cohort, _ = build_cohort(seed=seed)
    X, y = cohort.X, cohort.y
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for C in C_GRID:
            errs, aucs, nz = [], [], []
            for tr, te in ev.cv_splits(X, y, folds=folds, seed=seed):
                pre = Preprocessor().fit(X[tr])
                Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
                m = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                       max_iter=5000).fit(Xtr, y[tr])
                p = m.predict_proba(Xte)[:, 1]
                errs.append(float(np.mean((p >= 0.5).astype(int) != y[te])))
                aucs.append(float(roc_auc_score(y[te], p)))
                nz.append(int((np.abs(m.coef_) > 1e-8).sum()))
            rows.append({"C": C, "error_rate": float(np.mean(errs)),
                         "auroc": float(np.mean(aucs)),
                         "n_nonzero": float(np.mean(nz))})
    return rows


def report():
    rows = sweep()
    by_err = min(rows, key=lambda r: r["error_rate"])
    by_auc = max(rows, key=lambda r: r["auroc"])
    cost = by_auc["auroc"] - by_err["auroc"]

    print(f"{'C':>8s} {'nonzero':>10s} {'error':>9s} {'AUROC':>9s}")
    for r in rows:
        print(f"{r['C']:8g} {r['n_nonzero']:8.1f}/17 {r['error_rate']:9.4f} "
              f"{r['auroc']:9.4f}")
    print(f"\npaper's rule (minimise error) picks C={by_err['C']:g}"
          f"  -> AUROC {by_err['auroc']:.4f}, {by_err['n_nonzero']:.1f}/17 features kept")
    print(f"maximising AUROC would pick  C={by_auc['C']:g}"
          f"  -> AUROC {by_auc['auroc']:.4f}")
    print(f"AUROC cost of the paper's tuning objective: {cost:+.4f}")
    print(f"paper's reported Lasso AUROC deficit:       {PAPER_LASSO_DEFICIT:+.4f}")
    verdict = ("SUPPORTED" if cost >= 0.5 * abs(PAPER_LASSO_DEFICIT)
               else "NOT SUPPORTED")
    print(f"\nhypothesis: {verdict} -- the tuning objective explains "
          f"{cost / abs(PAPER_LASSO_DEFICIT):.1%} of the reported gap")

    out = {"sweep": rows, "chosen_by_error": by_err, "chosen_by_auroc": by_auc,
           "auroc_cost_of_error_tuning": cost,
           "paper_lasso_deficit": PAPER_LASSO_DEFICIT, "verdict": verdict}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "lasso_diagnostic.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'lasso_diagnostic.json'}")


if __name__ == "__main__":
    report()
