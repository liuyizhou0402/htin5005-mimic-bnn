"""
Reproduce Figure 2 and the Appendix C.3 weight histograms, and check the
paper's three interpretability claims.

Figure 2 has three panels: percentage of missing data per feature, the weight
norms of the LinearHorseshoe model, and the weight norms of the HorseshoeBNN.
It carries no error bars, so it is one fit on the cohort rather than a
cross-validated quantity -- which is why this runs separately from
run_experiments.py and on a much longer training budget (see that file's
docstring, and shrinkage_probe.py for where the budget comes from).

Claims under test, taken from the paper's own Interpretability section:

  C1  Height and Capillary refill rate are irrelevant to both horseshoe models.
  C2  The HorseshoeBNN picks up Blood pH; the LinearHorseshoe model does not.
      (The paper attributes this to pH acting non-linearly.)
  C3  The horseshoe weight histogram splits into two groups "which differ by
      orders of magnitude"; the Gaussian models show no such split.

The paper also verifies that dropping the features its model calls irrelevant
leaves the metrics unchanged.  refit_without_irrelevant() repeats that check.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import bnn
import evaluation as ev
from paper_spec import FEATURE_NAMES, HYPERPARAMS, REPORTED_RELEVANCE
from preprocessing import Preprocessor
from run_experiments import build_cohort

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# A feature counts as "selected" if its weight norm is above this fraction of
# the largest norm.  The paper picks its threshold by eye from the gap in the
# histogram; we fix it up front so the claim checks are not tuned after seeing
# the answer.  sensitivity_to_threshold() reports how much it matters.
RELEVANCE_THRESHOLD = 0.10


def fit_full(name, X, y, epochs, hp, seed=0):
    m = bnn.build(name, X.shape[1], seed=seed, hp=hp)
    m.fit(X, y, max_epochs=epochs, patience=10 ** 9)
    return m


def selected(relevance, threshold=RELEVANCE_THRESHOLD):
    r = np.asarray(relevance, dtype=float)
    keep = r / r.max() >= threshold
    return [FEATURE_NAMES[i] for i in np.where(keep)[0]]


def check_claims(rel_lin, rel_bnn, threshold=RELEVANCE_THRESHOLD):
    sel_lin = set(selected(rel_lin, threshold))
    sel_bnn = set(selected(rel_bnn, threshold))
    out = {}

    # C1 -- height and capillary refill rate irrelevant to both
    c1 = {}
    for f in REPORTED_RELEVANCE["irrelevant_both"]:
        c1[f] = {"in_linear": f in sel_lin, "in_bnn": f in sel_bnn,
                 "holds": (f not in sel_lin) and (f not in sel_bnn)}
    out["C1_irrelevant_both"] = {"per_feature": c1,
                                 "holds": all(v["holds"] for v in c1.values())}

    # C2 -- pH picked up by the BNN only
    f = "Blood pH"
    out["C2_pH_bnn_only"] = {
        "in_linear": f in sel_lin, "in_bnn": f in sel_bnn,
        "holds": (f in sel_bnn) and (f not in sel_lin),
    }
    out["selected_linear"] = sorted(sel_lin)
    out["selected_bnn"] = sorted(sel_bnn)
    return out


def histogram_dichotomy(relevance, min_side=2):
    """
    C3: is there a gap of orders of magnitude in the log-spaced histogram?

    Returns the largest empty run of log10 decades that actually splits the
    features into two groups.  `min_side` requires at least that many features
    on each side: without it the metric is decided by whichever single feature
    happens to be the most extreme outlier, which made a Gaussian model score
    a wider "gap" (1.19 decades) than the horseshoe model beside it (1.15) in
    an earlier run -- one isolated feature, not a dichotomy.
    """
    r = np.asarray(relevance, dtype=float)
    r = r[r > 0]
    if len(r) < 2 * min_side:
        return {"gap_decades": 0.0, "n_below": 0, "n_above": 0}
    logs = np.sort(np.log10(r / r.max()))
    gaps = np.diff(logs)
    lo, hi = min_side - 1, len(gaps) - (min_side - 1)
    i = lo + int(np.argmax(gaps[lo:hi]))
    return {"gap_decades": float(gaps[i]),
            "below": float(10 ** logs[i]), "above": float(10 ** logs[i + 1]),
            "n_below": int(i + 1), "n_above": int(len(logs) - i - 1),
            "min_side": min_side}


def refit_without_irrelevant(X, y, rel_bnn, epochs, hp, seed=0, folds=5):
    """
    The paper's own robustness check: retrain the HorseshoeBNN using only the
    features it considers relevant, and confirm the metrics do not move.
    """
    keep_names = set(selected(rel_bnn))
    keep = np.array([n in keep_names for n in FEATURE_NAMES])
    res = {}
    for label, cols in (("all_features", np.ones(len(keep), bool)),
                        ("selected_only", keep)):
        fold_res = []
        for tr, te in ev.cv_splits(X, y, folds=folds, seed=seed):
            pre = Preprocessor().fit(X[tr])
            Xtr, Xte = pre.transform(X[tr])[:, cols], pre.transform(X[te])[:, cols]
            m = bnn.build("HorseshoeBNN", int(cols.sum()), seed=seed, hp=hp)
            m.fit(Xtr, y[tr], max_epochs=min(epochs, 120), patience=10)
            fold_res.append(ev.evaluate_fold(y[te], m.predict_proba(Xte)))
        res[label] = {k: list(v) for k, v in ev.aggregate(fold_res).items()}
    res["n_features_kept"] = int(keep.sum())
    res["kept"] = sorted(keep_names)
    res["dropped"] = sorted(set(FEATURE_NAMES) - keep_names)
    return res


def run(epochs=600, seed=0, batch_size=256, do_refit=True):
    cohort, cal = build_cohort(seed=seed)
    X_raw, y = cohort.X, cohort.y
    pre = Preprocessor().fit(X_raw)
    X = pre.transform(X_raw)

    hp = dict(HYPERPARAMS)
    hp["batch_size"] = batch_size

    models, relevance = {}, {}
    for name in ("LinearHorseshoe", "HorseshoeBNN",
                 "LinearGaussian", "GaussianBNN"):
        m = fit_full(name, X, y, epochs, hp, seed=seed)
        models[name] = m
        relevance[name] = m.feature_relevance().tolist()
        print(f"  fitted {name:16s} ({m.epochs_run_} epochs)")

    missing_pct = (np.isnan(X_raw).mean(axis=0) * 100).tolist()

    out = {
        "protocol": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
                     "relevance_threshold": RELEVANCE_THRESHOLD,
                     "calibration": cal},
        "feature_names": FEATURE_NAMES,
        "missing_pct": missing_pct,
        "relevance": relevance,
        "claims": check_claims(relevance["LinearHorseshoe"],
                               relevance["HorseshoeBNN"]),
        "histogram": {k: histogram_dichotomy(v) for k, v in relevance.items()},
        "tau": {"HorseshoeBNN": models["HorseshoeBNN"].tau.mean.numpy().tolist(),
                "LinearHorseshoe": models["LinearHorseshoe"].tau.mean.numpy().tolist()},
    }
    if do_refit:
        print("  running the paper's drop-irrelevant-features check ...")
        out["refit_check"] = refit_without_irrelevant(
            X_raw, y, relevance["HorseshoeBNN"], epochs, hp, seed=seed)
    return out


def report(out):
    names = out["feature_names"]
    rl = np.array(out["relevance"]["LinearHorseshoe"])
    rb = np.array(out["relevance"]["HorseshoeBNN"])
    print("\n" + "=" * 84)
    print("FIGURE 2 -- feature relevance (weight norms, normalised to the max)")
    print("=" * 84)
    print(f"{'feature':28s} {'% missing':>10s} {'LinearHorseshoe':>18s} {'HorseshoeBNN':>16s}")
    print("-" * 84)
    for i, n in enumerate(names):
        print(f"{n:28s} {out['missing_pct'][i]:9.1f}% "
              f"{rl[i]/rl.max():18.4f} {rb[i]/rb.max():16.4f}")
    print("-" * 84)

    c = out["claims"]
    print("\nCLAIM CHECKS")
    print(f"  C1  height + capillary refill irrelevant to both : "
          f"{'REPRODUCED' if c['C1_irrelevant_both']['holds'] else 'NOT REPRODUCED'}")
    for f, v in c["C1_irrelevant_both"]["per_feature"].items():
        print(f"        {f:26s} linear={'sel' if v['in_linear'] else 'drop'}  "
              f"bnn={'sel' if v['in_bnn'] else 'drop'}")
    print(f"  C2  pH selected by the BNN only                  : "
          f"{'REPRODUCED' if c['C2_pH_bnn_only']['holds'] else 'NOT REPRODUCED'}"
          f"  (linear={'sel' if c['C2_pH_bnn_only']['in_linear'] else 'drop'}, "
          f"bnn={'sel' if c['C2_pH_bnn_only']['in_bnn'] else 'drop'})")
    print("  C3  order-of-magnitude gap in the weight histogram:")
    for k, v in out["histogram"].items():
        print(f"        {k:18s} largest gap {v['gap_decades']:.2f} decades "
              f"({v['n_below']} below / {v['n_above']} above)")

    if "refit_check" in out:
        r = out["refit_check"]
        print(f"\n  paper's check: refit on the {r['n_features_kept']} selected "
              f"features only")
        for label in ("all_features", "selected_only"):
            m = r[label]
            print(f"        {label:15s} err {m['error_rate'][0]:.4f}  "
                  f"auc {m['auroc'][0]:.4f}  nll {m['nll'][0]:.4f}")
        print(f"        dropped: {', '.join(r['dropped']) or '(none)'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-refit", action="store_true")
    args = ap.parse_args()
    res = run(epochs=args.epochs, seed=args.seed,
              batch_size=args.batch_size, do_refit=not args.no_refit)
    report(res)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "feature_relevance.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {RESULTS / 'feature_relevance.json'}")
