"""
Bonus experiments: four things the paper leaves on the table.

B1  A gradient-boosted comparator.  The paper's non-Bayesian baselines stop at
    RandomForest.  Boosted trees are the standard strong baseline on tabular
    clinical data, and if one beats the HorseshoeBNN outright then the case for
    the BNN rests on interpretability and uncertainty rather than accuracy --
    which is worth knowing before anyone deploys it.

B2  Calibration.  The whole argument for a Bayesian model here is honest
    uncertainty, but the paper reports NLL only, which mixes discrimination and
    calibration together.  We separate them with expected calibration error and
    a reliability curve.  A model can win on NLL and still be badly calibrated
    in the risk range clinicians actually act on.

B3  The cost of the paper's own tuning protocol.  Appendix B says the baselines'
    "regularization strength is optimized to minimize the test error", i.e. on
    the test fold.  We run the baselines both ways and report the gap.  This is
    a fairness problem in the paper's comparison, and it runs in the direction
    that flatters the baselines, not the authors' own model.

B4  The operating point.  Table 3 shows the HorseshoeBNN correctly flags only
    25.7% of the patients who die.  At a 13.5% event rate a 0.5 threshold is
    close to meaningless clinically -- a screening tool that misses three
    quarters of deaths would not be used.  We report what class weighting and
    threshold selection buy, and at what cost.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

tf.get_logger().setLevel("ERROR")

import baselines
import bnn
import evaluation as ev
from paper_spec import HYPERPARAMS, TABLE2
from preprocessing import Preprocessor
from run_experiments import build_cohort

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# ------------------------------------------------------------------- B1 -----
def gradient_boosting():
    """
    Histogram-based gradient boosting.

    This is scikit-learn's implementation of the algorithm LightGBM
    introduced, chosen because xgboost and lightgbm both need libomp, which is
    not installed on this machine and which we are not going to install
    silently.  `pip install xgboost lightgbm` after `brew install libomp` if
    you want those two by name; the comparison does not change.
    """
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0)


# ------------------------------------------------------------------- B2 -----
def expected_calibration_error(y, p, n_bins=15):
    """Equal-width binned |accuracy - confidence|, weighted by bin size."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    curve = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(acc - conf)
        curve.append({"bin": b, "n": int(m.sum()), "confidence": conf,
                      "observed": acc})
    return float(ece), curve


# ------------------------------------------------------------------- B4 -----
def sensitivity_at_thresholds(y, p):
    """Recall for deceased patients, and its cost, across the threshold range."""
    rows = []
    for thr in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.135, 0.1):
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        rows.append({
            "threshold": thr,
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1),
            "error_rate": (fp + fn) / len(y),
            "flagged_pct": (tp + fp) / len(y),
        })
    return rows


def youden_threshold(y, p):
    """Threshold maximising sensitivity + specificity - 1."""
    order = np.argsort(p)
    ps = p[order]
    best, best_j = 0.5, -np.inf
    for thr in np.unique(np.round(ps, 3)):
        pred = (p >= thr).astype(int)
        sens = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
        spec = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
        j = sens + spec - 1.0
        if j > best_j:
            best, best_j = float(thr), j
    return best, float(best_j)


# ------------------------------------------------------------------ driver --
def run(folds=5, max_epochs=80, patience=8, batch_size=256, seed=0):
    cohort, cal = build_cohort(seed=seed)
    X, y = cohort.X, cohort.y
    hp = dict(HYPERPARAMS)
    hp["batch_size"] = batch_size

    gb_folds, hs_folds = [], []
    gb_p, hs_p, all_y = [], [], []
    tuning = {n: {"test": [], "nested": []} for n in ("Lasso", "SVM", "RandomForest")}

    for k, (tr, te) in enumerate(ev.cv_splits(X, y, folds=folds, seed=seed)):
        pre = Preprocessor().fit(X[tr])
        Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
        ytr, yte = y[tr], y[te]

        # B1 -- boosted trees
        gb = gradient_boosting().fit(Xtr, ytr)
        pg = gb.predict_proba(Xte)[:, 1]
        gb_folds.append(ev.evaluate_fold(yte, pg))

        # the paper's own model, for a like-for-like comparison
        m = bnn.build("HorseshoeBNN", Xtr.shape[1], seed=seed * 100 + k, hp=hp)
        m.fit(Xtr, ytr, max_epochs=max_epochs, patience=patience)
        ph = m.predict_proba(Xte)
        hs_folds.append(ev.evaluate_fold(yte, ph))

        gb_p.append(pg); hs_p.append(ph); all_y.append(yte)

        # B3 -- tuning protocol
        for name in tuning:
            for mode in ("test", "nested"):
                s, has_proba, _ = baselines.fit_predict(
                    name, Xtr, ytr, Xte, yte, tune=mode, seed=seed)
                tuning[name][mode].append(ev.evaluate_fold(yte, s, has_proba=False))
        print(f"  fold {k+1}/{folds} done")

    gb_p = np.concatenate(gb_p); hs_p = np.concatenate(hs_p)
    all_y = np.concatenate(all_y)

    ece_gb, curve_gb = expected_calibration_error(all_y, gb_p)
    ece_hs, curve_hs = expected_calibration_error(all_y, hs_p)
    thr_hs, j_hs = youden_threshold(all_y, hs_p)

    return {
        "protocol": {"folds": folds, "max_epochs": max_epochs,
                     "batch_size": batch_size, "seed": seed,
                     "calibration": cal},
        "B1_gradient_boosting": {
            "GradientBoosting": {k: list(v) for k, v in ev.aggregate(gb_folds).items()},
            "HorseshoeBNN": {k: list(v) for k, v in ev.aggregate(hs_folds).items()},
            "paper_HorseshoeBNN": {k: list(v) if v else None
                                   for k, v in TABLE2["HorseshoeBNN"].items()},
        },
        "B2_calibration": {
            "GradientBoosting": {"ece": ece_gb, "curve": curve_gb},
            "HorseshoeBNN": {"ece": ece_hs, "curve": curve_hs},
        },
        "B3_tuning_protocol": {
            n: {mode: {k: list(v) for k, v in ev.aggregate(r).items()}
                for mode, r in modes.items()}
            for n, modes in tuning.items()
        },
        "B4_operating_point": {
            "HorseshoeBNN_thresholds": sensitivity_at_thresholds(all_y, hs_p),
            "youden_threshold": thr_hs, "youden_J": j_hs,
            "paper_sensitivity_at_0.5": 0.257,
        },
    }


def report(out):
    print("\n" + "=" * 78)
    print("B1 -- gradient boosting vs the paper's model")
    print("=" * 78)
    b1 = out["B1_gradient_boosting"]
    print(f"{'model':22s} {'error':>16s} {'AUROC':>16s} {'NLL':>16s}")
    for k in ("HorseshoeBNN", "GradientBoosting"):
        v = b1[k]
        print(f"{k:22s} {v['error_rate'][0]:10.4f}+-{v['error_rate'][1]:.3f} "
              f"{v['auroc'][0]:10.4f}+-{v['auroc'][1]:.3f} "
              f"{v['nll'][0]:10.4f}+-{v['nll'][1]:.3f}")
    p = b1["paper_HorseshoeBNN"]
    print(f"{'(paper HorseshoeBNN)':22s} {p['error_rate'][0]:16.4f} "
          f"{p['auroc'][0]:16.4f} {p['nll'][0]:16.4f}")

    print("\n" + "=" * 78)
    print("B2 -- calibration (expected calibration error, lower is better)")
    print("=" * 78)
    for k, v in out["B2_calibration"].items():
        print(f"  {k:22s} ECE {v['ece']:.4f}")

    print("\n" + "=" * 78)
    print("B3 -- what the paper's 'tune on the test fold' protocol is worth")
    print("=" * 78)
    print(f"{'baseline':16s} {'tuned on test':>16s} {'tuned on train':>16s} {'inflation':>12s}")
    for n, modes in out["B3_tuning_protocol"].items():
        a = modes["test"]["error_rate"][0]
        b = modes["nested"]["error_rate"][0]
        print(f"{n:16s} {a:16.4f} {b:16.4f} {b - a:+12.4f}")
    print("  (positive inflation = the paper's protocol reports a lower error "
          "than an honest one)")

    print("\n" + "=" * 78)
    print("B4 -- operating point: who actually gets flagged")
    print("=" * 78)
    print(f"{'threshold':>10s} {'sensitivity':>12s} {'specificity':>12s} "
          f"{'PPV':>8s} {'error':>8s} {'% flagged':>10s}")
    for r in out["B4_operating_point"]["HorseshoeBNN_thresholds"]:
        print(f"{r['threshold']:10.3f} {r['sensitivity']:12.3f} "
              f"{r['specificity']:12.3f} {r['ppv']:8.3f} "
              f"{r['error_rate']:8.3f} {r['flagged_pct']:10.1%}")
    o = out["B4_operating_point"]
    print(f"\n  Youden-optimal threshold {o['youden_threshold']:.3f} (J={o['youden_J']:.3f})")
    print(f"  paper's sensitivity at 0.5: {o['paper_sensitivity_at_0.5']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = run(folds=args.folds, max_epochs=args.max_epochs,
              batch_size=args.batch_size, seed=args.seed)
    report(res)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bonus.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {RESULTS / 'bonus.json'}")
