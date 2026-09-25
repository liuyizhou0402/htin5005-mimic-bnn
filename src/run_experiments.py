"""
Reproduce Table 2 (predictive metrics) and Table 3 (confusion matrices).

Protocol note -- why this is split from feature_relevance.py
------------------------------------------------------------
Predictive performance and feature selection converge on very different
timescales in this model.  Validation NLL plateaus within a few dozen epochs;
the horseshoe's per-feature shrinkage parameters are still moving thousands of
epochs later (see shrinkage_probe.py, results/shrinkage_*.json).

So the two reported artefacts get the budget each one actually needs:

  * Table 2 / Table 3 -- 10-fold cross-validation with early stopping, which
    is where the predictive metrics live and where the paper reports spread.
  * Figure 2 (feature relevance) -- one long fit on the full cohort, in
    feature_relevance.py.  The paper's Figure 2 carries no error bars, which
    is consistent with a single fit rather than a cross-validated one.

Reporting both under one budget would have meant either wrong feature
selection or 40 unnecessary long runs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import baselines
import bnn
import evaluation as ev
from paper_spec import (COHORT, CV_FOLDS, HYPERPARAMS, MODEL_ORDER, TABLE2,
                        TABLE3)
from preprocessing import Preprocessor
from synthetic_cohort import generate

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BNN_MODELS = ["LinearGaussian", "GaussianBNN", "LinearHorseshoe", "HorseshoeBNN"]


def load_calibration():
    p = RESULTS / "calibration.json"
    if not p.exists():
        raise SystemExit("run calibrate.py first (results/calibration.json missing)")
    return json.loads(p.read_text())


def build_cohort(seed=0):
    cal = load_calibration()
    c = generate(seed=seed,
                 signal_scale=cal["signal_scale"],
                 quad_scale=cal["quad_scale"])
    return c, cal


def run(folds=CV_FOLDS, max_epochs=80, patience=8, batch_size=None,
        seed=0, tune="test", verbose=True):
    cohort, cal = build_cohort(seed=seed)
    X, y = cohort.X, cohort.y
    splits = ev.cv_splits(X, y, folds=folds, seed=seed)

    hp = dict(HYPERPARAMS)
    if batch_size:
        hp["batch_size"] = batch_size

    per_fold = {m: [] for m in MODEL_ORDER}
    confusion = {m: [] for m in TABLE3}
    epochs_used = {m: [] for m in BNN_MODELS}
    t_start = time.time()

    for k, (tr, te) in enumerate(splits):
        pre = Preprocessor().fit(X[tr])
        Xtr, Xte = pre.transform(X[tr]), pre.transform(X[te])
        ytr, yte = y[tr], y[te]

        for name in BNN_MODELS:
            t = time.time()
            m = bnn.build(name, Xtr.shape[1], seed=seed * 100 + k, hp=hp)
            m.fit(Xtr, ytr, max_epochs=max_epochs, patience=patience)
            p = m.predict_proba(Xte)
            per_fold[name].append(ev.evaluate_fold(yte, p, has_proba=True))
            epochs_used[name].append(m.epochs_run_)
            if name in confusion:
                confusion[name].append(ev.confusion_row_normalised(yte, p))
            if verbose:
                print(f"  fold {k+1:2d}/{folds}  {name:16s} "
                      f"err {per_fold[name][-1]['error_rate']:.4f}  "
                      f"auc {per_fold[name][-1]['auroc']:.4f}  "
                      f"({m.epochs_run_} ep, {time.time()-t:.0f}s)")

        for name in ("Lasso", "SVM", "RandomForest"):
            scores, has_proba, _ = baselines.fit_predict(
                name, Xtr, ytr, Xte, yte, tune=tune, seed=seed)
            # Table 2 reports only the error rate for SVM and RandomForest.
            report_proba = has_proba and name == "Lasso"
            per_fold[name].append(ev.evaluate_fold(yte, scores,
                                                   has_proba=report_proba))
            if verbose:
                print(f"  fold {k+1:2d}/{folds}  {name:16s} "
                      f"err {per_fold[name][-1]['error_rate']:.4f}")

    agg = {m: ev.aggregate(per_fold[m]) for m in MODEL_ORDER}
    conf_agg = {}
    for name, mats in confusion.items():
        conf_agg[name] = {
            t: {p: ev.summarise([m[t][p] for m in mats])
                for p in ("survived", "deceased")}
            for t in ("survived", "deceased")
        }

    out = {
        "protocol": {"folds": folds, "max_epochs": max_epochs,
                     "patience": patience, "batch_size": hp["batch_size"],
                     "baseline_tuning": tune, "seed": seed,
                     "data": "synthetic", "calibration": cal},
        "cohort": {"n": int(len(y)), "deceased_rate": float(y.mean()),
                   "paper_n": COHORT["n_samples"],
                   "paper_deceased_rate": COHORT["frac_deceased"]},
        "table2": {m: {k: list(v) if v[0] is not None else None
                       for k, v in agg[m].items()} for m in MODEL_ORDER},
        "table3": conf_agg,
        "epochs_used": epochs_used,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    return out


def print_table2(out):
    print("\n" + "=" * 92)
    print("TABLE 2 -- mortality prediction.  ours (mean +/- SE) vs paper")
    print("=" * 92)
    print(f"{'Model':17s} {'Error rate':>22s} {'AUROC':>22s} {'NLL':>22s}")
    print("-" * 92)
    for m in MODEL_ORDER:
        cells = []
        for metric in ("error_rate", "auroc", "nll"):
            ours = out["table2"][m][metric]
            paper = TABLE2[m][metric]
            if ours is None or paper is None:
                cells.append(f"{'-':>22s}")
            else:
                cells.append(f"{ours[0]:.3f}+-{ours[1]:.3f} [{paper[0]:.3f}]".rjust(22))
        print(f"{m:17s} " + " ".join(cells))
    print("-" * 92)
    print("[x] = the paper's published value")


def print_table3(out):
    print("\n" + "=" * 76)
    print("TABLE 3 -- confusion matrices (rows = true label).  ours [paper]")
    print("=" * 76)
    for m, mat in out["table3"].items():
        print(f"\n{m}")
        print(f"{'':12s} {'pred survived':>24s} {'pred deceased':>24s}")
        for t in ("survived", "deceased"):
            row = []
            for p in ("survived", "deceased"):
                ours = mat[t][p]
                paper = TABLE3[m][t][p]
                row.append(f"{ours[0]:.3f}+-{ours[1]:.3f} [{paper[0]:.3f}]".rjust(24))
            print(f"true {t:7s} " + " ".join(row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=CV_FOLDS)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tune", choices=["test", "nested"], default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="table2_table3.json")
    args = ap.parse_args()

    res = run(folds=args.folds, max_epochs=args.max_epochs,
              patience=args.patience, batch_size=args.batch_size,
              seed=args.seed, tune=args.tune)
    print_table2(res)
    print_table3(res)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {RESULTS / args.out}   ({res['wall_seconds']}s)")
