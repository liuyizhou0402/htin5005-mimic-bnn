"""
How long does the horseshoe need before its feature selection converges?

This matters because early stopping on predictive NLL fires long before the
shrinkage parameters have settled: predictive accuracy plateaus after a few
dozen epochs, while tau_j for an irrelevant feature is still drifting down.
Stopping there would report "the horseshoe barely shrinks anything", which
would be an artefact of our training budget, not a finding about the paper.

Controlled test: 10 standard-normal features, only the first three carry
signal.  We track, per epoch, the ratio

    min(relevance of the 3 real features) / max(relevance of the 7 noise ones)

The paper describes a histogram with two groups "which differ by orders of
magnitude", so we are looking for this ratio to reach at least ~10x and then
flatten.  Where it flattens sets the epoch budget in run_experiments.py.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import bnn  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
N_RELEVANT = 3


def toy(n=4000, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    logit = 1.4 * X[:, 0] - 1.1 * X[:, 1] + 0.9 * X[:, 2] - 1.0
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
    return X, y


def separation(model):
    rel = model.feature_relevance()
    return float(np.min(rel[:N_RELEVANT]) / np.max(rel[N_RELEVANT:]))


def run(max_epochs=2000, every=25):
    X, y = toy()
    hp = dict(bnn.HYPERPARAMS)
    hp["batch_size"] = 128
    trace = []

    def cb(model, epoch, vnll):
        if epoch % every == 0 or epoch == max_epochs - 1:
            trace.append({"epoch": epoch, "val_nll": vnll,
                          "separation": separation(model)})

    for name in ("HorseshoeBNN", "GaussianBNN"):
        trace.clear()
        t = time.time()
        m = bnn.build(name, X.shape[1], seed=0, hp=hp)
        # patience huge -> no early stop; we want the full curve
        m.fit(X, y, max_epochs=max_epochs, patience=10 ** 9, on_epoch=cb)
        rec = {"model": name, "seconds": round(time.time() - t, 1),
               "trace": list(trace),
               "final_relevance": m.feature_relevance().tolist()}
        if name == "HorseshoeBNN":
            rec["final_tau"] = m.tau.mean.numpy().tolist()
            rec["final_lambda"] = float(m.lam.mean.numpy())
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"shrinkage_{name}.json").write_text(json.dumps(rec, indent=2))
        print(f"\n{name}  ({rec['seconds']}s)")
        for r in rec["trace"]:
            print(f"  epoch {r['epoch']:5d}  val NLL {r['val_nll']:.4f}  "
                  f"separation {r['separation']:8.2f}x")


if __name__ == "__main__":
    run()
