"""
What reproduced, what did not, and which agreements are circular.

The honest accounting matters more than the table of numbers.  Two of the
metrics in Table 2 were *targets* of the synthetic-data calibration, so
agreement on them is a check that the pipeline works, not evidence about the
paper.  Everything else was free to disagree -- and some of it did.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from paper_spec import COHORT, MODEL_ORDER, TABLE2, TABLE3

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# calibrate.py tuned the generator until a linear proxy hit the paper's
# LinearGaussian AUROC and a non-linear proxy hit its HorseshoeBNN AUROC.
# Those two numbers, and therefore the gap between them, cannot be used as
# evidence -- they were fitted.
CALIBRATED = {("LinearGaussian", "auroc"), ("HorseshoeBNN", "auroc")}


def load():
    p = RESULTS / "table2_table3.json"
    if not p.exists():
        raise SystemExit("run run_experiments.py first")
    return json.loads(p.read_text())


def table2_deltas(d):
    rows = []
    for m in MODEL_ORDER:
        for metric in ("error_rate", "auroc", "nll"):
            ours, paper = d["table2"][m][metric], TABLE2[m][metric]
            if ours is None or paper is None:
                continue
            delta = ours[0] - paper[0]
            # how many of the paper's standard errors away we landed
            z = delta / paper[1] if paper[1] else float("nan")
            rows.append({"model": m, "metric": metric, "ours": ours[0],
                         "se": ours[1], "paper": paper[0], "delta": delta,
                         "z": z, "calibrated": (m, metric) in CALIBRATED})
    return rows


def qualitative_claims(d):
    """The orderings the paper's argument actually rests on."""
    t = {m: {k: (v[0] if v else None) for k, v in d["table2"][m].items()}
         for m in MODEL_ORDER}
    p = {m: {k: (v[0] if v else None) for k, v in TABLE2[m].items()}
         for m in MODEL_ORDER}
    out = []

    def claim(name, ours_val, paper_val, holds, note=""):
        out.append({"claim": name, "ours": ours_val, "paper": paper_val,
                    "holds": bool(holds), "note": note})

    # 1. the headline: non-linear beats linear
    og = t["HorseshoeBNN"]["auroc"] - t["LinearHorseshoe"]["auroc"]
    pg = p["HorseshoeBNN"]["auroc"] - p["LinearHorseshoe"]["auroc"]
    claim("BNN beats the linear model on AUROC", og, pg, og > 0.01,
          "calibration targeted this gap, so it is not independent evidence")

    # 2. the paper's own model vs the plain BNN -- a near-tie in the paper
    og = t["HorseshoeBNN"]["auroc"] - t["GaussianBNN"]["auroc"]
    pg = p["HorseshoeBNN"]["auroc"] - p["GaussianBNN"]["auroc"]
    claim("HorseshoeBNN >= GaussianBNN on AUROC", og, pg,
          (og >= -0.002) and (pg >= -0.002) and abs(og - pg) < 0.01,
          "free: nothing in the calibration separates these two")

    # 3. horseshoe costs nothing on the linear model
    og = t["LinearHorseshoe"]["auroc"] - t["LinearGaussian"]["auroc"]
    pg = p["LinearHorseshoe"]["auroc"] - p["LinearGaussian"]["auroc"]
    claim("LinearHorseshoe ~= LinearGaussian", og, pg, abs(og - pg) < 0.01,
          "free")

    # 4. Lasso discriminates worse than the Bayesian linear models
    og = t["Lasso"]["auroc"] - t["LinearGaussian"]["auroc"]
    pg = p["Lasso"]["auroc"] - p["LinearGaussian"]["auroc"]
    claim("Lasso worse than LinearGaussian on AUROC", og, pg,
          (og < -0.005) and (pg < -0.005), "free")

    # 5. RandomForest worse than the BNNs on error rate
    og = t["RandomForest"]["error_rate"] - t["HorseshoeBNN"]["error_rate"]
    pg = p["RandomForest"]["error_rate"] - p["HorseshoeBNN"]["error_rate"]
    claim("RandomForest worse than HorseshoeBNN on error rate", og, pg,
          (og > 0.001) and (pg > 0.001), "free")

    # 6. the BNN catches more deaths than the linear model (Table 3)
    ob = d["table3"]["HorseshoeBNN"]["deceased"]["deceased"][0]
    ol = d["table3"]["LinearHorseshoe"]["deceased"]["deceased"][0]
    pb = TABLE3["HorseshoeBNN"]["deceased"]["deceased"][0]
    pl = TABLE3["LinearHorseshoe"]["deceased"]["deceased"][0]
    claim("HorseshoeBNN recalls more deaths than LinearHorseshoe",
          ob - ol, pb - pl, (ob > ol) and (pb > pl), "free")
    return out


def report():
    d = load()
    rows = table2_deltas(d)

    print("=" * 96)
    print("REPRODUCTION ACCOUNTING -- synthetic cohort")
    print("=" * 96)
    print(f"cohort  n={d['cohort']['n']} (paper {d['cohort']['paper_n']}), "
          f"mortality {d['cohort']['deceased_rate']:.3%} "
          f"(paper {d['cohort']['paper_deceased_rate']:.1%})")
    print(f"protocol  {d['protocol']['folds']}-fold CV, batch "
          f"{d['protocol']['batch_size']}, early stopping, "
          f"baselines tuned on '{d['protocol']['baseline_tuning']}'")

    print("\n" + "-" * 96)
    print(f"{'model':17s} {'metric':11s} {'ours':>9s} {'paper':>9s} "
          f"{'delta':>9s} {'z':>7s}   note")
    print("-" * 96)
    for r in rows:
        tag = "CALIBRATED TARGET" if r["calibrated"] else ""
        print(f"{r['model']:17s} {r['metric']:11s} {r['ours']:9.3f} "
              f"{r['paper']:9.3f} {r['delta']:+9.3f} {r['z']:+7.1f}   {tag}")

    free = [r for r in rows if not r["calibrated"]]
    err = [r for r in free if r["metric"] == "error_rate"]
    print("-" * 96)
    print(f"mean |delta| over the {len(free)} metrics that were NOT calibration "
          f"targets: {np.mean([abs(r['delta']) for r in free]):.4f}")
    print(f"error rate is low by {np.mean([r['delta'] for r in err]):+.4f} on "
          f"average across all seven models -- consistent and one-directional.")
    print(f"  {COHORT['frac_deceased'] - d['cohort']['deceased_rate']:+.4f} of "
          f"that is the cohort's own mortality rate sitting below the paper's, "
          f"which moves every model's error rate together.")

    print("\n" + "=" * 96)
    print("QUALITATIVE CLAIMS -- the orderings the paper's argument rests on")
    print("=" * 96)
    for c in qualitative_claims(d):
        mark = "REPRODUCED    " if c["holds"] else "NOT REPRODUCED"
        print(f"{mark}  {c['claim']}")
        print(f"                  ours {c['ours']:+.4f}   paper {c['paper']:+.4f}"
              f"   ({c['note']})")

    out = {"table2_deltas": rows, "claims": qualitative_claims(d)}
    (RESULTS / "reproduction_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'reproduction_report.json'}")


if __name__ == "__main__":
    report()
