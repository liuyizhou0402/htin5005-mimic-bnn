"""
Figures for the report.

  fig1_table2_comparison.png   our Table 2 numbers against the paper's
  fig2_feature_relevance.png   the paper's Figure 2, three panels
  fig3_weight_histograms.png   Appendix C.3: the horseshoe dichotomy, and its
                               absence under a Gaussian prior
  fig4_shrinkage_timescales.png  why Table 2 and Figure 2 get different budgets
  fig5_operating_point.png     bonus B4: sensitivity against threshold
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_spec import FEATURE_NAMES, MODEL_ORDER, TABLE2

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

PAPER_C = "#8c8c8c"
OURS_C = "#2b6cb0"
ACCENT = "#c05621"


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------- figure 1 ---
def fig_table2():
    d = _load("table2_table3.json")
    if not d:
        return print("  skip fig1 (no table2_table3.json)")
    metrics = [("error_rate", "Error rate", False),
               ("auroc", "AUROC", True),
               ("nll", "NLL", False)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (key, label, higher_better) in zip(axes, metrics):
        names, ours, oerr, paper = [], [], [], []
        for m in MODEL_ORDER:
            o, p = d["table2"][m][key], TABLE2[m][key]
            if o is None or p is None:
                continue
            names.append(m); ours.append(o[0]); oerr.append(o[1]); paper.append(p[0])
        yy = np.arange(len(names))
        ax.barh(yy - 0.19, paper, height=0.36, color=PAPER_C, label="paper")
        ax.barh(yy + 0.19, ours, height=0.36, xerr=oerr, color=OURS_C,
                error_kw={"ecolor": "#333", "lw": 1}, label="reproduction")
        ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{label}  ({'higher' if higher_better else 'lower'} is better)",
                     fontsize=10)
        lo = min(min(paper), min(ours)); hi = max(max(paper), max(ours))
        pad = 0.12 * (hi - lo + 1e-9)
        ax.set_xlim(max(0, lo - pad), hi + pad)
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Table 2 reproduction on the synthetic cohort", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_table2_comparison.png", dpi=160)
    plt.close(fig)
    print("  fig1_table2_comparison.png")


# --------------------------------------------------------------- figure 2 ---
def fig_feature_relevance():
    d = _load("feature_relevance.json")
    if not d:
        return print("  skip fig2 (no feature_relevance.json)")
    names = d["feature_names"]
    y = np.arange(len(names))
    miss = np.array(d["missing_pct"])
    rl = np.array(d["relevance"]["LinearHorseshoe"]); rl = rl / rl.max()
    rb = np.array(d["relevance"]["HorseshoeBNN"]); rb = rb / rb.max()
    thr = d["protocol"]["relevance_threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), sharey=True)
    axes[0].barh(y, miss, color="#718096")
    axes[0].set_xlabel("% missing"); axes[0].set_xlim(0, 100)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()

    for ax, vals, title in ((axes[1], rl, "LinearHorseshoe"),
                            (axes[2], rb, "HorseshoeBNN")):
        cols = [OURS_C if v >= thr else "#cbd5e0" for v in vals]
        ax.barh(y, vals, color=cols)
        ax.axvline(thr, color=ACCENT, ls="--", lw=1.2)
        ax.set_xlabel("weight norm (relative)")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 1.05)
    axes[2].text(thr + 0.02, len(names) - 0.4, f"threshold {thr}",
                 color=ACCENT, fontsize=8, va="center")
    fig.suptitle("Figure 2 reproduction: missingness and learned feature relevance",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_feature_relevance.png", dpi=160)
    plt.close(fig)
    print("  fig2_feature_relevance.png")


# --------------------------------------------------------------- figure 3 ---
def fig_histograms():
    d = _load("feature_relevance.json")
    if not d:
        return print("  skip fig3 (no feature_relevance.json)")
    pairs = [("HorseshoeBNN", "GaussianBNN"), ("LinearHorseshoe", "LinearGaussian")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, pair in enumerate(pairs):
        for col, name in enumerate(pair):
            ax = axes[row, col]
            r = np.array(d["relevance"][name]); r = r / r.max()
            r = np.clip(r, 1e-6, None)
            bins = np.logspace(np.log10(r.min() * 0.7), 0.05, 16)
            ax.hist(r, bins=bins, color=OURS_C if "Horseshoe" in name else PAPER_C,
                    edgecolor="white")
            ax.set_xscale("log")
            gap = d["histogram"][name]["gap_decades"]
            ax.set_title(f"{name}   largest gap {gap:.2f} decades", fontsize=10)
            ax.set_xlabel("weight norm (relative, log scale)")
            ax.set_ylabel("features")
    fig.suptitle("Appendix C.3: the horseshoe splits weights into two groups; "
                 "the Gaussian prior does not", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_weight_histograms.png", dpi=160)
    plt.close(fig)
    print("  fig3_weight_histograms.png")


# --------------------------------------------------------------- figure 4 ---
def fig_timescales():
    hs = _load("shrinkage_HorseshoeBNN.json")
    if not hs:
        return print("  skip fig4 (no shrinkage_HorseshoeBNN.json)")
    ep = [r["epoch"] for r in hs["trace"]]
    nll = [r["val_nll"] for r in hs["trace"]]
    sep = [r["separation"] for r in hs["trace"]]

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(ep, nll, color=ACCENT, lw=1.8, label="validation NLL")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("validation NLL", color=ACCENT)
    ax1.tick_params(axis="y", labelcolor=ACCENT)
    ax2 = ax1.twinx()
    ax2.plot(ep, sep, color=OURS_C, lw=1.8, label="feature separation")
    ax2.set_ylabel("relevant / irrelevant weight ratio", color=OURS_C)
    ax2.tick_params(axis="y", labelcolor=OURS_C)

    knee = next((e for e, v in zip(ep, nll) if v <= min(nll) * 1.01), ep[-1])
    ax1.axvline(knee, color="#444", ls=":", lw=1)
    ax1.text(knee + 25, max(nll) * 0.98,
             f"predictive performance\nconverged (~{knee} epochs)",
             fontsize=8, va="top")
    fig.suptitle("Two timescales: predictive metrics settle early, "
                 "feature selection does not", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_shrinkage_timescales.png", dpi=160)
    plt.close(fig)
    print("  fig4_shrinkage_timescales.png")


# --------------------------------------------------------------- figure 5 ---
def fig_operating_point():
    d = _load("bonus.json")
    if not d:
        return print("  skip fig5 (no bonus.json)")
    rows = d["B4_operating_point"]["HorseshoeBNN_thresholds"]
    thr = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(thr, [r["sensitivity"] for r in rows], "o-", color=ACCENT,
            label="sensitivity (deaths caught)")
    ax.plot(thr, [r["ppv"] for r in rows], "s-", color=OURS_C,
            label="positive predictive value")
    ax.plot(thr, [r["error_rate"] for r in rows], "^-", color="#718096",
            label="error rate")
    ax.axvline(0.5, color="#444", ls=":", lw=1)
    ax.text(0.5, 0.95, " paper's threshold", fontsize=8, va="top")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("rate")
    ax.invert_xaxis(); ax.grid(alpha=0.25); ax.legend(fontsize=9)
    ax.set_title("Bonus B4: at a 0.5 threshold the model misses most deaths",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_operating_point.png", dpi=160)
    plt.close(fig)
    print("  fig5_operating_point.png")


if __name__ == "__main__":
    print("writing figures ->", FIGURES)
    fig_table2()
    fig_feature_relevance()
    fig_histograms()
    fig_timescales()
    fig_operating_point()
