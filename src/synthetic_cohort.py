"""
Synthetic stand-in for the paper's MIMIC-III cohort.

Why this exists
---------------
The real cohort needs PhysioNet credentialed access plus an overnight run of
YerevaNN/mimic3-benchmarks.  While that is pending we generate a cohort with
the same *shape* as the paper's (n=17903, 17 features, 13.5% mortality), the
same Table 7 value ranges, and -- crucially -- the same qualitative structure
that the paper's Interpretability section reports:

  * Height and Capillary refill rate come out irrelevant.  We do NOT hardcode
    a zero coefficient for them.  They get a real (small) effect in the latent
    generative model and are then made ~95-99% missing, exactly as in
    MIMIC-III.  Mean imputation flattens them into near-constants, and the
    signal disappears on its own.  Same mechanism as the real data.
  * Blood pH acts through a U-shape (both acidosis and alkalosis are bad), so
    a linear model cannot use it but a non-linear one can.  This is the
    paper's stated explanation for why the HorseshoeBNN picks up pH and the
    LinearHorseshoe model does not.
  * GCS total is the sum of the three GCS sub-scores, so the design carries
    the real collinearity a sparsity prior has to resolve.

Honest limits (repeated in the README and meant for the report's Discussion):
  * Missingness rates are read off Figure 2's left panel by eye.  The paper
    prints no table of them.  They are estimates, not measurements.
  * Missingness here is MCAR.  In real ICU data it is informative (a test is
    ordered because the patient is unwell).  The paper does not model this
    either, but on real data that difference is real.
  * Marginal means and SDs are clinically plausible values, not MIMIC-III
    statistics.  The paper publishes no Table 1 of cohort characteristics, so
    unlike a Table-1-driven synthesis there is nothing to match them against.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from paper_spec import COHORT, FEATURES, FEATURE_NAMES

# --------------------------------------------------------------- marginals --
# (mean, sd, kind).  kind: "normal" | "lognormal" | "ordinal"
MARGINALS = {
    "Height":                   (168.0, 10.5, "normal"),
    "Temperature":              (36.9,   0.8, "normal"),
    "Blood pH":                 (7.38,  0.09, "normal"),
    "Fraction inspired oxygen": (0.45,  0.15, "normal"),
    "Capillary refill rate":    (1.60,  0.85, "lognormal"),
    "Heart Rate":               (88.0,  17.0, "normal"),
    "Systolic blood pressure":  (120.0, 20.0, "normal"),
    "Diastolic blood pressure": (60.0,  12.0, "normal"),
    "Mean blood pressure":      (78.0,  14.0, "normal"),
    "Weight":                   (81.0,  22.0, "lognormal"),
    "Glucose":                  (140.0, 55.0, "lognormal"),
    "Respiratory rate":         (19.0,   5.0, "normal"),
    "Oxygen saturation":        (97.0,   2.5, "normal"),
    "GCS: eye opening":         (3.4,   0.95, "ordinal"),
    "GCS: motor response":      (5.2,   1.35, "ordinal"),
    "GCS: verbal response":     (3.6,   1.45, "ordinal"),
    "GCS: total":               (None,  None, "derived"),   # = sum of the three
}

# ------------------------------------------------------------- missingness --
# Estimated from Figure 2 (left panel) of the paper.  See "Honest limits".
MISSINGNESS = {
    "Capillary refill rate":    0.99,
    "Height":                   0.95,
    "Fraction inspired oxygen": 0.75,
    "Weight":                   0.60,
    "Blood pH":                 0.55,
    "Glucose":                  0.45,
    "Temperature":              0.30,
    "GCS: total":               0.25,
    "GCS: eye opening":         0.22,
    "GCS: motor response":      0.22,
    "GCS: verbal response":     0.22,
    "Mean blood pressure":      0.15,
    "Systolic blood pressure":  0.10,
    "Diastolic blood pressure": 0.10,
    "Respiratory rate":         0.08,
    "Oxygen saturation":        0.06,
    "Heart Rate":               0.05,
}

# Correlation blocks among the latent Gaussians.  Only pairs that are
# physiologically linked; everything else is independent.
CORRELATIONS = [
    ("Systolic blood pressure", "Diastolic blood pressure",  0.62),
    ("Systolic blood pressure", "Mean blood pressure",       0.85),
    ("Diastolic blood pressure", "Mean blood pressure",      0.88),
    ("Heart Rate",              "Respiratory rate",          0.30),
    ("Oxygen saturation",       "Fraction inspired oxygen", -0.35),
    ("Height",                  "Weight",                    0.45),
    ("GCS: eye opening",        "GCS: motor response",       0.78),
    ("GCS: eye opening",        "GCS: verbal response",      0.75),
    ("GCS: motor response",     "GCS: verbal response",      0.80),
    ("Blood pH",                "Respiratory rate",         -0.25),
    ("Glucose",                 "Heart Rate",                0.18),
]

# ------------------------------------------------------- outcome structure --
# Standardised-scale log-odds contributions, before global scaling.
# Sign convention: positive = raises mortality risk.
LINEAR_EFFECTS = {
    "GCS: total":               -0.95,   # lower GCS -> worse
    "GCS: motor response":      -0.30,
    "GCS: verbal response":     -0.18,
    "GCS: eye opening":         -0.12,
    "Oxygen saturation":        -0.42,
    "Mean blood pressure":      -0.40,
    "Diastolic blood pressure": -0.34,
    "Respiratory rate":          0.36,
    "Heart Rate":                0.30,
    "Fraction inspired oxygen":  0.33,
    "Glucose":                   0.22,
    "Temperature":              -0.16,
    "Weight":                   -0.08,
    "Height":                   -0.05,   # tiny, and 95% missing -> vanishes
    "Capillary refill rate":     0.10,   # small, and 99% missing -> vanishes
    "Blood pH":                  0.00,   # pH acts only through the U-shape
    "Systolic blood pressure":   0.00,   # ditto
}

# U-shaped (non-linear) effects: contribution = coef * ((x - centre)/width)^2
#
# centre=None means "use this feature's own sample mean", and that is not a
# convenience -- it is required for the effect to be purely non-linear.
# Expanding about the sample mean m:
#
#     (x - c)^2 = (x - m)^2 + 2(m - c)(x - m) + (m - c)^2
#                             ^^^^^^^^^^^^^^^
#                             a LINEAR term, which a linear model can use
#
# An earlier version of this file centred pH at 7.40 (physiological normal)
# while drawing it around 7.38, a 0.02 offset that leaked a linear effect of
# -0.244 in standardised log-odds -- larger than most of the intended linear
# effects in LINEAR_EFFECTS.  The LinearHorseshoe model duly "discovered" pH,
# which destroyed the very contrast this cohort exists to test.  Centring on
# the sample mean makes the leak exactly zero.
QUADRATIC_EFFECTS = {
    "Blood pH":                (0.55, None, 0.09),   # coef, centre, width (raw)
    "Systolic blood pressure": (0.30, None, 20.0),
}


@dataclass
class Cohort:
    """Generated cohort.  `X` is post-missingness (NaN where missing)."""
    X: np.ndarray                     # (n, 17) with NaN for missing
    y: np.ndarray                     # (n,) 1 = deceased
    X_true: np.ndarray                # (n, 17) latent, no missingness, no range filter
    logits: np.ndarray                # (n,) true log-odds
    feature_names: list = field(default_factory=lambda: list(FEATURE_NAMES))


def _correlation_matrix(names):
    idx = {n: i for i, n in enumerate(names)}
    C = np.eye(len(names))
    for a, b, r in CORRELATIONS:
        i, j = idx[a], idx[b]
        C[i, j] = C[j, i] = r
    # Project to the nearest positive-definite matrix if the hand-written
    # correlations are not internally consistent.
    w, V = np.linalg.eigh(C)
    if w.min() < 1e-6:
        w = np.clip(w, 1e-6, None)
        C = V @ np.diag(w) @ V.T
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
    return C


def _lognormal_params(mean, sd):
    """Solve for the underlying normal's (mu, sigma) given target mean/sd."""
    var = sd ** 2
    sigma2 = np.log(1.0 + var / mean ** 2)
    mu = np.log(mean) - 0.5 * sigma2
    return mu, np.sqrt(sigma2)


def _apply_ranges(X):
    """Table 7 range filter -> out-of-range values become NaN."""
    X = X.copy()
    for j, (_, _, lo, hi, strict) in enumerate(FEATURES):
        col = X[:, j]
        bad = np.zeros(len(col), dtype=bool)
        with np.errstate(invalid="ignore"):
            bad |= (col <= lo) if strict else (col < lo)
            if hi is not None:
                bad |= col > hi
        bad &= ~np.isnan(col)
        X[bad, j] = np.nan
    return X


def generate(n=None, seed=0, signal_scale=1.0, quad_scale=1.0,
             contamination=0.004, return_details=False):
    """
    Build the synthetic cohort.

    signal_scale / quad_scale multiply the linear and quadratic effect blocks.
    They are what `calibrate.py` tunes so the fitted models land on the
    paper's AUROC values.  `contamination` is the fraction of physiologically
    impossible values injected per feature so that the Table 7 filter has
    something real to remove.
    """
    n = n or COHORT["n_samples"]
    rng = np.random.default_rng(seed)
    names = list(FEATURE_NAMES)
    k = len(names)

    # 1. latent Gaussian draw with the correlation structure
    C = _correlation_matrix(names)
    Z = rng.multivariate_normal(np.zeros(k), C, size=n)

    # 2. map to marginals
    X = np.empty((n, k))
    for j, name in enumerate(names):
        mean, sd, kind = MARGINALS[name]
        z = Z[:, j]
        if kind == "normal":
            X[:, j] = mean + sd * z
        elif kind == "lognormal":
            mu, sigma = _lognormal_params(mean, sd)
            X[:, j] = np.exp(mu + sigma * z)
        elif kind == "ordinal":
            X[:, j] = mean + sd * z          # rounded/clipped below
        elif kind == "derived":
            X[:, j] = np.nan                 # filled after the components exist
        else:
            raise ValueError(kind)

    # GCS sub-scores -> integers inside their Table 7 bounds
    for name, lo, hi in [("GCS: eye opening", 1, 4),
                         ("GCS: motor response", 1, 6),
                         ("GCS: verbal response", 1, 5)]:
        j = names.index(name)
        X[:, j] = np.clip(np.rint(X[:, j]), lo, hi)

    # GCS total is the sum of its components, by definition
    j_tot = names.index("GCS: total")
    X[:, j_tot] = (X[:, names.index("GCS: eye opening")]
                   + X[:, names.index("GCS: motor response")]
                   + X[:, names.index("GCS: verbal response")])

    # FiO2 and SpO2 live on bounded scales
    j = names.index("Fraction inspired oxygen")
    X[:, j] = np.clip(X[:, j], 0.21, 1.0)
    j = names.index("Oxygen saturation")
    X[:, j] = np.clip(X[:, j], 50.0, 100.0)

    X_true = X.copy()

    # 3. outcome from the *true* values (before missingness / filtering)
    logits = np.zeros(n)
    for name, beta in LINEAR_EFFECTS.items():
        if beta == 0.0:
            continue
        col = X_true[:, names.index(name)]
        z = (col - col.mean()) / col.std()
        logits += signal_scale * beta * z
    for name, (coef, centre, width) in QUADRATIC_EFFECTS.items():
        col = X_true[:, names.index(name)]
        c = col.mean() if centre is None else centre
        logits += quad_scale * coef * ((col - c) / width) ** 2

    # centre the intercept on the paper's 13.5% mortality
    target = COHORT["frac_deceased"]
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(logits + mid)))
        if p.mean() > target:
            hi = mid
        else:
            lo = mid
    logits = logits + 0.5 * (lo + hi)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int)

    # 4. inject physiologically impossible values so the Table 7 filter bites
    if contamination > 0:
        for j, (_, _, flo, fhi, _strict) in enumerate(FEATURES):
            m = rng.random(n) < contamination
            if not m.any():
                continue
            if fhi is not None:
                X[m, j] = fhi * rng.uniform(1.2, 3.0, m.sum())
            else:
                X[m, j] = -rng.uniform(1.0, 5.0, m.sum())

    # 5. MCAR missingness
    for j, name in enumerate(names):
        rate = MISSINGNESS.get(name, 0.0)
        if rate > 0:
            X[rng.random(n) < rate, j] = np.nan

    # 6. Table 7 range filter
    X = _apply_ranges(X)

    cohort = Cohort(X=X, y=y, X_true=X_true, logits=logits)
    if return_details:
        return cohort, {"correlation": C, "names": names}
    return cohort


if __name__ == "__main__":
    c = generate(seed=0)
    print(f"n={len(c.y)}  features={c.X.shape[1]}  "
          f"deceased={c.y.mean():.3%} (paper {COHORT['frac_deceased']:.1%})")
    print("\nmissing rate after range filter:")
    for j, name in enumerate(c.feature_names):
        obs = np.isnan(c.X[:, j]).mean()
        print(f"  {name:28s} {obs:6.1%}   (target {MISSINGNESS.get(name, 0):.0%})")
