"""
Everything this reproduction takes from the paper, in one place.

Overweg, Popkes, Ercole, Li, Hernandez-Lobato, Zaykov, Zhang (2019).
"Interpretable Outcome Prediction with Sparse Bayesian Neural Networks
in Intensive Care."  arXiv:1905.02599v2.

Nothing here is invented.  Every constant carries the table it came from so
that the report can cite it and a reader can check it against the PDF in
docs/paper_text.txt.
"""

PAPER = {
    "title": "Interpretable Outcome Prediction with Sparse Bayesian "
             "Neural Networks in Intensive Care",
    "arxiv": "1905.02599v2",
    "code": "https://github.com/microsoft/horseshoe-bnn",
    "preprocessing_code": "https://github.com/YerevaNN/mimic3-benchmarks",
}

# ---------------------------------------------------------------- cohort ----
# Paper, "Mortality prediction on MIMIC-III" -> Cohort / Preprocessing.
COHORT = {
    "n_samples": 17903,
    "n_features": 17,
    "frac_survived": 0.865,
    "frac_deceased": 0.135,
    "window_hours": 48,        # features = mean over first 48h of ICU stay
    "imputation": "mean",
    "task": "in-hospital mortality",
}

# ------------------------------------------------------- Table 7: features --
# (name, unit, min_threshold, max_threshold, min_is_strict)
# max_threshold None  -> no upper bound stated
# min_is_strict True  -> paper writes "> 0" rather than "0"
FEATURES = [
    ("Height",                       "cm",       0.0,   250.0,  True),
    ("Temperature",                  "degC",    20.0,    49.0,  False),
    ("Blood pH",                     "-",        6.0,     8.0,  False),
    ("Fraction inspired oxygen",     "-",        0.0,     1.0,  False),
    ("Capillary refill rate",        "s",        0.0,    None,  False),
    ("Heart Rate",                   "bpm",      0.0,   300.0,  False),
    ("Systolic blood pressure",      "mmHg",     0.0,   275.0,  True),
    ("Diastolic blood pressure",     "mmHg",     0.0,   150.0,  True),
    ("Mean blood pressure",          "mmHg",     0.0,   190.0,  True),
    ("Weight",                       "kg",       0.0,   250.0,  False),
    ("Glucose",                      "mg/dL",    0.0,  1250.0,  False),
    ("Respiratory rate",             "/min",     0.0,   150.0,  False),
    ("Oxygen saturation",            "%",        0.0,   100.0,  True),
    ("GCS: eye opening",             "-",        1.0,     4.0,  False),
    ("GCS: motor response",          "-",        1.0,     6.0,  False),
    ("GCS: verbal response",         "-",        1.0,     5.0,  False),
    ("GCS: total",                   "-",        3.0,    15.0,  False),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
assert len(FEATURES) == COHORT["n_features"], "Table 7 must list 17 features"

# ------------------------------------------------- Table 8: hyperparameters --
HYPERPARAMS = {
    "n_weight_samples_train": 10,
    "n_weight_samples_test": 100,
    "batch_size": 64,
    "n_hidden_units": 50,
    "learning_rate": 0.001,
    "n_epochs": 5000,          # paper's stated maximum; see README on our budget
    "gaussian_prior_sd": 1.0,
    "horseshoe_b_g": 1.0,      # global shrinkage scale
    "horseshoe_b_0": 1.0,      # local (per-feature) shrinkage scale
}

CV_FOLDS = 10                  # "10 fold cross-validation"

# ------------------------------------------------------- Table 2: targets ---
# mean +/- standard error over the 10 folds.  None = not reported by the paper.
TABLE2 = {
    "LinearGaussian":  {"error_rate": (0.129, 0.003), "auroc": (0.807, 0.004), "nll": (0.321, 0.004)},
    "GaussianBNN":     {"error_rate": (0.123, 0.003), "auroc": (0.830, 0.004), "nll": (0.304, 0.004)},
    "LinearHorseshoe": {"error_rate": (0.130, 0.003), "auroc": (0.807, 0.004), "nll": (0.320, 0.004)},
    "HorseshoeBNN":    {"error_rate": (0.122, 0.002), "auroc": (0.831, 0.004), "nll": (0.304, 0.004)},
    "Lasso":           {"error_rate": (0.129, 0.002), "auroc": (0.795, 0.004), "nll": (0.325, 0.004)},
    "SVM":             {"error_rate": (0.129, 0.003), "auroc": None,           "nll": None},
    "RandomForest":    {"error_rate": (0.125, 0.002), "auroc": None,           "nll": None},
}
MODEL_ORDER = list(TABLE2)

# ------------------------------------------------------- Table 3: confusion --
# Row-normalised, rows = true label.  (mean, standard error)
# Verified self-consistent with TABLE2 error rates -- see check_consistency().
TABLE3 = {
    "LinearHorseshoe": {
        "survived": {"survived": (0.979, 0.001), "deceased": (0.021, 0.001)},
        "deceased": {"survived": (0.825, 0.008), "deceased": (0.175, 0.008)},
    },
    "HorseshoeBNN": {
        "survived": {"survived": (0.975, 0.003), "deceased": (0.025, 0.003)},
        "deceased": {"survived": (0.743, 0.018), "deceased": (0.257, 0.018)},
    },
}

# ------------------------------------------- Figure 2: reported relevance ----
# The paper's own reading of Figure 2 / its Interpretability section.  These are
# the qualitative claims our reproduction is asked to recover.
REPORTED_RELEVANCE = {
    # "irrelevant to both horseshoe models"
    "irrelevant_both": ["Height", "Capillary refill rate"],
    # "The HorseshoeBNN picks up the pH feature, whereas the LinearHorseshoe
    #  model does not" -- attributed to a non-linear (U-shaped) pH effect.
    "bnn_only": ["Blood pH", "Systolic blood pressure"],
    # "the LinearHorseshoe model captures the diastolic blood pressure only"
    "both": ["Diastolic blood pressure"],
    # "Glasgow coma scale total is selected only by ..." (sentence continues
    #  across the page break; treated as uncertain, not asserted here).
    "uncertain": ["GCS: total"],
}


def check_consistency(tol=1e-3):
    """Table 3 must imply the Table 2 error rates.  Returns {model: (implied, paper)}."""
    prev = COHORT["frac_deceased"]
    out = {}
    for model, cm in TABLE3.items():
        fpr = cm["survived"]["deceased"][0]
        fnr = cm["deceased"]["survived"][0]
        implied = (1 - prev) * fpr + prev * fnr
        paper = TABLE2[model]["error_rate"][0]
        assert abs(implied - paper) < tol, (model, implied, paper)
        out[model] = (implied, paper)
    return out


if __name__ == "__main__":
    for m, (implied, paper) in check_consistency().items():
        print(f"{m:16s} implied error {implied:.4f}  paper {paper:.3f}  OK")
    print(f"\n{len(FEATURE_NAMES)} features, n={COHORT['n_samples']}, "
          f"deceased={COHORT['frac_deceased']:.1%}")
