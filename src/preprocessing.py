"""
The paper's preprocessing, applied to whichever cohort we hand it.

Paper: "we remove invalid feature values beyond the allowed range defined by
medical experts", "we reduce the time series for each patient by computing the
mean value of each feature in the first 48 hours", "We use mean imputation for
missing values".

The 48h reduction happens upstream (mimic3-benchmarks for real data; by
construction for synthetic data), so what is left here is the range filter,
mean imputation and standardisation.

Standardisation is a reimplementation choice: the paper does not state it, but
the horseshoe prior is scale-sensitive -- a per-feature shrinkage parameter
only means "is this feature relevant" if the features share a scale.  Without
it, Glucose (mg/dL, sd ~55) and Fraction inspired oxygen (unitless, sd ~0.15)
are not comparable and the selected feature set becomes an artefact of units.
Statistics are fit on the training fold only, never on the test fold.
"""
from __future__ import annotations

import numpy as np
from paper_spec import FEATURES


def apply_ranges(X):
    """Table 7 filter: values outside the expert-defined range become NaN."""
    X = np.asarray(X, dtype=float).copy()
    for j, (_, _, lo, hi, strict) in enumerate(FEATURES):
        col = X[:, j]
        with np.errstate(invalid="ignore"):
            bad = (col <= lo) if strict else (col < lo)
            if hi is not None:
                bad = bad | (col > hi)
        bad = bad & ~np.isnan(col)
        X[bad, j] = np.nan
    return X


class Preprocessor:
    """Mean imputation + standardisation, fit on train, applied to test."""

    def __init__(self):
        self.means_ = None
        self.scales_ = None

    def fit(self, X):
        X = apply_ranges(X)
        self.means_ = np.nanmean(X, axis=0)
        # A fully-missing column in a fold would give nan; fall back to 0.
        self.means_ = np.where(np.isnan(self.means_), 0.0, self.means_)
        Xi = self._impute(X)
        sd = Xi.std(axis=0)
        # A constant column carries no information; scaling it by ~0 would
        # blow it up into pure noise.  Leave it alone instead.
        self.scales_ = np.where(sd < 1e-8, 1.0, sd)
        self.centres_ = Xi.mean(axis=0)
        return self

    def _impute(self, X):
        Xi = X.copy()
        idx = np.where(np.isnan(Xi))
        Xi[idx] = np.take(self.means_, idx[1])
        return Xi

    def transform(self, X):
        X = apply_ranges(X)
        Xi = self._impute(X)
        return (Xi - self.centres_) / self.scales_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
