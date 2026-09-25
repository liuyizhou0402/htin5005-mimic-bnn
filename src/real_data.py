"""
Load the real MIMIC-III cohort, once PhysioNet access comes through.

Pipeline the paper uses, in order:

  1. PhysioNet credentialed access + CITI "Data or Specimens Only Research"
     + the signed MIMIC-III DUA.
  2. YerevaNN/mimic3-benchmarks builds the in-hospital mortality benchmark
     from the raw MIMIC-III CSVs:

         python -m mimic3benchmark.scripts.extract_subjects      {MIMIC_CSV} data/root/
         python -m mimic3benchmark.scripts.validate_events       data/root/
         python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
         python -m mimic3benchmark.scripts.split_train_and_test  data/root/
         python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/

     That leaves per-stay time-series CSVs plus a `listfile.csv` of
     (filename, y_true) pairs.
  3. This module does the paper's own reduction: "we reduce the time series for
     each patient by computing the mean value of each feature in the first 48
     hours of the patient's stay".

The result is an (n, 17) array in paper_spec.FEATURE_NAMES order plus a binary
outcome -- exactly what synthetic_cohort.generate() produces, so everything
downstream is unchanged.

Nothing here has been run against real data yet: PhysioNet access is pending.
The column mapping below is written against mimic3-benchmarks' documented
header names and must be checked against the real listfile before use --
verify_columns() is provided for that.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from paper_spec import COHORT, FEATURE_NAMES
from synthetic_cohort import Cohort

# mimic3-benchmarks column header -> our paper_spec name.
# Its per-stay CSVs use these headers (see mimic3benchmark/resources/).
COLUMN_MAP = {
    "Height":                        "Height",
    "Temperature":                   "Temperature",
    "pH":                            "Blood pH",
    "Fraction inspired oxygen":      "Fraction inspired oxygen",
    "Capillary refill rate":         "Capillary refill rate",
    "Heart Rate":                    "Heart Rate",
    "Systolic blood pressure":       "Systolic blood pressure",
    "Diastolic blood pressure":      "Diastolic blood pressure",
    "Mean blood pressure":           "Mean blood pressure",
    "Weight":                        "Weight",
    "Glucose":                       "Glucose",
    "Respiratory rate":              "Respiratory rate",
    "Oxygen saturation":             "Oxygen saturation",
    "Glascow coma scale eye opening":    "GCS: eye opening",
    "Glascow coma scale motor response": "GCS: motor response",
    "Glascow coma scale verbal response": "GCS: verbal response",
    "Glascow coma scale total":          "GCS: total",
}

# The GCS columns are categorical strings in mimic3-benchmarks
# ("4 Spontaneously", "6 Obeys Commands", ...).  The paper treats them as the
# ordinal scores of Table 7, so we take the leading integer.
GCS_COLUMNS = {"GCS: eye opening", "GCS: motor response",
               "GCS: verbal response", "GCS: total"}


def _to_numeric(series, is_gcs):
    if is_gcs:
        extracted = series.astype(str).str.extract(r"^\s*(\d+)")[0]
        return pd.to_numeric(extracted, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def verify_columns(episode_csv):
    """
    Check a real per-stay CSV against COLUMN_MAP before trusting the loader.
    Returns (found, missing, unexpected).  Run this first -- header names have
    drifted between mimic3-benchmarks releases.
    """
    cols = set(pd.read_csv(episode_csv, nrows=1).columns)
    found = {c for c in COLUMN_MAP if c in cols}
    missing = set(COLUMN_MAP) - found
    unexpected = cols - set(COLUMN_MAP) - {"Hours"}
    return sorted(found), sorted(missing), sorted(unexpected)


def load(benchmark_dir, window_hours=None, limit=None, strict=True):
    """
    Read a mimic3-benchmarks in-hospital-mortality directory into a Cohort.

    benchmark_dir must contain listfile.csv and the per-stay episode CSVs.
    """
    window_hours = window_hours or COHORT["window_hours"]
    root = pathlib.Path(benchmark_dir)
    listfile = root / "listfile.csv"
    if not listfile.exists():
        raise FileNotFoundError(
            f"{listfile} not found. Run mimic3-benchmarks' "
            "create_in_hospital_mortality step first (see module docstring).")

    index = pd.read_csv(listfile)
    if limit:
        index = index.head(limit)

    rows, labels, skipped = [], [], 0
    for _, r in index.iterrows():
        path = root / r["stay"]
        if not path.exists():
            skipped += 1
            continue
        df = pd.read_csv(path)
        if "Hours" in df.columns:
            df = df[df["Hours"] <= window_hours]
        if df.empty:
            skipped += 1
            continue

        vals = np.full(len(FEATURE_NAMES), np.nan)
        for src, dst in COLUMN_MAP.items():
            if src not in df.columns:
                continue
            col = _to_numeric(df[src], dst in GCS_COLUMNS)
            if col.notna().any():
                vals[FEATURE_NAMES.index(dst)] = float(col.mean())
        rows.append(vals)
        labels.append(int(r["y_true"]))

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)

    if strict and len(y):
        n_paper, rate_paper = COHORT["n_samples"], COHORT["frac_deceased"]
        if abs(len(y) - n_paper) > 0.05 * n_paper:
            print(f"  ! cohort size {len(y)} differs from the paper's {n_paper} "
                  f"by more than 5% -- check the benchmark build")
        if abs(y.mean() - rate_paper) > 0.02:
            print(f"  ! mortality {y.mean():.3f} differs from the paper's "
                  f"{rate_paper:.3f} -- check the label column")
    if skipped:
        print(f"  skipped {skipped} stays with no usable data in the "
              f"first {window_hours}h")

    return Cohort(X=X, y=y, X_true=X.copy(), logits=np.full(len(y), np.nan))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_dir",
                    help="mimic3-benchmarks in-hospital-mortality directory")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    c = load(args.benchmark_dir, limit=args.limit)
    print(f"loaded n={len(c.y)}  features={c.X.shape[1]}  "
          f"mortality={c.y.mean():.3%}")
    print(f"missing per feature:")
    for j, n in enumerate(FEATURE_NAMES):
        print(f"  {n:28s} {np.isnan(c.X[:, j]).mean():6.1%}")
