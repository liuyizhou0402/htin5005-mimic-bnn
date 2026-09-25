HTIN5005 Assignment 1 -- Reproduction of Overweg et al. (2020)
"Interpretable Outcome Prediction with Sparse Bayesian Neural Networks in Intensive Care"

Team: YIZHOU LIU (SID 510109165), Jennifer Zhao (SID 312019661)

----------------------------------------------------------------------
REQUIREMENTS
----------------------------------------------------------------------
Python 3.10 or 3.11.  CPU only -- no GPU is used or required.

    python -m venv .venv
    source .venv/bin/activate          (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt

----------------------------------------------------------------------
RUN, IN ORDER  (from the src/ directory)
----------------------------------------------------------------------
    cd src
    python calibrate.py                                  ~1 min
    python shrinkage_probe.py                            ~4 min
    python run_experiments.py                           ~19 min   -> report Tables 4, 6
    python reproduction_report.py                        <1 min   -> report Table 5
    python lasso_diagnostic.py                           <1 min   -> report Section B.3.3
    python feature_relevance.py --epochs 250 --batch-size 64   ~15 min -> report Figure 1
    python bonus.py --folds 5 --batch-size 64            ~8 min   -> report Table 7, Section B.4
    python make_figures.py                               <1 min   -> figures/

Total: about 45 minutes.  Every script writes JSON to results/.
calibrate.py must run first: every later script reads results/calibration.json.

----------------------------------------------------------------------
DATASET
----------------------------------------------------------------------
data/synthetic_cohort.csv is the exact cohort used in every experiment
(17,903 stays, 17 features, 13.27% in-hospital mortality).  It is also
regenerated deterministically from seed 0 by src/synthetic_cohort.py, so
no download is required.

The real MIMIC-III cohort needs PhysioNet credentialed access.  Once
available, see src/real_data.py; the substitution is one function call
in run_experiments.build_cohort().

----------------------------------------------------------------------
SOURCE CODE ATTRIBUTION
----------------------------------------------------------------------
All code in src/ was written by the team.  The authors' reference
implementation, https://github.com/microsoft/horseshoe-bnn (PyTorch, MIT
licence), was consulted for the layer structure and hyperparameters but
no code was copied from it.  The MIMIC-III preprocessing pipeline,
https://github.com/YerevaNN/mimic3-benchmarks, is cited in
src/real_data.py and was not executed.
