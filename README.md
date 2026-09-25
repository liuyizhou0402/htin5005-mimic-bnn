# Reproducing *Interpretable Outcome Prediction with Sparse Bayesian Neural Networks in Intensive Care*

Reproduction of Overweg, Popkes, Ercole, Li, Hernández-Lobato, Zaykov & Zhang
(2019), [arXiv:1905.02599](https://arxiv.org/abs/1905.02599).
Authors' code: [microsoft/horseshoe-bnn](https://github.com/microsoft/horseshoe-bnn).

**What is reproduced:** Table 2 (mortality prediction metrics for seven
models), Table 3 (confusion matrices), Figure 2 (learned feature relevance),
and the Appendix C.3 weight histograms.

**What the data is right now:** a synthetic cohort, not MIMIC-III. See
[Data status](#data-status).

---

## The paper in one paragraph

A Bayesian neural network is flexible but opaque. The authors make one
interpretable by putting a **tied horseshoe prior** on the first layer: every
weight leaving input feature *j* shares a single shrinkage parameter τ*ⱼ*, so
shrinking τ*ⱼ* switches that feature off as a whole. After training, reading
off which τ are near zero tells you which clinical measurements the model
considers irrelevant. They evaluate on in-hospital mortality in MIMIC-III
(17,903 ICU stays, 17 features, 13.5% mortality) and in CENTER-TBI.

## Data status

| | |
|---|---|
| Paper's cohort | MIMIC-III, 17,903 stays × 17 features, 13.5% mortality |
| Access | PhysioNet credentialed access + signed DUA, then an overnight run of [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) |
| Used here | **synthetic cohort** matching the paper's shape and Table 7 ranges |

The real pipeline is two-stage: `mimic3-benchmarks` builds the in-hospital
mortality benchmark from the raw MIMIC-III CSVs, and the authors'
`mimic_preprocessing/` reads its `listfile.csv`. Neither stage runs without
credentialed access, so this repository ships a synthetic stand-in and a
loader boundary (`synthetic_cohort.py` / `preprocessing.py`) that the real data
drops into unchanged.

### What the synthetic cohort does and does not buy you

The generator does **not** hardcode the paper's conclusions. It plants
*mechanisms* and lets the reproduction recover the conclusions — or fail to:

* Height and Capillary refill rate are given real (small) effects, then made
  95% and 99% missing, as they are in MIMIC-III. Mean imputation flattens them
  into near-constants and the signal disappears on its own. This is the same
  mechanism that makes them irrelevant in the real cohort.
* Blood pH acts through a **U-shape** — both acidosis and alkalosis are
  dangerous — so a linear model structurally cannot use it and a non-linear one
  can. This is the paper's own stated explanation for why its BNN selects pH
  and its linear model does not.
* GCS total is the sum of the three GCS sub-scores, so the design carries the
  real collinearity that a sparsity prior has to resolve.

Honest limits, all repeated in `synthetic_cohort.py` and belonging in the
report's Discussion:

* **Missingness rates are read off Figure 2's left panel by eye.** The paper
  prints no table of them.
* **Missingness here is MCAR.** In real ICU data it is informative — a blood
  gas is ordered *because* the patient is unwell. Neither the paper nor this
  reproduction models that, but on real data it matters.
* **Marginals are clinically plausible values, not MIMIC-III statistics.** The
  paper publishes no cohort-characteristics table, so unlike a Table-1-driven
  synthesis there is nothing to match them against.

Consequently: agreement with Table 2 here demonstrates that the *pipeline and
models* are right, not that the paper's numbers are right. Only the real
cohort can do the latter.

## Calibrating the synthetic signal

The paper's headline is a 0.024 AUROC gap between the linear models (0.807) and
the BNNs (0.830/0.831). A synthetic cohort is only a fair test bed if it
reproduces *that gap*, since the gap is the claim under test.

`calibrate.py` tunes two knobs against cheap proxies (logistic regression,
gradient boosting). The two are **not** independent: U-shaped signal is *noise*
to a linear model, so raising it pushes the linear AUROC down. A naive
one-after-the-other search lands both metrics below target — the first attempt
here produced linear 0.779 / non-linear 0.801. The fix is a nested search that
re-pins the linear AUROC for every candidate amount of U-shaped signal.

| | linear AUROC | non-linear AUROC |
|---|---|---|
| Paper | 0.807 | 0.831 |
| Calibrated proxies | 0.8078 | 0.8330 |

## The two-timescale problem

Predictive performance and feature selection converge at very different rates,
and this determines the whole experimental protocol.

`shrinkage_probe.py` runs a controlled test — 10 features, 3 of them real — and
tracks both quantities per epoch:

| epoch | validation NLL | relevant/irrelevant weight ratio |
|---:|---:|---:|
| 25 | 0.4450 | 2.4× |
| 200 | 0.4557 | 13.0× |
| 900 | 0.4445 | 24.9× |
| 1900 | 0.4463 | 23.7× |

**Validation NLL is flat after ~25 epochs. Feature selection is still moving
30× longer than that.** Early stopping on predictive loss therefore terminates
before the horseshoe has finished shrinking, and reporting Figure 2 from such a
run would show "the horseshoe barely shrinks anything" — an artefact of the
training budget, not a property of the model.

So the two artefacts get different budgets:

* **Table 2 / Table 3** — 10-fold CV with early stopping (`run_experiments.py`).
  This is where the predictive metrics live and where the paper reports spread.
* **Figure 2 / histograms** — one long fit on the full cohort
  (`feature_relevance.py`). The paper's Figure 2 carries no error bars, which is
  consistent with a single fit.

At convergence the contrast is the one the paper describes:

| | weight ratio | largest histogram gap | split |
|---|---:|---:|---|
| HorseshoeBNN | 23.4× | **1.40 decades** | 3 above / 7 below ✓ (truth: 3 relevant) |
| GaussianBNN | 5.9× | 0.25 decades | 5/5 — no dichotomy |

matching *"the histogram shows two groups of weights which differ by orders of
magnitude ... The clear dichotomy is absent for Gaussian models."*

## Results (final run, after the generator fix)

### Table 2 — ours [paper], 10-fold CV, mean

| Model | Error rate | AUROC | NLL |
|---|---|---|---|
| LinearGaussian | 0.123 [0.129] | 0.808 [0.807] † | 0.312 [0.321] |
| GaussianBNN | 0.119 [0.123] | 0.835 [0.830] | 0.296 [0.304] |
| LinearHorseshoe | 0.124 [0.130] | 0.808 [0.807] | 0.312 [0.320] |
| HorseshoeBNN | 0.117 [0.122] | 0.838 [0.831] † | 0.294 [0.304] |
| Lasso | 0.123 [0.129] | 0.808 [0.795] | 0.312 [0.325] |
| SVM | 0.125 [0.129] | – | – |
| RandomForest | 0.117 [0.125] | – | – |

† calibration target, not evidence. Mean |Δ| over the 15 free metrics: 0.0070.
Error rate is low by 0.0055 across all seven models; 0.0023 of that is the
cohort's mortality (13.27%) sitting below the paper's 13.5%.

### Qualitative claims

| Claim | Ours | Paper | |
|---|---|---|---|
| BNN beats linear on AUROC | +0.030 | +0.024 | reproduced (calibrated — not evidence) |
| HorseshoeBNN ≥ GaussianBNN | +0.003 | +0.001 | **reproduced** |
| LinearHorseshoe ≈ LinearGaussian | +0.000 | +0.000 | **reproduced** |
| HorseshoeBNN recalls more deaths (Table 3) | +0.047 | +0.082 | **reproduced**, smaller effect |
| Lasso worse than LinearGaussian | +0.000 | −0.012 | not reproduced |
| RandomForest worse than HorseshoeBNN | −0.001 | +0.003 | not reproduced |

The Lasso gap was investigated (`lasso_diagnostic.py`): tuning L1 on error
rate instead of AUROC explains 1.2% of it. Not supported; most likely real
MIMIC-III structure the synthetic cohort lacks.

### Figure 2 claims (pre-registered threshold 0.10 of the max norm)

| Claim | Result |
|---|---|
| C1 Height & capillary refill irrelevant to both | Height: dropped by both. Capillary refill: dropped by the BNN (0.023) but kept by the linear model at **0.121** — just over the line. Strictly: not reproduced. |
| C2 pH selected by the BNN only | BNN ranks pH **first** (1.00); linear gives it **0.111**, 9× lower but just over 0.10. Strictly: not reproduced. |
| C3 order-of-magnitude histogram gap | Toy problem: 1.40 decades vs 0.25 Gaussian. Full 17-feature cohort: 0.42 vs 0.13 — the horseshoe separates ~3× more cleanly, but not by orders of magnitude, because relevance here is a continuum. |
| Paper's drop-irrelevant check | Refit on 13 kept features: AUROC 0.8388 → 0.8376, error unchanged. **Reproduced.** |

Both C1 and C2 fail only because a value lands between 0.10 and 0.13.
The threshold was fixed before the results were seen and has deliberately
**not** been moved; at 0.15 both would pass. Report that as a sensitivity
note, not as a reproduction.

Before the generator fix, the linear model gave pH 0.788 — the leak described
in `synthetic_cohort.py` (a quadratic centred 0.02 off the mean leaked a −0.244
linear effect). After the fix it is 0.111.

### Bonus

| | Result |
|---|---|
| B1 gradient boosting | AUROC 0.839 vs HorseshoeBNN 0.837 — ties/edges the paper's model. The BNN's case rests on interpretability and uncertainty, not accuracy. |
| B2 calibration (ECE) | GradientBoosting 0.0060, HorseshoeBNN 0.0099. Both well calibrated; the Bayesian model is not the better calibrated one here. |
| B3 tune-on-test inflation | +0.0002 to +0.0007 error. Real but negligible on this cohort. |
| B4 operating point | At 0.5 the BNN catches **21.6%** of deaths (paper 25.7%). Youden-optimal threshold 0.153 → sensitivity ~0.74, specificity ~0.78, flags ~29% of patients. |

## Reimplementation choices

Everything the paper specifies is used as specified: 50 hidden units, Adam at
lr 1e-3, batch 64, 10 weight samples for training and 100 for testing, prior
sd 1.0, b₀ = b_g = 1.0 (Table 8), 10-fold CV, and the Table 7 value ranges.
Where the paper is silent or impractical:

| Choice | What the paper says | What we do, and why |
|---|---|---|
| Framework | PyTorch | TensorFlow — what was installed on this machine. The maths is unchanged. |
| Half-Cauchy KL | cites Louizos et al. (2017), who add auxiliary inverse-gamma variables to make it analytic | Keep the half-Cauchy prior exactly as specified, estimate its KL by Monte Carlo using the samples already drawn for the likelihood. Same prior, different estimator, no auxiliary variables. |
| Training length | 5000 epochs (Table 8), "trained until convergence" | Early stopping for Table 2; a fixed long budget for Figure 2. Epoch counts are logged per fold. |
| Standardisation | not mentioned | Applied, fit on the training fold only. The horseshoe is scale-sensitive: a per-feature shrinkage parameter only means "is this feature relevant" if features share a scale. Without it the selected set is an artefact of units (mg/dL vs a unitless fraction). |
| Relevance threshold | chosen by eye from the histogram gap | Fixed at 0.10 of the maximum norm **before** looking at results, so the claim checks are not tuned after the fact. |

### A flaw worth reporting

Appendix B: *"For all models the regularization strength is optimized to
minimize the test error."* The baselines' hyperparameters are selected on the
test fold. This inflates the baselines — the comparators the authors' own model
is measured against — so it runs *against* the paper's thesis rather than for
it, but it makes the reported baseline numbers optimistic. `baselines.py`
implements both the paper's protocol (`tune="test"`) and an honest nested-CV
one (`tune="nested"`), and bonus experiment **B3** reports the gap.

## Bonus experiments

| | |
|---|---|
| **B1** | A gradient-boosted comparator. The paper's baselines stop at RandomForest; boosted trees are the standard strong baseline on tabular clinical data. |
| **B2** | Calibration. The argument for a Bayesian model here is honest uncertainty, but the paper reports NLL only, which mixes discrimination and calibration together. Reported as expected calibration error plus a reliability curve. |
| **B3** | What the paper's tune-on-test protocol is worth, in error-rate points. |
| **B4** | The operating point. Table 3 shows the HorseshoeBNN correctly flags only **25.7%** of patients who die. At a 13.5% event rate a 0.5 threshold is close to meaningless clinically — a screening tool that misses three quarters of deaths would not be deployed. B4 reports what threshold selection buys and at what cost. |

## Layout

```
src/
  paper_spec.py          every constant taken from the paper, with its table
  synthetic_cohort.py    the stand-in cohort generator
  preprocessing.py       Table 7 range filter, mean imputation, standardisation
  calibrate.py           nested search for the synthetic signal strength
  bnn.py                 the four Bayesian models
  baselines.py           Lasso / LinearSVC / RandomForest, both tuning protocols
  evaluation.py          Table 2 metrics, Table 3 confusion matrices, CV harness
  shrinkage_probe.py     how long feature selection actually needs
  run_experiments.py     Table 2 + Table 3
  reproduction_report.py what reproduced, what did not, what is circular
  lasso_diagnostic.py    investigation of the non-reproduced Lasso claim
  real_data.py           loader for the real MIMIC-III cohort (untested until access)
  feature_relevance.py   Figure 2, the histograms, and the paper's claim checks
  bonus.py               B1-B4
  make_figures.py        figures for the report
data/                    synthetic_cohort.csv, the exact cohort used in every experiment
results/                 JSON output from every script
figures/                 generated figures
```

## Running it

```bash
cd src
python calibrate.py           # writes results/calibration.json  (~1 min)
python shrinkage_probe.py     # sets the epoch budget            (~4 min)
python run_experiments.py     # Table 2 + Table 3                (~20 min)
python feature_relevance.py   # Figure 2 + claim checks          (~20 min)
python bonus.py               # B1-B4                            (~10 min)
python make_figures.py
```

CPU only; no GPU is used or needed.

`paper_spec.py` runs a self-check: Table 3's confusion matrices must imply
Table 2's error rates. They do, to within 5×10⁻⁴ — which is also how the
row/column orientation of Table 3 was pinned down, since PDF extraction returns
those numbers out of order.

## Swapping in the real MIMIC-III cohort

1. Get PhysioNet credentialed access, complete CITI "Data or Specimens Only
   Research", sign the MIMIC-III DUA.
2. Run `mimic3-benchmarks` to build the in-hospital mortality benchmark.
3. Take the mean of each feature over the first 48 hours to get one row per
   stay, giving an `(n, 17)` array in the `paper_spec.FEATURE_NAMES` order plus
   a binary outcome.
4. Replace the `build_cohort()` call in `run_experiments.py` with that array.
   Everything downstream — range filter, imputation, models, metrics — is
   unchanged, because none of it knows where the numbers came from.
