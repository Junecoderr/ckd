# CKD Classification — Minor Project

Machine-learning classification of Chronic Kidney Disease on the UCI CKD dataset
(400 patients), evaluated with leak-free 5-fold stratified cross-validation.

Four models are compared under five experiments, with bootstrap confidence
intervals, paired significance tests, screening-threshold selection,
calibration, fold-stability analysis and nested-CV hyperparameter tuning.

## Quick start (Google Colab — recommended for beginners)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. `File ▸ Upload notebook` → upload **`notebooks/CKD_Classification_Colab.ipynb`**
3. Run the first cell and upload `data/kidney_disease.csv` when prompted
4. `Runtime ▸ Run all`

Colab already has pandas, numpy, scikit-learn, matplotlib and seaborn — nothing to install.

## Quick start (local)

```bash
pip install -r requirements.txt

python src/ckd_pipeline.py --experiment no_outlier          # Exp 1 (primary)
python src/ckd_pipeline.py --experiment outlier_capped      # Exp 2
python src/ckd_pipeline.py --experiment ablation            # Exp 3
python src/ckd_pipeline.py --experiment missing_indicator   # Exp 4
python src/ckd_pipeline.py --experiment no_outlier --tune   # Exp 5 (nested CV)
python src/missingness_report.py                            # is missingness predictive?
PYTHONPATH=src python src/leakage_demo.py                   # leakage check
```

Or regenerate everything in `results/` in one go:

```bash
make all          # or: ./run_all.sh
make test         # 26 tests: cleaning, leakage control, metrics, statistics
```

`requirements.txt` pins the exact versions the committed results were produced
with, so `make all` reproduces them byte for byte.

## Layout

```
data/kidney_disease.csv                   raw dataset (400 rows)
notebooks/CKD_Classification_Colab.ipynb  step-by-step teaching notebook (53 cells)
src/ckd_pipeline.py                       reproducible analysis script (all experiments)
src/stats_tests.py                        bootstrap CIs, exact McNemar, Holm correction
src/missingness_report.py                 is the missing-data pattern itself predictive?
src/build_comparison.py                   assembles the cross-experiment table
src/leakage_demo.py                       measures leaky vs correct preprocessing
tests/                                    pytest suite (cleaning, leakage, metrics, drift)
results/<experiment>/                     tables (CSV) + figures (PNG) per experiment
results/missingness/                      missingness-by-outcome analysis
results/table_experiment_comparison.csv   side-by-side comparison
REPORT.md                                 methodology, results, discussion, conclusion
CITATION.md                               data source, licence, provenance and ethics
```

## Headline result (Experiment 1, pooled out-of-fold, n = 400, threshold 0.5)

| Model | Accuracy | Sensitivity | Specificity | Precision | NPV | F1 | AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9925 | 0.9920 | 0.9933 | 0.9960 | 0.9868 | 0.9940 | 0.9998 | 0.0086 |
| Decision Tree | 0.9800 | 0.9880 | 0.9667 | 0.9802 | 0.9797 | 0.9841 | 0.9773 | 0.0200 |
| Random Forest | 0.9900 | **0.9960** | 0.9800 | 0.9881 | **0.9932** | 0.9920 | 0.9997 | 0.0093 |
| SVM (RBF) | **0.9950** | 0.9920 | **1.0000** | **1.0000** | 0.9868 | **0.9960** | **1.0000** | **0.0046** |

Positive class = CKD.

**No pair of models differs significantly.** Exact McNemar tests on the paired
out-of-fold predictions, Holm-corrected across all six pairs, return
0 significant differences at α = 0.05 (smallest adjusted *p* = 0.66). The
0.005 accuracy gap between SVM and Random Forest is **two patients**. "Best
model" in this report is therefore a ranking, not a demonstrated superiority —
and 95% bootstrap CIs overlap heavily (SVM accuracy 0.9950 [0.9875, 1.0000] vs
Random Forest 0.9900 [0.9800, 0.9975]).

## Screening operating point

At the default threshold of 0.5, Random Forest still misses one CKD patient.
Selecting the threshold *inside each training fold* to minimise a stated
screening cost (a missed case weighted 5× a false alarm) removes the misses:

| Model | Threshold | Sensitivity | Specificity | FN | FP | Cost (5·FN + FP) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.396 | **1.0000** | 0.9800 | 0 | 3 | 3 (was 11) |
| Random Forest | 0.489 | 0.9960 | 0.9800 | 1 | 3 | 8 (was 8) |
| SVM (RBF) | 0.421 | **1.0000** | 0.9867 | 0 | 2 | **2** (was 10) |

The threshold is chosen by inner cross-validation on the training folds only,
so the held-out fold never influences its own operating point.

## Two findings that qualify the headline

**1. Most of the signal is the diagnosis itself.** The top predictors are the
defining diagnostic criteria of CKD. Removing all eight of them (`sc`, `bu`,
`al`, `sg`, `hemo`, `pcv`, `rc`, `rbc`) and re-running the whole pipeline
(`--experiment ablation`) drops the best model from 0.9950 to **0.9500**
accuracy and sensitivity from 0.9960 to **0.9400**. The benchmark is
near-separable because the features encode the label.

**2. Which tests were ordered predicts the outcome.** `rbc` is missing for
57.2% of CKD patients but only 6.0% of non-CKD patients; 12 of 24 features
show a significant missingness gap by outcome. A logistic regression trained
on **nothing but the 24 was-this-measured flags — no measured value at all —
reaches AUC 0.850**. A meaningful share of the benchmark's accuracy is an
artefact of how the records were assembled, and would not survive a
prospective setting where every patient gets the same panel.

Neither score would transfer to raw hospital data. See `REPORT.md` for the
full discussion.

## How data leakage is prevented

Imputation, scaling, one-hot encoding, missingness indicators, IQR fences,
hyperparameter selection and the decision threshold all live inside a
`sklearn.pipeline.Pipeline` that is refitted from scratch inside every training
fold. Held-out folds are only `.transform()`-ed, never `.fit()`-ted on, and
every fold clones a fresh estimator so no state survives between folds.
`tests/test_pipeline.py` asserts this directly: the imputer fitted on a subset
must reproduce that subset's medians and *not* the full-data medians.

## Data

UCI Chronic Kidney Disease dataset (DOI
[10.24432/C5G020](https://doi.org/10.24432/C5G020)), CC BY 4.0. Creators:
L. Rubini, P. Soundarapandian, P. Eswaran. See `CITATION.md` for provenance,
licence and the ethics note.

**This is coursework, not a medical device.** No output here should inform the
care of any patient.

## Licence

MIT — see `LICENSE`.
