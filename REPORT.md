# Chronic Kidney Disease Classification using Machine Learning
### Minor Project — Results Report

All numbers in this report were produced by `src/ckd_pipeline.py` on the supplied
`kidney_disease.csv` (400 patients). Nothing is estimated or illustrative.
Reproduce with:

```bash
make all          # every experiment, or run them individually:

python src/ckd_pipeline.py --experiment no_outlier          # Exp 1 (primary)
python src/ckd_pipeline.py --experiment outlier_capped      # Exp 2
python src/ckd_pipeline.py --experiment ablation            # Exp 3
python src/ckd_pipeline.py --experiment missing_indicator   # Exp 4
python src/ckd_pipeline.py --experiment no_outlier --tune   # Exp 5 (nested CV)
python src/missingness_report.py                            # missingness analysis
```

`make test` runs 26 tests covering the cleaning rules, the leakage guarantees, the metric
definitions and the statistical helpers.

---

## 1. Methodology

### 1.1 Dataset

UCI Chronic Kidney Disease dataset: **400 patient records**, 24 clinical attributes
(after dropping the `id` column) and one binary target, `classification`.

| | |
|---|---|
| Total records | 400 |
| Features used | 24 (14 numeric, 10 categorical) |
| Target | `classification` — **ckd = 1 (positive)**, **notckd = 0** |
| Class distribution | **250 CKD (62.5%)** / **150 notckd (37.5%)** |
| Duplicate records | 0 |

CKD was assigned the **positive** class because in a screening setting the disease is
the event to be detected; this makes Sensitivity mean *"of all true CKD patients, how
many did the model catch?"*

**Numeric (14):** `age`, `bp`, `sg`, `al`, `su`, `bgr`, `bu`, `sc`, `sod`, `pot`,
`hemo`, `pcv`, `wc`, `rc`
**Categorical (10):** `rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`

`sg`, `al` and `su` were kept numeric rather than one-hot encoded because they are
*graded* clinical readings whose ordering carries clinical meaning.

### 1.2 Data cleaning — four defects found in the raw file

Inspection of the raw CSV with `repr()` (which makes invisible characters visible)
revealed four concrete data-quality problems:

| # | Problem | Evidence | Consequence if ignored |
|---|---|---|---|
| 1 | Target had **3** distinct values | `'ckd'`, `'notckd'`, **`'ckd\t'`** (2 records) | `'ckd\t'` becomes a spurious third class |
| 2 | `dm` had **5** values instead of 2 | `'yes'`, `'no'`, `'\tyes'`, `'\tno'`, `' yes'` | One-hot encoder creates 5 bogus columns |
| 3 | `cad` had **3** values instead of 2 | `'yes'`, `'no'`, `'\tno'` | Same as above |
| 4 | `pcv`, `wc`, `rc` loaded as **text** | contain the literal string `'\t?'` | Three genuinely numeric predictors unusable |

All four were repaired by: stripping whitespace/tabs from every text cell, replacing
`'?'` and `''` with `NaN`, coercing the 14 numeric columns with
`pd.to_numeric(..., errors="coerce")`, lower-casing categories, and mapping the target
to {1, 0}.

**This cleaning is structural, not statistical** — it uses no mean, median, or range
from the data — so performing it before splitting cannot cause leakage.

### 1.3 Missing values

**1,012 missing cells (10.54% of all values). 242 of 400 rows (60.5%) have at least one
missing value; only 158 rows are complete.**

| Feature | Missing | % | Feature | Missing | % |
|---|---|---|---|---|---|
| `rbc` | 152 | 38.0 | `sg` | 47 | 11.8 |
| `rc` | 131 | 32.8 | `al` | 46 | 11.5 |
| `wc` | 106 | 26.5 | `bgr` | 44 | 11.0 |
| `pot` | 88 | 22.0 | `bu` | 19 | 4.8 |
| `sod` | 87 | 21.8 | `sc` | 17 | 4.2 |
| `pcv` | 71 | 17.8 | `bp` | 12 | 3.0 |
| `pc` | 65 | 16.2 | `age` | 9 | 2.2 |
| `hemo` | 52 | 13.0 | `pcc`, `ba` | 4 | 1.0 |
| `su` | 49 | 12.2 | `htn`, `dm`, `cad` | 2 | 0.5 |
| | | | `appet`, `pe`, `ane` | 1 | 0.2 |

**Listwise deletion was rejected.** Dropping incomplete rows would discard 242 of 400
records (60.5%) and leave only 158 — and the loss would not be random, since sicker
patients receive more laboratory tests. Imputation was used instead:

* **numeric → median** (robust to extreme values, which matters because Experiment 1
  deliberately retains all outliers)
* **categorical → mode** (most frequent category)

### 1.4 Encoding

One-hot encoding (`OneHotEncoder(handle_unknown="ignore", drop="if_binary")`). All ten
categorical variables are binary, so each produces a single 0/1 column (e.g. `htn` →
`htn_yes`), avoiding redundant columns. `handle_unknown="ignore"` prevents a crash if a
category appears only in a held-out fold. One-hot was preferred over integer label
encoding because the latter would impose a false ordering on unordered categories.

### 1.5 Scaling

`StandardScaler` (zero mean, unit variance), required by Logistic Regression and the
RBF-kernel SVM.

### 1.6 Leakage control

**Every quantity that is learned from data** — imputation medians and modes, scaler
means and standard deviations, one-hot category lists, the IQR fences of Experiment 2,
the missing-value indicators of Experiment 4, the hyperparameters of Experiment 5 and
the decision threshold of §1.11 — **is learned inside a `sklearn.pipeline.Pipeline` that
is refitted from scratch within every training fold.** For each of the 5 folds, `.fit()`
sees only the 4 training folds; the held-out fold is merely `.transform()`-ed using those
training-derived statistics. Every fold also builds a fresh estimator via
`sklearn.base.clone`, so no fitted state survives from one fold to the next. No
information from a held-out fold ever influences anything applied to it.

This is asserted mechanically rather than by inspection: `tests/test_pipeline.py`
fits the preprocessor on a subset and checks that the imputer's learned medians equal
that subset's medians and are **not** the full-data medians, and that an `IQRWinsorizer`
fitted on training rows caps unseen extreme values at the *training* fences.

The common alternative — imputing and scaling the full dataset before calling
`cross_val_score` — is a leak, because the medians and scaler parameters would be
computed partly from the test rows.

### 1.7 Validation design

5-fold **Stratified** cross-validation, `shuffle=True`, `random_state=42`.

* **Stratified** because the data are imbalanced (62.5% / 37.5%); stratification holds
  that ratio in every fold, keeping Specificity estimates stable. Each test fold
  contained exactly 50 CKD and 30 notckd patients.
* **`shuffle=True`** is essential because the raw file is **sorted by class** (all CKD
  records first). Without shuffling the folds would be severely unbalanced.
* Cross-validation rather than a single split because with n=400 a single 80/20 split
  yields an 80-patient test set whose score swings substantially with the random seed.

Two complementary reporting styles are used: **per-fold mean ± SD** (shows stability)
and **pooled out-of-fold** (every patient predicted exactly once while held out, giving
one honest confusion matrix over all 400 records).

A single 5-fold split still rests on one particular shuffle. To separate the signal from
the seed, every experiment is additionally run with
`RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` — **50 independent test sets** — and
the spread of each metric is reported in §2.8.

### 1.8 Algorithms

Logistic Regression (`max_iter=5000`), Decision Tree, Random Forest (300 trees),
and SVM with RBF kernel (`probability=True`). All at scikit-learn defaults otherwise,
with `random_state=42`.

Because "defaults" is itself a modelling choice, a fifth experiment re-runs the primary
analysis under **nested cross-validation**: inside each of the 5 outer training folds a
`GridSearchCV` with its own inner 5-fold split selects hyperparameters on ROC-AUC, and
only then is the outer held-out fold scored. The outer fold never participates in
selection, so the resulting estimate is unbiased. Search spaces are deliberately small —
with 400 patients a large grid mostly fits noise:

| Model | Grid |
|---|---|
| Logistic Regression | `C` ∈ {0.01, 0.1, 1, 10, 100}; `class_weight` ∈ {None, balanced} |
| Decision Tree | `max_depth` ∈ {2, 3, 5, 8, None}; `min_samples_leaf` ∈ {1, 3, 5, 10}; `criterion` ∈ {gini, entropy} |
| Random Forest | `max_depth` ∈ {None, 5, 10}; `min_samples_leaf` ∈ {1, 2, 4}; `max_features` ∈ {sqrt, 0.5} |
| SVM (RBF) | `C` ∈ {0.1, 1, 10, 100}; `gamma` ∈ {scale, 0.01, 0.1}; `class_weight` ∈ {None, balanced} |

### 1.9 Metrics

With CKD = 1 as positive and `labels=[0, 1]` fixing the matrix orientation:

|  | Predicted notckd | Predicted ckd |
|---|---|---|
| **Actual notckd (0)** | TN | FP |
| **Actual ckd (1)** | FN | TP |

$$\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}\quad
\text{Sensitivity}=\frac{TP}{TP+FN}\quad
\text{Specificity}=\frac{TN}{TN+FP}$$

$$\text{Precision}=\frac{TP}{TP+FP}\quad
\text{NPV}=\frac{TN}{TN+FN}\quad
F_1=2\cdot\frac{\text{Precision}\times\text{Sensitivity}}{\text{Precision}+\text{Sensitivity}}$$

**NPV** (negative predictive value) answers the question a screening programme actually
asks of a negative result: *given that this patient was cleared, what is the chance they
are truly free of CKD?* It was computed but omitted from the summary tables of the first
version of this report; it is reported throughout here.

**Brier score** is the mean squared error of the predicted probabilities,
$\frac{1}{n}\sum(p_i - y_i)^2$ — lower is better. A model can rank patients perfectly
(AUC 1.0) while its probabilities are badly miscalibrated, which matters as soon as an
output is shown to a clinician as a risk rather than a label. See §2.7.

**AUC** is the area under the ROC curve (Sensitivity plotted against 1 − Specificity
across all decision thresholds). It equals the probability that the model assigns a
randomly chosen CKD patient a higher risk score than a randomly chosen healthy person,
and is threshold-independent.

### 1.10 Quantifying uncertainty, and comparing models honestly

Point estimates alone cannot support a claim that one model beats another. Two
instruments are used.

**Bootstrap confidence intervals.** The 400 pooled out-of-fold predictions are resampled
with replacement 2,000 times and every metric recomputed, giving percentile 95% CIs.
Stated plainly, this quantifies the uncertainty of *evaluating* a fixed set of
predictions on this cohort; it does not capture the extra variability from refitting on a
different sample of patients, so it is a **lower bound** on total uncertainty.

**Exact McNemar tests.** Every model produces exactly one out-of-fold prediction per
patient, so the four models are evaluated on *the same* 400 patients — paired data. Only
the patients on which two models disagree carry information: if $n_{ab}$ is the count
that only model B got right and $n_{ba}$ the count that only A got right, then under the
null hypothesis of equal error rates $n_{ba} \sim \text{Binomial}(n_{ab}+n_{ba},\,0.5)$.
The **exact binomial** form is used rather than the chi-square approximation because the
discordant counts here are single digits. Comparing four models means six tests, so
p-values are **Holm-Bonferroni** corrected.

### 1.11 Choosing the decision threshold

Reporting metrics at a probability cut-off of 0.5 embeds an assumption that a false
negative and a false positive cost the same. In CKD screening they do not: a missed case
delays treatment of a progressive disease, while a false alarm costs one confirmatory
test. This report therefore also selects the threshold that minimises an explicit,
**stated** cost,

$$\text{cost} = 5 \times FN + 1 \times FP,$$

where the 5:1 weighting is a declared assumption, not a quantity read from the data
(the two constants sit at the top of `src/ckd_pipeline.py` and can be changed).

Crucially the threshold is a *learned* quantity and is treated like every other one: it
is selected by an **inner cross-validation of the training folds only**, then applied
unchanged to the held-out fold. The held-out fold never influences its own operating
point. Results at both 0.5 and the selected threshold are reported side by side.

---

## 2. Results

### 2.1 Experiment 1 — without outlier removal (primary experiment)

**Table 1. Pooled out-of-fold performance, 5-fold Stratified CV (n = 400).**

| Model | Accuracy | Sensitivity | Specificity | Precision | NPV | F1 | AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9925 | 0.9920 | 0.9933 | 0.9960 | 0.9868 | 0.9940 | 0.9998 | 0.0086 |
| Decision Tree | 0.9800 | 0.9880 | 0.9667 | 0.9802 | 0.9797 | 0.9841 | 0.9773 | 0.0200 |
| Random Forest | 0.9900 | **0.9960** | 0.9800 | 0.9881 | **0.9932** | 0.9920 | 0.9997 | 0.0093 |
| **SVM (RBF)** | **0.9950** | 0.9920 | **1.0000** | **1.0000** | 0.9868 | **0.9960** | **1.0000** | **0.0046** |

Read the NPV column alongside Sensitivity: Random Forest's NPV of 0.9932 means that of
the patients it cleared, 99.3% were genuinely CKD-free. Logistic Regression and the SVM
both sit at 0.9868 because each missed two CKD patients.

**Table 2. Pooled confusion-matrix counts (out of all 400 patients).**

| Model | TP | TN | FP | FN |
|---|---|---|---|---|
| Logistic Regression | 248 | 149 | 1 | 2 |
| Decision Tree | 247 | 145 | 5 | 3 |
| Random Forest | 249 | 147 | 3 | **1** |
| SVM (RBF) | 248 | **150** | **0** | 2 |

**Table 3. Stability across the 5 folds (mean ± SD).**

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 0.9925 ± 0.0112 | 1.0000 ± 0.0000 |
| Decision Tree | 0.9800 ± 0.0168 | 0.9773 ± 0.0198 |
| Random Forest | 0.9900 ± 0.0105 | 0.9999 ± 0.0003 |
| SVM (RBF) | 0.9950 ± 0.0068 | 1.0000 ± 0.0000 |

**Table 3b. 95% bootstrap confidence intervals (2,000 resamples of the pooled
out-of-fold predictions).**

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.9925 [0.9825, 1.0000] | 0.9920 [0.9795, 1.0000] | 0.9933 [0.9787, 1.0000] | 0.9998 [0.9994, 1.0000] |
| Decision Tree | 0.9800 [0.9650, 0.9925] | 0.9880 [0.9724, 1.0000] | 0.9667 [0.9351, 0.9932] | 0.9773 [0.9598, 0.9925] |
| Random Forest | 0.9900 [0.9800, 0.9975] | 0.9960 [0.9873, 1.0000] | 0.9800 [0.9545, 1.0000] | 0.9997 [0.9989, 1.0000] |
| SVM (RBF) | 0.9950 [0.9875, 1.0000] | 0.9920 [0.9795, 1.0000] | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |

Every interval overlaps every other one except on Specificity, where the SVM's zero
false positives give it a degenerate interval. The apparent ordering of the models is
well inside sampling noise; §2.5 tests it directly.

**Worked example — reading the Random Forest confusion matrix:**
TP = 249, TN = 147, FP = 3, FN = 1.
Sensitivity = 249/(249+1) = **0.9960**; Specificity = 147/(147+3) = **0.9800**;
Precision = 249/(249+3) = **0.9881**; F1 = 2(0.9881×0.9960)/(0.9881+0.9960) = **0.9920**.

**Figures** (in `results/no_outlier/`): `missing_values.png`,
`confusion_matrix_no_outlier_*.png` (and `*_threshold.png` at the selected operating
point), `roc_curve_no_outlier_*.png` (5 fold curves plus mean ROC with ±1 SD band),
`model_comparison_no_outlier.png`, `feature_importance_no_outlier.png`,
`permutation_importance_no_outlier.png`, `calibration_no_outlier.png`,
`threshold_no_outlier_*.png`, `stability_no_outlier_accuracy.png` and
`stability_no_outlier_sensitivity.png`.

### 2.2 Feature importance (Random Forest)

| Rank | Feature | Importance | Rank | Feature | Importance |
|---|---|---|---|---|---|
| 1 | `hemo` (haemoglobin) | 0.2298 | 6 | `htn_yes` (hypertension) | 0.0525 |
| 2 | `pcv` (packed cell volume) | 0.1571 | 7 | `al` (albumin) | 0.0513 |
| 3 | `sc` (serum creatinine) | 0.1308 | 8 | `dm_yes` (diabetes) | 0.0373 |
| 4 | `sg` (specific gravity) | 0.1176 | 9 | `sod` (sodium) | 0.0244 |
| 5 | `rc` (red cell count) | 0.0845 | 10 | `bgr` (blood glucose) | 0.0244 |

Gini importance is computed on training data and is biased towards features with many
split points, so it is reported here as **descriptive only**. The out-of-sample
counterpart is permutation importance: permute one raw column in the held-out fold and
measure how far the AUC falls, averaged over the 5 folds.

**Table 3c. Out-of-fold permutation importance (mean drop in held-out AUC ± SD).**

| Rank | Feature | AUC drop | Rank | Feature | AUC drop |
|---|---|---|---|---|---|
| 1 | `sg` | 0.00937 ± 0.00546 | 6 | `htn` | 0.00052 ± 0.00074 |
| 2 | `hemo` | 0.00263 ± 0.00286 | 7 | `bu` | 0.00039 ± 0.00052 |
| 3 | `sc` | 0.00162 ± 0.00179 | 8 | `bgr` | 0.00039 ± 0.00062 |
| 4 | `al` | 0.00086 ± 0.00135 | 9 | `pcv` | 0.00038 ± 0.00049 |
| 5 | `dm` | 0.00059 ± 0.00084 | 10 | `sod` | 0.00027 ± 0.00030 |

The two rankings broadly agree on which measurements matter, but the *magnitudes* are
revealing: permuting even the strongest feature costs under 0.01 AUC. Because the
diagnostic markers are highly redundant with one another, removing any single one barely
hurts — the model simply reads the diagnosis off a different column. §2.10 removes them
all at once, which is the only way to see the effect.

### 2.3 Experiment 2 — with outlier treatment

Outliers were **capped (winsorised)** at the 1.5×IQR fences rather than deleted, for
three reasons: (i) clinically, the extreme values are the sickest patients (serum
creatinine up to 76 mg/dl, blood urea up to 391 mg/dl) — exactly the cases a CKD
detector must find; (ii) methodologically, removing rows would change the test set and
make the two experiments incomparable; (iii) the IQR fences are learned statistics, so
they were implemented as a pipeline transformer fitted on training folds only.

Outliers detected by the 1.5×IQR rule: `su` 61, `sc` 51, `bu` 38, `bp` 36, `bgr` 34,
`sod` 16, `age` 10, `wc` 10, `pot` 4, `hemo`/`pcv`/`rc` 1 each, `sg`/`al` 0.

**Table 4. Experiment 1 vs Experiment 2 (pooled out-of-fold).**

| Model | Metric | Exp 1 (no removal) | Exp 2 (IQR capped) | Δ |
|---|---|---|---|---|
| Logistic Regression | Accuracy | 0.9925 | 0.9875 | −0.0050 |
| | Sensitivity | 0.9920 | 0.9880 | −0.0040 |
| | AUC | 0.9998 | 0.9998 | 0.0000 |
| Decision Tree | Accuracy | 0.9800 | 0.9750 | −0.0050 |
| | Sensitivity | 0.9880 | 0.9800 | −0.0080 |
| | AUC | 0.9773 | 0.9733 | −0.0040 |
| Random Forest | Accuracy | 0.9900 | 0.9900 | 0.0000 |
| | Sensitivity | 0.9960 | 0.9960 | 0.0000 |
| | AUC | 0.9997 | 0.9998 | +0.0001 |
| SVM (RBF) | Accuracy | 0.9950 | 0.9950 | 0.0000 |
| | Sensitivity | 0.9920 | 0.9920 | 0.0000 |
| | AUC | 1.0000 | 0.9998 | −0.0002 |

**Outlier capping did not improve any model.** It left Random Forest and SVM unchanged
and slightly *degraded* Logistic Regression (−0.0050 accuracy) and the Decision Tree
(−0.0050 accuracy, −0.0080 sensitivity). **Experiment 1, without outlier removal, is
therefore retained as the primary result.**

### 2.4 Leakage verification

The wrong approach (preprocessing the full dataset once, then cross-validating only the
model) was measured against the correct approach (preprocessing inside the pipeline),
using Logistic Regression:

| Approach | AUC | Accuracy |
|---|---|---|
| Preprocess before CV (leaky) | 1.0000 | 0.9925 |
| Preprocess inside CV (correct) | 1.0000 | 0.9925 |
| **Optimistic bias** | **+0.0000** | **+0.0000** |

On this dataset leakage produced **no measurable inflation** — see the Discussion for why
this must not be read as evidence that leakage is harmless.

### 2.5 Are the models actually different?

The first version of this report named a "best model" on a 0.005 accuracy gap. That gap
is **two patients**. Exact McNemar tests on the paired out-of-fold predictions settle the
question.

**Table 5. Pairwise exact McNemar tests, Holm-corrected across all six comparisons.**

| Model A | Model B | Discordant | Only A right | Only B right | *p* | *p* (Holm) | Significant |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Decision Tree | 9 | 7 | 2 | 0.1797 | 0.8984 | no |
| Logistic Regression | Random Forest | 3 | 2 | 1 | 1.0000 | 1.0000 | no |
| Logistic Regression | SVM (RBF) | 5 | 2 | 3 | 1.0000 | 1.0000 | no |
| Decision Tree | Random Forest | 6 | 1 | 5 | 0.2188 | 0.8984 | no |
| Decision Tree | SVM (RBF) | 10 | 2 | 8 | 0.1094 | 0.6562 | no |
| Random Forest | SVM (RBF) | 6 | 2 | 4 | 0.6875 | 1.0000 | no |

**No pair of models differs significantly at α = 0.05**; the smallest corrected p-value
is 0.66. Even the widest gap in the study — Decision Tree versus SVM, 0.015 accuracy —
rests on ten disagreements split 8:2, which a fair coin produces one time in nine.

This does not mean the models are identical; it means **400 patients at this accuracy
cannot tell them apart**. Any ranking in this report is therefore a ranking, not a
demonstrated superiority. Detecting a difference this small would need a cohort roughly
an order of magnitude larger.

### 2.6 Choosing an operating point for screening

At the default 0.5 cut-off, Random Forest still misses one CKD patient and Logistic
Regression and the SVM miss two each. Selecting the threshold inside each training fold
to minimise 5·FN + FP (§1.11) changes that.

**Table 6. Pooled out-of-fold performance at 0.5 versus at the fold-selected threshold.**

| Model | Threshold | Sensitivity | Specificity | NPV | FN | FP | Cost | Cost at 0.5 |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.396 | **1.0000** | 0.9800 | **1.0000** | **0** | 3 | 3 | 11 |
| Decision Tree | 1.000 | 0.9880 | 0.9667 | 0.9797 | 3 | 5 | 20 | 20 |
| Random Forest | 0.489 | 0.9960 | 0.9800 | 0.9932 | 1 | 3 | 8 | 8 |
| SVM (RBF) | 0.421 | **1.0000** | 0.9867 | **1.0000** | **0** | 2 | **2** | 10 |

Logistic Regression and the SVM reach **sensitivity 1.000 — every one of the 250 CKD
patients detected — at a cost of three and two false alarms respectively**. Under the
stated cost model the SVM's screening cost falls from 10 to 2 and Logistic Regression's
from 11 to 3.

Two honest caveats. First, the thresholds move only a little (0.40–0.49 rather than 0.5),
which is what one expects when the classes are near-separable: there is a wide band of
thresholds where almost nothing changes. Second, the **Decision Tree cannot be tuned this
way at all** — its leaves are nearly pure, so its predicted probabilities are almost all
exactly 0 or 1 and no intermediate cut-off exists. Its selected threshold of 1.000 is
degenerate and leaves its predictions unchanged. Probability-based threshold tuning
requires a model that emits graded probabilities.

### 2.7 Calibration

AUC measures ranking; it says nothing about whether a predicted 0.8 corresponds to an
80% chance of CKD. That distinction matters the moment an output is presented to a
clinician as a risk. Brier scores (lower is better, §1.9) and the calibration curves in
`results/no_outlier/calibration_no_outlier.png`:

| Model | Brier |
|---|---|
| SVM (RBF) | **0.0046** |
| Logistic Regression | 0.0086 |
| Random Forest | 0.0093 |
| Decision Tree | 0.0200 |

All four are low, which is unsurprising on a near-separable problem where almost every
prediction is confident and correct. The Decision Tree is worst by a factor of four: its
probabilities are essentially hard 0/1 votes, so every one of its 8 errors contributes a
full unit of squared error. Note that the SVM's probabilities do not come from its
decision function directly but from Platt scaling fitted on an internal split — the
calibration is a property of that extra fitting step as much as of the SVM.

### 2.8 How much of the result is the random seed?

Every number above rests on one shuffle of 400 patients into 5 folds. Repeating the
whole split 10 times gives 50 independent test sets.

**Table 7. Accuracy and Sensitivity across 50 repeated stratified folds.**

| Model | Accuracy mean ± SD | Accuracy min–max | Sensitivity mean ± SD | Sensitivity min |
|---|---|---|---|---|
| Logistic Regression | 0.9938 ± 0.0095 | 0.9625 – 1.0000 | 0.9944 ± 0.0121 | 0.94 |
| SVM (RBF) | 0.9930 ± 0.0076 | 0.9750 – 1.0000 | 0.9892 ± 0.0123 | 0.96 |
| Random Forest | 0.9918 ± 0.0097 | 0.9750 – 1.0000 | **0.9964 ± 0.0078** | **0.98** |
| Decision Tree | 0.9728 ± 0.0215 | 0.8875 – 1.0000 | 0.9764 ± 0.0264 | 0.88 |

The ordering of the top three **changes** under repetition: Logistic Regression, not the
SVM, has the highest mean accuracy across 50 folds, and the three are separated by 0.002
against a standard deviation of 0.008–0.010. This is the same conclusion §2.5 reached by
a different route. Random Forest does retain the best mean sensitivity *and* the best
worst case (0.98, versus 0.94 for Logistic Regression and 0.96 for the SVM), which is the
property that matters for screening. The Decision Tree is both worse and markedly less
stable, with one fold falling to 0.8875 accuracy.

### 2.9 Experiment 5 — nested cross-validation

Hyperparameters were left at library defaults throughout §2.1–2.8. Nested CV (§1.8)
tests whether that cost anything.

**Table 8. Defaults versus nested-CV tuning (pooled out-of-fold).**

| Model | Accuracy (default) | Accuracy (tuned) | AUC (default) | AUC (tuned) |
|---|---|---|---|---|
| Logistic Regression | 0.9925 | 0.9925 | 0.9998 | 0.9995 |
| Decision Tree | 0.9800 | 0.9700 | 0.9773 | 0.9829 |
| Random Forest | 0.9900 | 0.9900 | 0.9997 | 0.9996 |
| SVM (RBF) | 0.9950 | 0.9875 | 1.0000 | 0.9999 |

**Unbiased tuning did not improve a single model, and slightly degraded two.** This is
the expected result when defaults already sit near the ceiling: the inner search selects
on 320 patients' worth of noisy AUC differences, and the selection itself is variable.
The per-fold choices in `results/no_outlier_tuned/table_tuned_params_*.csv` show it
directly — Logistic Regression's `C` jumps between 0.1, 1 and 100 across the five outer
folds, which is a search finding nothing to hold on to. Only the SVM is stable
(`C = 0.1`, `gamma = scale` in all five folds), and that setting scores slightly worse
than the default.

The value of this experiment is not a better model but a closed question: the earlier
report's "hyperparameters left at library defaults" limitation cost nothing measurable.

### 2.10 Experiment 3 — removing the diagnostic criteria

The original report warned that the top predictors *are* the defining diagnostic
criteria of CKD, so the benchmark is near-separable. That was an assertion. This
experiment tests it by deleting all eight such features — serum creatinine (`sc`), blood
urea (`bu`), albumin (`al`), specific gravity (`sg`), haemoglobin (`hemo`), packed cell
volume (`pcv`), red-cell count (`rc`) and red blood cells in urine (`rbc`) — and
re-running the entire pipeline on the remaining 16 features (history, blood pressure,
glucose, electrolytes, white-cell count, and the categorical comorbidities).

Clinically, `sc` and `al` are the KDIGO definition of CKD: reduced glomerular filtration
rate, computed from serum creatinine, or markers of kidney damage, chiefly albuminuria.
`bu` and `sg` are the other direct renal read-outs, and `hemo`/`pcv`/`rc`/`rbc` track the
anaemia and haematuria of established disease.

**Table 9. All features versus diagnostic criteria removed (pooled out-of-fold).**

| Model | Accuracy | | Sensitivity | | AUC | |
|---|---|---|---|---|---|---|
| | all | ablated | all | ablated | all | ablated |
| Logistic Regression | 0.9925 | 0.9200 | 0.9920 | 0.8840 | 0.9998 | 0.9659 |
| Decision Tree | 0.9800 | 0.8775 | 0.9880 | 0.8720 | 0.9773 | 0.8793 |
| Random Forest | 0.9900 | **0.9500** | 0.9960 | **0.9400** | 0.9997 | **0.9893** |
| SVM (RBF) | 0.9950 | 0.9125 | 0.9920 | 0.8640 | 1.0000 | 0.9771 |

**The headline collapses.** The best accuracy falls from 0.9950 to 0.9500 and the best
sensitivity from 0.9960 to 0.9400 — from one missed patient in 250 to fifteen. The SVM,
top of the leaderboard with all features, drops to 0.9125 accuracy and 0.8640
sensitivity; Random Forest degrades most gracefully and is now clearly the best model
rather than nominally so.

Three things follow. First, roughly **nine-tenths of the residual error rate of this
benchmark is created by giving the model the diagnosis**. Second, 0.95 accuracy from
history and basic bloods alone is still a real signal — hypertension, diabetes and
glucose do carry information about kidney disease — but it is a different, and far more
honest, result than 0.995. Third, the ablated setting is the one where model choice
actually matters: here the McNemar tests do separate Random Forest from the Decision
Tree, because there are now enough errors to compare.

### 2.11 Experiment 4 — is the *pattern* of missing data predictive?

The original report speculated that missingness is informative because sicker patients
receive more tests. Two analyses test it, neither of which uses a single measured value
(`src/missingness_report.py`, `results/missingness/`).

**Missingness rate by outcome.** Twelve of the 24 features show a significant difference
(Fisher exact, Holm-corrected):

| Feature | % missing, CKD | % missing, not CKD | Gap (pp) |
|---|---|---|---|
| `rbc` | 57.2 | 6.0 | **+51.2** |
| `rc` | 49.6 | 4.7 | +44.9 |
| `wc` | 39.6 | 4.7 | +34.9 |
| `pot` | 33.2 | 3.3 | +29.9 |
| `sod` | 32.8 | 3.3 | +29.5 |
| `pcv` | 26.8 | 2.7 | +24.1 |
| `pc` | 22.4 | 6.0 | +16.4 |
| `hemo` | 18.4 | 4.0 | +14.4 |

The direction is the opposite of the naive guess: **the sicker group has *more* missing
data, not less.** Whatever produced this file, the CKD records are systematically less
complete.

**A model built only from missingness.** A logistic regression trained on nothing but the
24 binary was-this-measured flags — no measured value at all — evaluated with the same
leak-free 5-fold stratified CV:

| Metric | Value |
|---|---|
| Accuracy | 0.8075 |
| Sensitivity | 0.7840 |
| Specificity | 0.8467 |
| **AUC** | **0.8499** |

An AUC of 0.5 would mean missingness carries nothing. **0.85 means the record-keeping
alone identifies CKD patients most of the time.**

**Adding the flags as features helps, which is the problem.** Experiment 4 re-runs the
primary analysis with `SimpleImputer(add_indicator=True)`:

| Model | Accuracy | Sensitivity | NPV | FN | FP |
|---|---|---|---|---|---|
| Logistic Regression | **0.9975** | 0.9960 | 0.9934 | 1 | **0** |
| Random Forest | 0.9950 | **1.0000** | **1.0000** | **0** | 2 |
| SVM (RBF) | 0.9850 | **1.0000** | **1.0000** | **0** | 6 |
| Decision Tree | 0.9725 | 0.9800 | 0.9664 | 5 | 6 |

Random Forest reaches **sensitivity 1.000 with zero false negatives**, and
`missingindicator_rbc_True` ranks 8th by Gini importance — above diabetes. Five indicator
features appear in the top 20.

This is the most cautionary result in the report. The improvement is real *on this
benchmark* and entirely an artefact: it encodes which tests a clinician chose to order,
which is downstream of a suspicion that the patient had kidney disease. In a prospective
screening programme where every patient receives the same panel, these features would be
constant and the gain would vanish — or worse, a model that had learned to rely on them
would silently degrade. Experiment 4 is reported as a diagnosis of the dataset, **not as
a recommended configuration.**

---

## 3. Discussion

**No model is demonstrably better than another.** SVM (RBF) leads on accuracy (0.9950),
specificity (1.0000) and AUC (1.0000); Random Forest leads on sensitivity (0.9960) and
NPV (0.9932), missing 1 of 250 CKD patients versus 2 for the SVM. But six Holm-corrected
McNemar tests return no significant difference (§2.5), the bootstrap intervals overlap
(§2.1), and under 50 repeated folds the ranking of the top three reverses (§2.8). The
correct statement is that **four very different algorithms all saturate this benchmark**,
and 400 patients cannot separate them.

If a model must nonetheless be chosen for screening, **Random Forest** remains the
defensible pick — not because its point estimate is best, but because it has the best
*worst case* across 50 folds (minimum sensitivity 0.98, versus 0.94 for Logistic
Regression and 0.96 for the SVM) and it degrades most gracefully when the diagnostic
features are removed (§2.10). Robustness, not a two-patient margin, is the reason.

**The default threshold was the wrong place to stop.** The first version of this report
recommended Random Forest "for screening" while evaluating every model at a 0.5 cut-off,
which silently assumes a missed CKD case costs the same as a false alarm. Once the
threshold is selected inside the training folds under a stated 5:1 cost (§2.6), Logistic
Regression and the SVM both reach **sensitivity 1.000 — no missed patients — for two or
three false alarms**. That is the operating point a screening programme would actually
deploy, and it is available at no cost in methodology.

**Most of the accuracy is the diagnosis, and it is now measured rather than asserted.**
Removing the eight diagnostic-criterion features drops the best model from 0.9950 to
0.9500 accuracy and from 0.9960 to 0.9400 sensitivity (§2.10). Roughly nine-tenths of
this benchmark's headroom comes from handing the model the clinical definition of the
label. The remaining 0.95 from history, blood pressure, glucose and comorbidities is a
genuine and more interesting signal — and it is the number that should be compared
against any real screening tool.

**A second, subtler leak lives in the record-keeping.** Missingness is not random and not
in the direction one would guess: CKD patients have *more* missing values, `rbc` being
absent for 57.2% of them against 6.0% of the healthy group. A model built from nothing
but the was-this-measured flags reaches AUC 0.850 (§2.11). Adding those flags as features
pushes Random Forest to sensitivity 1.000, which looks like an improvement and is in fact
the model learning which tests a clinician decided to order. This is the classic shape of
a dataset artefact: it improves every offline metric and would disappear, or invert, in
prospective use.

**Interpreting the null leakage result.** The measured optimistic bias from preprocessing
before cross-validation was exactly zero (§2.4). This is a property of *this* dataset,
not a general finding: with near-perfect separability, performance is insensitive to
small shifts in imputation values, and with n = 400 a fold median is almost identical to
the global median. Leakage causes real damage when the dataset is small, the signal is
weak, or preprocessing is aggressive. The pipeline discipline cost nothing here and
protects against those cases, so it was retained — and `tests/test_pipeline.py` now
asserts it mechanically rather than by inspection.

**Tuning was not the missing ingredient.** Nested cross-validation, the standard remedy
for the "defaults were never tuned" objection, improved nothing and degraded two models
(§2.9). The per-fold hyperparameter choices are unstable, which is what a search looks
like when there is nothing left to find.

**Clinical plausibility.** The top features align with established CKD pathophysiology:
haemoglobin and packed cell volume capture the anaemia of chronic kidney disease
(reduced erythropoietin production), while serum creatinine and urine specific gravity
capture impaired glomerular filtration and urine-concentrating ability. Hypertension and
diabetes, the two leading causes of CKD, also appear in the top eight — and remain
predictive in the ablated setting, where they are among the few clinical signals left.
The model is learning recognised medicine as well as label leakage.

**Limitations.** Only 400 patients from a single source and no external validation
cohort. The bootstrap intervals in §2.1 quantify evaluation uncertainty on a fixed set of
predictions, not the extra variability from refitting on a different sample, so they are
a lower bound. The 5:1 screening cost in §1.11 is a declared assumption, not an estimated
quantity; a different weighting moves the recommended thresholds. The Decision Tree
cannot be threshold-tuned at all because its probabilities are near-binary (§2.6). The
reported values are cross-validated estimates, not prospective clinical performance. And
the near-separability documented in §2.10, together with the collection artefact in
§2.11, means these scores describe a benchmark, not a screening problem.

---

## 4. Conclusion

A leak-free machine-learning pipeline for chronic kidney disease classification was built
and evaluated on 400 patient records using 5-fold stratified cross-validation. Four
data-quality defects in the raw file were identified and corrected, and 10.54% missing
values were handled by median/mode imputation performed strictly within training folds.

At the default threshold, all four models exceed 0.97 accuracy: SVM (RBF) leads on
accuracy (0.9950) and Random Forest on sensitivity (0.9960), correctly identifying 249 of
250 CKD patients. **No pairwise difference between the four models is statistically
significant** (six Holm-corrected exact McNemar tests, smallest *p* = 0.66), and the
ranking of the top three reverses across 50 repeated folds — so the appropriate
conclusion is that all four saturate this benchmark rather than that any one wins.
Selecting the decision threshold inside the training folds under an explicit 5:1
screening cost lifts Logistic Regression and the SVM to **sensitivity 1.000 — no missed
CKD patients — at two or three false alarms**. Fold-safe IQR winsorisation produced no
improvement, and unbiased nested cross-validation improved nothing and degraded two
models, so the untuned no-outlier-removal configuration is retained as primary.

Two experiments explain the scores rather than celebrate them. Removing the eight
features that constitute the clinical definition of CKD drops the best model to
**0.9500 accuracy and 0.9400 sensitivity**, showing that most of the benchmark's
headroom is the diagnosis handed to the model. And the *pattern* of missing data is
itself predictive: a model built from nothing but the was-this-measured flags reaches
**AUC 0.850**, because CKD records in this file are systematically less complete. Both
findings say the same thing in different ways — the near-perfect scores reflect a clean,
near-separable benchmark and its collection process, not the difficulty of real-world
screening. **These models are not clinically validated and must not be used to inform
patient care.**

Future work: external validation on an independent cohort; a prospective dataset in which
every patient receives the same panel, which would neutralise the missingness artefact;
and calibrated risk estimates evaluated against clinician judgement rather than against
a label derived from the same measurements.

---

## Appendix A — A note on the three AUC values

Three AUC numbers appear in this project and they are *not* expected to be identical:

1. **Per-fold AUC** — computed within one held-out fold (Table 3 reports mean ± SD).
2. **Pooled out-of-fold AUC** — one AUC over all 400 out-of-fold scores (Table 1).
3. **The AUC printed on the ROC figures** — labelled as the mean ± SD of the *fold*
   AUCs, following the scikit-learn convention.

The area under the *interpolated mean* ROC curve is a fourth quantity and is slightly
lower than the mean of the fold areas (interpolating five near-square curves onto a
common FPR grid rounds off their corners). The figures therefore quote the mean of the
fold AUCs rather than the area of the drawn mean curve, so that the number on the plot
matches Table 3. If your teacher asks for "the AUC", quote the **pooled out-of-fold
AUC** from Table 1 and state that it is out-of-fold.

## Appendix B — Reproducibility

| Item | Value |
|---|---|
| `random_state` | 42 (splits, all models, bootstrap and repeated CV) |
| CV | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Fold composition | every test fold: 50 CKD + 30 notckd |
| Stability run | `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` |
| Bootstrap | 2,000 resamples, percentile method, `random_state=42` |
| Verified with | Python 3.13.7, scikit-learn 1.7.1, pandas 2.2.3, numpy 2.1.3, scipy 1.16.1 |
| Notebook | 53 cells, executed end-to-end with 0 errors |

Exact versions are pinned in `requirements.txt`, and each experiment's
`summary.json` records the interpreter and library versions it was produced with, so a
rebuilt result can always be traced to the environment that produced it.

Regenerate everything with `make all` (or `./run_all.sh`); check the invariants with
`make test`. The CI workflow in `.github/workflows/ci.yml` re-runs the primary experiment
on a clean machine and asserts that the rebuilt Accuracy, Sensitivity, Specificity and
AUC match the committed `results/no_outlier/summary.json` to four decimal places, so the
reproducibility claim on this page is checked rather than asserted.

The cleaning step casts categorical columns to `object`/`np.nan` rather than leaving
them as pandas nullable `string`, because scikit-learn's `SimpleImputer` cannot consume
the `pd.NA` sentinel that pandas ≥ 3.0 produces by default. The same code therefore runs
unchanged on pandas 2.x (Colab's current version) and 3.x.
