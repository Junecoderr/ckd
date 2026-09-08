# Chronic Kidney Disease Classification using Machine Learning
### Minor Project — Results Report

All numbers in this report were produced by `src/ckd_pipeline.py` on the supplied
`kidney_disease.csv` (400 patients). Nothing is estimated or illustrative.
Reproduce with:

```bash
python src/ckd_pipeline.py --experiment no_outlier       # Experiment 1
python src/ckd_pipeline.py --experiment outlier_capped   # Experiment 2
```

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

**Every preprocessing step that learns from data** — imputation medians and modes,
scaler means and standard deviations, one-hot category lists, and (in Experiment 2) the
IQR fences — **was placed inside a `sklearn.pipeline.Pipeline` and refitted from scratch
within every training fold.** For each of the 5 folds, `.fit()` sees only the 4 training
folds; the held-out fold is merely `.transform()`-ed using those training-derived
statistics. No information from a held-out fold ever influences preprocessing.

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

### 1.8 Algorithms

Logistic Regression (`max_iter=5000`), Decision Tree, Random Forest (300 trees),
and SVM with RBF kernel (`probability=True`). All at scikit-learn defaults otherwise,
with `random_state=42`.

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
F_1=2\cdot\frac{\text{Precision}\times\text{Sensitivity}}{\text{Precision}+\text{Sensitivity}}$$

**AUC** is the area under the ROC curve (Sensitivity plotted against 1 − Specificity
across all decision thresholds). It equals the probability that the model assigns a
randomly chosen CKD patient a higher risk score than a randomly chosen healthy person,
and is threshold-independent.

---

## 2. Results

### 2.1 Experiment 1 — without outlier removal (primary experiment)

**Table 1. Pooled out-of-fold performance, 5-fold Stratified CV (n = 400).**

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9925 | 0.9920 | 0.9933 | 0.9960 | 0.9940 | 0.9998 |
| Decision Tree | 0.9800 | 0.9880 | 0.9667 | 0.9802 | 0.9841 | 0.9773 |
| Random Forest | 0.9900 | **0.9960** | 0.9800 | 0.9881 | 0.9920 | 0.9997 |
| **SVM (RBF)** | **0.9950** | 0.9920 | **1.0000** | **1.0000** | **0.9960** | **1.0000** |

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

**Worked example — reading the Random Forest confusion matrix:**
TP = 249, TN = 147, FP = 3, FN = 1.
Sensitivity = 249/(249+1) = **0.9960**; Specificity = 147/(147+3) = **0.9800**;
Precision = 249/(249+3) = **0.9881**; F1 = 2(0.9881×0.9960)/(0.9881+0.9960) = **0.9920**.

**Figures** (in `results/no_outlier/`): `missing_values.png`,
`confusion_matrix_no_outlier_*.png`, `roc_curve_no_outlier_*.png` (5 fold curves plus
mean ROC with ±1 SD band), `model_comparison_no_outlier.png`,
`feature_importance_no_outlier.png`.

### 2.2 Feature importance (Random Forest)

| Rank | Feature | Importance | Rank | Feature | Importance |
|---|---|---|---|---|---|
| 1 | `hemo` (haemoglobin) | 0.2298 | 6 | `htn_yes` (hypertension) | 0.0525 |
| 2 | `pcv` (packed cell volume) | 0.1571 | 7 | `al` (albumin) | 0.0513 |
| 3 | `sc` (serum creatinine) | 0.1308 | 8 | `dm_yes` (diabetes) | 0.0373 |
| 4 | `sg` (specific gravity) | 0.1176 | 9 | `sod` (sodium) | 0.0244 |
| 5 | `rc` (red cell count) | 0.0845 | 10 | `bgr` (blood glucose) | 0.0244 |

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

---

## 3. Discussion

**Model choice depends on which error matters.** SVM (RBF) achieved the best accuracy
(0.9950), specificity (1.0000), precision (1.0000) and AUC (1.0000), producing zero false
alarms. Random Forest achieved the best **sensitivity** (0.9960), missing only **1** of
250 CKD patients versus 2 for the SVM. In a screening context a false negative — an
untreated CKD patient sent home — is clinically far costlier than a false positive, which
merely triggers a confirmatory test. **Random Forest is therefore recommended as the
primary model**, with the SVM noted as marginally stronger on overall accuracy. The
differences involve one or two patients out of 400 and are well within fold-to-fold
variation (SD ≈ 0.01), so no model is meaningfully superior on this evidence.

**Why the scores are so high — an honest caveat.** These results should not be presented
as evidence of a deployment-ready screening tool. The top predictors — haemoglobin,
packed cell volume, serum creatinine, specific gravity, red cell count — are the
*defining diagnostic criteria* of chronic kidney disease, so the two classes are close to
linearly separable. The dataset is a clean teaching benchmark; performance near 99% here
would not transfer to raw hospital data with noisier labels and less complete testing.

**Interpreting the null leakage result.** The measured optimistic bias was exactly zero.
This is a property of *this* dataset, not a general finding: with near-perfect
separability, model performance is insensitive to small shifts in imputation values, and
with n = 400 a fold median is almost identical to the global median. Leakage causes real
damage when the dataset is small, the signal is weak, or preprocessing is aggressive
(feature selection on all data, resampling before splitting, target encoding). Correct
pipeline construction cost nothing here and protects against those cases, so it was
retained.

**Missingness is probably informative.** `rbc` is 38% missing and `rc` 32.8%. Missingness
is unlikely to be random — sicker patients receive more tests — so mode-imputing `rbc`
may destroy genuine signal. Adding explicit missing-indicator features would let a model
exploit that pattern and is worth testing.

**Clinical plausibility.** The top features align with established CKD pathophysiology:
haemoglobin and packed cell volume capture the anaemia of chronic kidney disease
(reduced erythropoietin production), while serum creatinine and urine specific gravity
capture impaired glomerular filtration and urine-concentrating ability. Hypertension and
diabetes, the two leading causes of CKD, also appear in the top eight. The model is
learning recognised medicine, not an artefact.

**Limitations.** Only 400 patients from a single source; no external validation cohort;
hyperparameters left at library defaults (tuning them without bias would require nested
cross-validation); the reported values are cross-validated estimates, not prospective
clinical performance; and the near-perfect separability limits how much can be inferred
about method quality from these scores.

---

## 4. Conclusion

A leak-free machine-learning pipeline for chronic kidney disease classification was built
and evaluated on 400 patient records using 5-fold stratified cross-validation. Four
data-quality defects in the raw file were identified and corrected, and 10.54% missing
values were handled by median/mode imputation performed strictly within training folds.

Without outlier removal, **Random Forest** correctly identified **249 of 250 CKD patients**
— sensitivity **0.9960**, specificity **0.9800**, precision **0.9881**, F1 **0.9920**,
AUC **0.9997**, accuracy **0.9900** — making it the preferred model for a screening
application, where missed cases are the costly error. SVM (RBF) attained marginally higher
accuracy (0.9950) and AUC (1.0000) with perfect specificity. A second experiment applying
1.5×IQR winsorisation produced no improvement and slightly degraded the linear and
tree models, so the no-outlier-removal configuration was retained.

The high scores reflect a clean, near-separable benchmark dataset rather than the
difficulty of real-world screening, and the models should not be treated as clinically
validated. Future work: external validation on an independent cohort, decision-threshold
tuning to trade specificity for sensitivity, missing-indicator features to exploit
informative missingness, and nested cross-validation for unbiased hyperparameter tuning.

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
| `random_state` | 42 (splits and all models) |
| CV | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Fold composition | every test fold: 50 CKD + 30 notckd |
| Verified with | pandas 3.0.5, scikit-learn 1.9.0, numpy 2.4.6 |
| Notebook | 53 cells, executed end-to-end with 0 errors |

The cleaning step casts categorical columns to `object`/`np.nan` rather than leaving
them as pandas nullable `string`, because scikit-learn's `SimpleImputer` cannot consume
the `pd.NA` sentinel that pandas ≥ 3.0 produces by default. The same code therefore runs
unchanged on pandas 2.x (Colab's current version) and 3.x.
