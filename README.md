# CKD Classification — Minor Project

Machine-learning classification of Chronic Kidney Disease on the UCI CKD dataset
(400 patients), evaluated with leak-free 5-fold stratified cross-validation.

## Quick start (Google Colab — recommended for beginners)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. `File ▸ Upload notebook` → upload **`notebooks/CKD_Classification_Colab.ipynb`**
3. Run the first cell and upload `data/kidney_disease.csv` when prompted
4. `Runtime ▸ Run all`

Colab already has pandas, numpy, scikit-learn, matplotlib and seaborn — nothing to install.

## Quick start (local)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python src/ckd_pipeline.py --experiment no_outlier       # Experiment 1 (primary)
python src/ckd_pipeline.py --experiment outlier_capped   # Experiment 2 (comparison)
PYTHONPATH=src python src/leakage_demo.py                # leakage check
```

## Layout

```
data/kidney_disease.csv                   raw dataset (400 rows)
notebooks/CKD_Classification_Colab.ipynb  step-by-step teaching notebook (53 cells)
src/ckd_pipeline.py                       reproducible analysis script
src/leakage_demo.py                       measures leaky vs correct preprocessing
results/no_outlier/                       Experiment 1 tables (CSV) + figures (PNG)
results/outlier_capped/                   Experiment 2 tables + figures
results/table_experiment_comparison.csv   side-by-side comparison
REPORT.md                                 methodology, results, discussion, conclusion
```

## Headline result (Experiment 1, pooled out-of-fold, n = 400)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 | AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9925 | 0.9920 | 0.9933 | 0.9960 | 0.9940 | 0.9998 |
| Decision Tree | 0.9800 | 0.9880 | 0.9667 | 0.9802 | 0.9841 | 0.9773 |
| Random Forest | 0.9900 | **0.9960** | 0.9800 | 0.9881 | 0.9920 | 0.9997 |
| SVM (RBF) | **0.9950** | 0.9920 | **1.0000** | **1.0000** | **0.9960** | **1.0000** |

Positive class = CKD. Random Forest missed only 1 of 250 CKD patients (FN = 1) and is
recommended for screening; SVM leads on accuracy and specificity.

**Caveat:** the top predictors are the defining diagnostic criteria of CKD, so this
benchmark is near-separable. These scores would not transfer to raw hospital data.
See `REPORT.md` for the full discussion.

## How data leakage is prevented

Imputation, scaling, one-hot encoding and (in Experiment 2) IQR fences all live inside a
`sklearn.pipeline.Pipeline`, which is refitted from scratch inside every training fold.
Held-out folds are only `.transform()`-ed, never `.fit()`-ted on.
