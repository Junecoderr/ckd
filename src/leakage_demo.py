"""Empirical check: does preprocessing BEFORE cross-validation inflate scores?

WRONG  : impute + scale on all 400 rows, then cross-validate the model only.
CORRECT: impute + scale inside each fold (fit on 4 folds, apply to the 5th).
"""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ckd_pipeline import (CATEGORICAL, NUMERIC, RANDOM_STATE, TARGET,
                          build_preprocessor, load_and_clean)

df = load_and_clean("data/kidney_disease.csv")
X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
clf = lambda: LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)

# ---- WRONG: fit the preprocessor once, on 100% of the data ----------------
leaky_prep = build_preprocessor(False).fit(X)      # <-- sees the test folds
X_leaky = leaky_prep.transform(X)
wrong = cross_val_score(clf(), X_leaky, y, cv=cv, scoring="roc_auc")
wrong_acc = cross_val_score(clf(), X_leaky, y, cv=cv, scoring="accuracy")

# ---- CORRECT: preprocessing is a Pipeline step, refit every fold ----------
pipe = Pipeline([("prep", build_preprocessor(False)), ("clf", clf())])
right = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
right_acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

print("                        AUC (mean +/- SD)      Accuracy (mean +/- SD)")
print(f"WRONG (preprocess first) {wrong.mean():.4f} +/- {wrong.std():.4f}"
      f"        {wrong_acc.mean():.4f} +/- {wrong_acc.std():.4f}")
print(f"CORRECT (in pipeline)    {right.mean():.4f} +/- {right.std():.4f}"
      f"        {right_acc.mean():.4f} +/- {right_acc.std():.4f}")
print(f"\nOptimistic bias: AUC {wrong.mean()-right.mean():+.4f}   "
      f"Accuracy {wrong_acc.mean()-right_acc.mean():+.4f}")
