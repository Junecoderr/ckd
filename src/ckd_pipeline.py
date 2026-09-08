"""
Chronic Kidney Disease (CKD) Classification
===========================================
Leak-free 5-fold Stratified Cross-Validation evaluation.

Experiment 1 : NO outlier removal  (--experiment no_outlier)
Experiment 2 : IQR outlier treatment via winsorisation, fold-safe
               (--experiment outlier_capped)

Usage:
    python src/ckd_pipeline.py --experiment no_outlier
    python src/ckd_pipeline.py --experiment outlier_capped
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5

# ----------------------------------------------------------------------------
# Column groups (fixed by domain knowledge of the UCI CKD dataset)
# ----------------------------------------------------------------------------
NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
           "sod", "pot", "hemo", "pcv", "wc", "rc"]
CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad",
               "appet", "pe", "ane"]
TARGET = "classification"

# Columns that are ordinal-but-clinical (sg, al, su are graded lab readings).
# They are treated as numeric because their order carries meaning.


# ----------------------------------------------------------------------------
# STEP 1 -- Load and clean (structural cleaning only; no statistics learned)
# ----------------------------------------------------------------------------
def load_and_clean(csv_path):
    """Read the raw CSV and repair *structural* defects only.

    Structural = typos, stray tab/space characters, '?' placeholders,
    wrong dtypes. These fixes use NO information from the data
    distribution, so doing them before cross-validation cannot leak.
    """
    df = pd.read_csv(csv_path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])          # row identifier, not a feature

    # 1a. strip stray whitespace / tabs from every text cell
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "str":
            df[col] = df[col].astype("string").str.strip()

    # 1b. '?' and '' are missing-value placeholders, not categories
    df = df.replace({"?": pd.NA, "": pd.NA})

    # 1c. pcv / wc / rc arrive as text because of the '\t?' cells -> force numeric
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1d. collapse duplicated category spellings (e.g. '\tno' -> 'no').
    #     Cast back to plain object/np.nan: scikit-learn's imputers do not
    #     accept pandas' nullable <NA> sentinel (pandas >= 3.0 default).
    for col in CATEGORICAL:
        df[col] = (df[col].astype("string").str.lower()
                   .astype(object).where(lambda s: s.notna(), np.nan))

    # 1e. target: 'ckd\t' -> 'ckd'; encode CKD as the POSITIVE class (1)
    df[TARGET] = df[TARGET].astype("string").str.strip().str.lower()
    df[TARGET] = df[TARGET].map({"ckd": 1, "notckd": 0})

    assert df[TARGET].notna().all(), "Unmapped label found in target column"
    df[TARGET] = df[TARGET].astype(int)

    return df


# ----------------------------------------------------------------------------
# STEP 2 -- Optional outlier treatment, implemented as a fold-safe transformer
# ----------------------------------------------------------------------------
class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """Cap numeric values at the IQR fences LEARNED ON TRAINING DATA ONLY.

    Because it is a transformer inside the Pipeline, `fit` only ever sees
    the 4 training folds; the held-out fold is merely `transform`-ed with
    those fences. This is why outlier treatment does not leak here.

    Winsorising (capping) is used instead of deleting rows: a transformer
    is not allowed to change the number of samples, and more importantly
    deleting "outlier" patients would discard exactly the severe cases a
    CKD screening model must detect.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        # capping does not change the columns, so names pass straight through
        return np.asarray(input_features, dtype=object)


# ----------------------------------------------------------------------------
# STEP 3 -- Preprocessing + model, all inside ONE Pipeline (no leakage)
# ----------------------------------------------------------------------------
def build_preprocessor(treat_outliers):
    """ColumnTransformer: median-impute + (optionally cap) + scale numerics;
    mode-impute + one-hot encode categoricals."""

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if treat_outliers:
        num_steps.append(("winsorize", IQRWinsorizer(factor=1.5)))
    num_steps.append(("scale", StandardScaler()))

    numeric_pipe = Pipeline(num_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000,
                                                  random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300,
                                                random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True,
                         random_state=RANDOM_STATE),
    }


# ----------------------------------------------------------------------------
# STEP 4 -- Metrics from the confusion matrix
# ----------------------------------------------------------------------------
def metrics_from_cm(y_true, y_pred, y_score):
    """All requested metrics, derived explicitly from TP/TN/FP/FN.

    labels=[0, 1] fixes the orientation so the matrix is always
        [[TN, FP],
         [FN, TP]]
    with CKD = 1 as the positive class.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan   # = recall = TPR
    specificity = tn / (tn + fp) if (tn + fp) else np.nan   # = TNR
    precision   = tp / (tp + fp) if (tp + fp) else np.nan   # = PPV
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) else np.nan)
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    auc_val = roc_auc_score(y_true, y_score)

    return dict(TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
                Accuracy=accuracy, Sensitivity=sensitivity,
                Specificity=specificity, Precision=precision,
                F1=f1, NPV=npv, AUC=auc_val)


# ----------------------------------------------------------------------------
# STEP 5 -- 5-fold stratified CV evaluation
# ----------------------------------------------------------------------------
def evaluate(name, model, X, y, preproc_factory, outdir):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)

    per_fold = []
    roc_curves = []                       # (fpr, tpr, auc) for each fold
    oof_pred = np.zeros(len(y), dtype=int)
    oof_score = np.zeros(len(y), dtype=float)

    for k, (tr, te) in enumerate(cv.split(X, y), start=1):
        pipe = Pipeline([("prep", preproc_factory()), ("clf", model)])
        # .fit sees ONLY the training folds -> imputation medians, scaler
        # means/SDs, one-hot categories and IQR fences are all learned there.
        pipe.fit(X.iloc[tr], y.iloc[tr])

        pred  = pipe.predict(X.iloc[te])
        score = pipe.predict_proba(X.iloc[te])[:, 1]

        oof_pred[te]  = pred
        oof_score[te] = score

        m = metrics_from_cm(y.iloc[te], pred, score)
        m["Fold"] = k
        per_fold.append(m)

        fpr, tpr, _ = roc_curve(y.iloc[te], score)
        roc_curves.append((fpr, tpr, auc(fpr, tpr)))

    fold_df = pd.DataFrame(per_fold).set_index("Fold")
    pooled  = metrics_from_cm(y, oof_pred, oof_score)   # out-of-fold pooled

    return dict(name=name, fold_df=fold_df, pooled=pooled,
                roc_curves=roc_curves, oof_pred=oof_pred, oof_score=oof_score)


# ----------------------------------------------------------------------------
# STEP 6 -- Plots
# ----------------------------------------------------------------------------
def plot_confusion(pooled, name, tag, outdir):
    cm = np.array([[pooled["TN"], pooled["FP"]],
                   [pooled["FN"], pooled["TP"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center",
                    va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred: notckd", "Pred: ckd"])
    ax.set_yticks([0, 1], ["True: notckd", "True: ckd"])
    ax.set_title(f"Confusion Matrix (pooled out-of-fold)\n{name}")
    fig.tight_layout()
    fig.savefig(outdir / f"confusion_matrix_{tag}_{slug(name)}.png", dpi=160)
    plt.close(fig)


def plot_roc(res, tag, outdir):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    for k, (fpr, tpr, a) in enumerate(res["roc_curves"], start=1):
        ax.plot(fpr, tpr, lw=1, alpha=0.45,
                label=f"Fold {k} (AUC = {a:.4f})")
        t = np.interp(mean_fpr, fpr, tpr); t[0] = 0.0
        tprs.append(t)
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    fold_aucs = [a for _, _, a in res["roc_curves"]]
    # Label with mean +/- SD of the FOLD AUCs (scikit-learn convention). The
    # area under the interpolated mean curve is not identical to the mean of
    # the fold areas, so quoting the fold statistics avoids a confusing third
    # number alongside the per-fold and pooled AUCs in the results tables.
    ax.plot(mean_fpr, mean_tpr, "b-", lw=2.4,
            label=f"Mean ROC (AUC = {np.mean(fold_aucs):.4f} "
                  f"$\\pm$ {np.std(fold_aucs):.4f})")
    sd = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - sd, 0),
                    np.minimum(mean_tpr + sd, 1), color="b", alpha=0.12,
                    label="± 1 SD")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.5)")
    ax.set(xlabel="1 - Specificity (False Positive Rate)",
           ylabel="Sensitivity (True Positive Rate)",
           title=f"ROC Curve - {res['name']}\n5-fold Stratified CV")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"roc_curve_{tag}_{slug(res['name'])}.png", dpi=160)
    plt.close(fig)


def plot_model_comparison(summary, tag, outdir):
    metrics = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "AUC"]
    x = np.arange(len(summary)); w = 0.13
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, m in enumerate(metrics):
        ax.bar(x + i * w - 2.5 * w, summary[m], w, label=m)
    ax.set_xticks(x, summary.index, fontsize=9)
    ax.set_ylim(0.90, 1.005)
    ax.set_ylabel("Score (pooled out-of-fold)")
    ax.set_title("Model comparison - 5-fold Stratified CV")
    ax.legend(ncol=6, fontsize=8, loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"model_comparison_{tag}.png", dpi=160)
    plt.close(fig)


def plot_missingness(df, outdir):
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(miss.index[::-1], miss.values[::-1], color="#c0392b")
    for i, v in enumerate(miss.values[::-1]):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% missing"); ax.set_title("Missing values per column")
    fig.tight_layout()
    fig.savefig(outdir / "missing_values.png", dpi=160)
    plt.close(fig)


def plot_feature_importance(X, y, preproc_factory, outdir, tag):
    pipe = Pipeline([("prep", preproc_factory()),
                     ("clf", RandomForestClassifier(n_estimators=300,
                                                    random_state=RANDOM_STATE))])
    pipe.fit(X, y)
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_,
                    index=[n.split("__", 1)[1] for n in names])
    top = imp.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(top.index[::-1], top.values[::-1], color="#2e7d32")
    ax.set_xlabel("Gini importance")
    ax.set_title("Top 15 features (Random Forest, full data - descriptive only)")
    fig.tight_layout()
    fig.savefig(outdir / f"feature_importance_{tag}.png", dpi=160)
    plt.close(fig)
    return imp.sort_values(ascending=False)


def slug(s):
    return s.lower().replace(" ", "_").replace("(", "").replace(")", "")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/kidney_disease.csv")
    ap.add_argument("--experiment", choices=["no_outlier", "outlier_capped"],
                    default="no_outlier")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    treat_outliers = args.experiment == "outlier_capped"
    tag = args.experiment
    outdir = Path(args.outdir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"CKD CLASSIFICATION  |  experiment = {tag}")
    print("=" * 78)

    # ---- STEP 1
    df = load_and_clean(args.csv)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]

    print(f"\n[1] Cleaned data: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"    Class balance -> ckd(1) = {(y == 1).sum()}  "
          f"notckd(0) = {(y == 0).sum()}  "
          f"({(y == 1).mean() * 100:.1f}% positive)")
    print(f"    Rows with >=1 missing value: {X.isna().any(axis=1).sum()} "
          f"({X.isna().any(axis=1).mean() * 100:.1f}%)")
    print(f"    Complete cases: {(~X.isna().any(axis=1)).sum()}")

    miss_tbl = pd.DataFrame({
        "missing_count": X.isna().sum(),
        "missing_pct": (X.isna().mean() * 100).round(2),
    }).sort_values("missing_count", ascending=False)
    miss_tbl.to_csv(outdir / "table_missing_values.csv")
    print("\n[2] Missing values (top 10):")
    print(miss_tbl.head(10).to_string())

    if tag == "no_outlier":
        plot_missingness(X, outdir)

    # descriptive stats + outlier counts (reported, NOT acted on in exp.1)
    desc = X[NUMERIC].describe().T
    q1, q3 = X[NUMERIC].quantile(.25), X[NUMERIC].quantile(.75)
    iqr = q3 - q1
    desc["outliers_IQR_1.5"] = (((X[NUMERIC] < q1 - 1.5 * iqr) |
                                 (X[NUMERIC] > q3 + 1.5 * iqr)).sum())
    desc.round(3).to_csv(outdir / "table_descriptive_stats.csv")
    print("\n[3] Numeric summary + IQR outlier counts:")
    print(desc[["mean", "std", "min", "max", "outliers_IQR_1.5"]]
          .round(2).to_string())

    # ---- STEPS 3-5
    factory = lambda: build_preprocessor(treat_outliers)
    print(f"\n[4] Running {N_SPLITS}-fold Stratified CV "
          f"(outlier capping = {treat_outliers}) ...")

    all_res, rows = {}, []
    for name, model in build_models().items():
        res = evaluate(name, model, X, y, factory, outdir)
        all_res[name] = res
        res["fold_df"].round(4).to_csv(outdir / f"table_folds_{slug(name)}.csv")

        p = res["pooled"]
        rows.append({"Model": name, **{k: p[k] for k in
                     ["Accuracy", "Sensitivity", "Specificity", "Precision",
                      "F1", "AUC", "TP", "TN", "FP", "FN"]},
                     "AUC_mean_folds": res["fold_df"]["AUC"].mean(),
                     "AUC_sd_folds": res["fold_df"]["AUC"].std(),
                     "Acc_mean_folds": res["fold_df"]["Accuracy"].mean(),
                     "Acc_sd_folds": res["fold_df"]["Accuracy"].std()})

        print(f"\n  --- {name} ---")
        print(res["fold_df"][["TP", "TN", "FP", "FN", "Accuracy",
                              "Sensitivity", "Specificity", "Precision",
                              "F1", "AUC"]].round(4).to_string())
        print(f"  Pooled OOF: Acc={p['Accuracy']:.4f}  Sens={p['Sensitivity']:.4f}  "
              f"Spec={p['Specificity']:.4f}  Prec={p['Precision']:.4f}  "
              f"F1={p['F1']:.4f}  AUC={p['AUC']:.4f}")

        plot_confusion(p, name, tag, outdir)
        plot_roc(res, tag, outdir)

    summary = pd.DataFrame(rows).set_index("Model")
    summary.round(4).to_csv(outdir / "table_model_summary.csv")
    plot_model_comparison(summary, tag, outdir)

    print("\n[5] SUMMARY (pooled out-of-fold, CKD = positive class)")
    print(summary[["Accuracy", "Sensitivity", "Specificity", "Precision",
                   "F1", "AUC", "TP", "TN", "FP", "FN"]].round(4).to_string())

    imp = plot_feature_importance(X, y, factory, outdir, tag)
    imp.round(4).to_csv(outdir / "table_feature_importance.csv",
                        header=["importance"])
    print("\n[6] Top 10 features (Random Forest):")
    print(imp.head(10).round(4).to_string())

    best = summary["AUC"].idxmax()
    print(f"\n[7] Best model by pooled AUC: {best} "
          f"(AUC = {summary.loc[best, 'AUC']:.4f})")

    with open(outdir / "summary.json", "w") as f:
        json.dump({"experiment": tag,
                   "n_rows": int(len(df)),
                   "n_ckd": int((y == 1).sum()),
                   "n_notckd": int((y == 0).sum()),
                   "best_model": best,
                   "results": summary.round(6).to_dict(orient="index")},
                  f, indent=2)
    print(f"\nAll tables + figures written to: {outdir}/")


if __name__ == "__main__":
    main()
