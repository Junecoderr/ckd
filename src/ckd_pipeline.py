"""
Chronic Kidney Disease (CKD) Classification
===========================================
Leak-free Stratified Cross-Validation evaluation, with uncertainty
quantification, paired significance testing, decision-threshold analysis,
probability calibration, fold stability and nested hyperparameter tuning.

Experiments (`--experiment`)
    no_outlier        NO outlier removal                       (primary)
    outlier_capped    IQR winsorisation, fold-safe             (comparison)
    ablation          diagnostic-criterion features removed    (leakage probe)
    missing_indicator missingness added as explicit features   (MNAR probe)

Orthogonal flags
    --tune                  nested CV: inner GridSearchCV per outer fold
    --stability-repeats N   RepeatedStratifiedKFold seed-stability run
    --n-boot N              bootstrap resamples for confidence intervals

Usage:
    python src/ckd_pipeline.py --experiment no_outlier
    python src/ckd_pipeline.py --experiment outlier_capped
    python src/ckd_pipeline.py --experiment no_outlier --tune
    python src/ckd_pipeline.py --experiment ablation
    python src/ckd_pipeline.py --experiment missing_indicator
"""

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_tests import bootstrap_ci, pairwise_mcnemar  # noqa: E402

# Silence only the two warnings this analysis legitimately triggers, instead of
# blanket-ignoring everything (which would also hide the API deprecations that
# break reproducibility on a future scikit-learn).
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

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
# Features that are the diagnosis, or a direct read-out of it
# ----------------------------------------------------------------------------
# CKD is *defined* clinically (KDIGO criteria) by reduced glomerular filtration
# rate -- computed from serum creatinine (`sc`) -- or by markers of kidney
# damage, chiefly albuminuria (`al`), persisting three months or more. Blood
# urea (`bu`) is the other standard renal-function read-out; urine specific
# gravity (`sg`) measures concentrating ability; and haemoglobin (`hemo`),
# packed cell volume (`pcv`), red-cell count (`rc`) and red blood cells in
# urine (`rbc`) track the anaemia and haematuria that accompany established
# disease.
#
# A model given these is largely re-deriving the label rather than predicting
# it. The `ablation` experiment removes them all and reports what is left.
DIAGNOSTIC_CRITERIA = ["sc", "bu", "al", "sg", "hemo", "pcv", "rc", "rbc"]

# ----------------------------------------------------------------------------
# Screening cost model for decision-threshold selection
# ----------------------------------------------------------------------------
# An explicit, stated assumption -- not a value read from the data. In CKD
# screening a missed case (FN) delays treatment of a progressive disease, while
# a false alarm (FP) costs one confirmatory test. We weight FN five times FP.
# Change these two numbers to change the recommended operating point.
FN_COST = 5.0
FP_COST = 1.0

EXPERIMENTS = {
    "no_outlier":        dict(outliers=False, features="all",
                              add_indicator=False),
    "outlier_capped":    dict(outliers=True,  features="all",
                              add_indicator=False),
    "ablation":          dict(outliers=False, features="non_diagnostic",
                              add_indicator=False),
    "missing_indicator": dict(outliers=False, features="all",
                              add_indicator=True),
}


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
def feature_sets(spec):
    """Return (numeric, categorical) column lists for a feature specification."""
    if spec == "all":
        return list(NUMERIC), list(CATEGORICAL)
    if spec == "non_diagnostic":
        return ([c for c in NUMERIC if c not in DIAGNOSTIC_CRITERIA],
                [c for c in CATEGORICAL if c not in DIAGNOSTIC_CRITERIA])
    raise ValueError(f"unknown feature spec: {spec}")


def build_preprocessor(treat_outliers, numeric=None, categorical=None,
                       add_indicator=False):
    """ColumnTransformer: median-impute + (optionally cap) + scale numerics;
    mode-impute + one-hot encode categoricals.

    `add_indicator=True` additionally emits a binary was-missing flag for every
    column that had a missing value in the training fold. The flags are learned
    inside the fold like everything else, so they do not leak.
    """
    numeric = list(NUMERIC) if numeric is None else list(numeric)
    categorical = list(CATEGORICAL) if categorical is None else list(categorical)

    num_steps = [("impute", SimpleImputer(strategy="median",
                                          add_indicator=add_indicator))]
    if treat_outliers:
        # Capping is applied after imputation so that the fences are computed on
        # a complete matrix; imputed values sit at the training median and are
        # therefore never themselves clipped.
        num_steps.append(("winsorize", IQRWinsorizer(factor=1.5)))
    num_steps.append(("scale", StandardScaler()))

    numeric_pipe = Pipeline(num_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent",
                                 add_indicator=add_indicator)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000,
                                                  random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True,
                         random_state=RANDOM_STATE),
    }


# Search spaces for nested cross-validation. Deliberately small: with 400
# patients a large grid mostly fits noise, and every candidate is refitted
# inside every outer fold.
PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": [None, "balanced"],
    },
    "Decision Tree": {
        "clf__max_depth": [2, 3, 5, 8, None],
        "clf__min_samples_leaf": [1, 3, 5, 10],
        "clf__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", 0.5],
    },
    "SVM (RBF)": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.1],
        "clf__class_weight": [None, "balanced"],
    },
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
    brier = brier_score_loss(y_true, y_score)

    return dict(TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
                Accuracy=accuracy, Sensitivity=sensitivity,
                Specificity=specificity, Precision=precision,
                F1=f1, NPV=npv, AUC=auc_val, Brier=brier)


# Metrics carried into the summary table, the bootstrap CIs and the report.
SUMMARY_METRICS = ["Accuracy", "Sensitivity", "Specificity", "Precision",
                   "F1", "NPV", "AUC", "Brier"]
COUNT_COLUMNS = ["TP", "TN", "FP", "FN"]


# ----------------------------------------------------------------------------
# STEP 5 -- Decision-threshold selection (fold-safe)
# ----------------------------------------------------------------------------
def select_threshold(y_true, y_score, fn_cost=FN_COST, fp_cost=FP_COST):
    """Threshold minimising `fn_cost * FN + fp_cost * FP`.

    Called ONLY on inner-CV predictions from the training folds, never on the
    held-out fold, so the operating point is chosen without seeing test data.
    Ties are broken towards the lower threshold (higher sensitivity), which is
    the conservative choice for screening.
    """
    y_true = np.asarray(y_true)
    candidates = np.unique(np.concatenate([[0.0], np.asarray(y_score), [1.0]]))

    best_thr, best_cost = 0.5, np.inf
    for thr in candidates:
        pred = (y_score >= thr).astype(int)
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)
    return best_thr


# ----------------------------------------------------------------------------
# STEP 6 -- Cross-validated evaluation
# ----------------------------------------------------------------------------
def evaluate(name, model, X, y, preproc_factory, *, tune=False,
             param_grid=None, seed=RANDOM_STATE, n_splits=N_SPLITS):
    """Outer 5-fold stratified CV.

    Every fold builds a FRESH pipeline (`clone` on the estimator, a new
    preprocessor from the factory), so no state survives between folds.
    `.fit` sees only the training folds: imputation medians, scaler
    means/SDs, one-hot categories, IQR fences, tuned hyperparameters and the
    decision threshold are all learned there.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    per_fold, roc_curves, best_params, thresholds = [], [], [], []
    oof_pred = np.zeros(len(y), dtype=int)
    oof_score = np.zeros(len(y), dtype=float)
    oof_pred_thr = np.zeros(len(y), dtype=int)

    for k, (tr, te) in enumerate(cv.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te = X.iloc[te]

        base = Pipeline([("prep", preproc_factory()), ("clf", clone(model))])

        if tune and param_grid:
            search = GridSearchCV(base, param_grid, cv=inner_cv,
                                  scoring="roc_auc", n_jobs=-1, refit=True)
            search.fit(X_tr, y_tr)
            fitted = search.best_estimator_
            best_params.append({"Fold": k,
                                **{p.replace("clf__", ""): v
                                   for p, v in search.best_params_.items()}})
        else:
            fitted = base.fit(X_tr, y_tr)

        # --- fold-safe threshold: inner CV on the TRAINING folds only --------
        inner_score = cross_val_predict(clone(fitted), X_tr, y_tr,
                                        cv=inner_cv, method="predict_proba",
                                        n_jobs=-1)[:, 1]
        thr = select_threshold(y_tr, inner_score)
        thresholds.append({"Fold": k, "threshold": thr})

        score = fitted.predict_proba(X_te)[:, 1]
        oof_pred[te] = fitted.predict(X_te)
        oof_score[te] = score
        oof_pred_thr[te] = (score >= thr).astype(int)

        m = metrics_from_cm(y.iloc[te], oof_pred[te], score)
        m["Fold"] = k
        m["threshold"] = thr
        per_fold.append(m)

        fpr, tpr, _ = roc_curve(y.iloc[te], score)
        roc_curves.append((fpr, tpr, auc(fpr, tpr)))

    fold_df = pd.DataFrame(per_fold).set_index("Fold")
    pooled = metrics_from_cm(y, oof_pred, oof_score)          # at 0.5
    pooled_thr = metrics_from_cm(y, oof_pred_thr, oof_score)  # at tuned thr

    return dict(name=name, fold_df=fold_df, pooled=pooled,
                pooled_thr=pooled_thr, roc_curves=roc_curves,
                oof_pred=oof_pred, oof_score=oof_score,
                oof_pred_thr=oof_pred_thr,
                thresholds=pd.DataFrame(thresholds).set_index("Fold"),
                best_params=best_params)


def stability_run(models, X, y, preproc_factory, n_repeats, n_splits=N_SPLITS,
                  seed=RANDOM_STATE):
    """RepeatedStratifiedKFold: how much of the headline score is seed luck?

    A single 5-fold split of 400 patients decides the reported score on the
    basis of five 80-patient test sets. Repeating the split `n_repeats` times
    with different shuffles gives the spread of that estimate.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    rows = []
    for name, model in models.items():
        for k, (tr, te) in enumerate(cv.split(X, y), start=1):
            pipe = Pipeline([("prep", preproc_factory()), ("clf", clone(model))])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            score = pipe.predict_proba(X.iloc[te])[:, 1]
            m = metrics_from_cm(y.iloc[te], pipe.predict(X.iloc[te]), score)
            rows.append({"Model": name, "Split": k,
                         **{c: m[c] for c in SUMMARY_METRICS}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# STEP 7 -- Plots
# ----------------------------------------------------------------------------
def plot_confusion(pooled, name, tag, outdir, suffix=""):
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
    title = "Confusion Matrix (pooled out-of-fold)"
    if suffix:
        title += " - screening threshold"
    ax.set_title(f"{title}\n{name}")
    fig.tight_layout()
    fig.savefig(outdir / f"confusion_matrix_{tag}_{slug(name)}{suffix}.png",
                dpi=160)
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


def plot_calibration(all_res, y, tag, outdir, n_bins=8):
    """Are the predicted probabilities usable as probabilities?

    A model can rank patients perfectly (AUC 1.0) and still be badly
    calibrated, which matters as soon as the output is shown to a clinician
    as a risk. SVC probabilities in particular come from Platt scaling fitted
    on an internal split, not from the decision function directly.
    """
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for name, res in all_res.items():
        frac, mean_pred = calibration_curve(y, res["oof_score"], n_bins=n_bins,
                                            strategy="quantile")
        ax.plot(mean_pred, frac, "o-", lw=1.6, ms=4,
                label=f"{name} (Brier = {res['pooled']['Brier']:.4f})")
    ax.set(xlabel="Mean predicted probability of CKD",
           ylabel="Observed fraction of CKD",
           title="Calibration - pooled out-of-fold probabilities")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"calibration_{tag}.png", dpi=160)
    plt.close(fig)


def plot_threshold_curve(res, y, tag, outdir):
    """Screening cost as a function of the decision threshold."""
    score = res["oof_score"]
    thrs = np.linspace(0.01, 0.99, 199)
    costs, sens, spec = [], [], []
    y_arr = np.asarray(y)
    for t in thrs:
        pred = (score >= t).astype(int)
        fn = np.sum((y_arr == 1) & (pred == 0))
        fp = np.sum((y_arr == 0) & (pred == 1))
        tp = np.sum((y_arr == 1) & (pred == 1))
        tn = np.sum((y_arr == 0) & (pred == 0))
        costs.append(FN_COST * fn + FP_COST * fp)
        sens.append(tp / (tp + fn) if (tp + fn) else np.nan)
        spec.append(tn / (tn + fp) if (tn + fp) else np.nan)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(thrs, sens, label="Sensitivity", color="#2e7d32")
    ax.plot(thrs, spec, label="Specificity", color="#1565c0")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Rate")
    ax2 = ax.twinx()
    ax2.plot(thrs, costs, color="#c0392b", ls="--",
             label=f"Cost ({FN_COST:g}xFN + {FP_COST:g}xFP)")
    ax2.set_ylabel("Screening cost")
    mean_thr = res["thresholds"]["threshold"].mean()
    ax.axvline(0.5, color="grey", lw=1, ls=":")
    ax.axvline(mean_thr, color="#c0392b", lw=1)
    ax.set_title(f"Threshold analysis - {res['name']}\n"
                 f"grey = 0.5, red = mean fold-selected {mean_thr:.3f}")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"threshold_{tag}_{slug(res['name'])}.png", dpi=160)
    plt.close(fig)


def plot_model_comparison(summary, tag, outdir):
    metrics = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "AUC"]
    x = np.arange(len(summary)); w = 0.13
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, m in enumerate(metrics):
        ax.bar(x + i * w - 2.5 * w, summary[m], w, label=m)
    ax.set_xticks(x, summary.index, fontsize=9)
    lo = float(np.nanmin(summary[metrics].to_numpy()))
    ax.set_ylim(max(0.0, lo - 0.05), 1.005)
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


def plot_stability(stab, tag, outdir, metric="Accuracy"):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    names = list(dict.fromkeys(stab["Model"]))
    data = [stab.loc[stab["Model"] == n, metric].to_numpy() for n in names]
    ax.boxplot(data, tick_labels=names, showmeans=True)
    n_splits = len(data[0]) if data else 0
    ax.set_ylabel(metric)
    ax.set_title(f"Stability of {metric} across {n_splits} "
                 f"repeated stratified folds")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"stability_{tag}_{metric.lower()}.png", dpi=160)
    plt.close(fig)


def plot_feature_importance(X, y, preproc_factory, outdir, tag):
    """Descriptive Gini importance from a forest fitted on ALL the data.

    Kept for continuity with the original report, and explicitly labelled
    descriptive: it is not an out-of-sample quantity. The honest counterpart
    is `permutation_importance_cv` below.
    """
    pipe = Pipeline([("prep", preproc_factory()),
                     ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1,
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


def permutation_importance_cv(X, y, preproc_factory, outdir, tag,
                              n_repeats=10, seed=RANDOM_STATE):
    """Out-of-fold permutation importance on the ORIGINAL columns.

    Gini importance is computed on training data and is biased towards
    high-cardinality features. Permuting a raw column in the held-out fold and
    measuring the AUC drop answers the question actually being asked: how much
    does this measurement contribute to out-of-sample discrimination?
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=seed)
    per_fold = []
    for tr, te in cv.split(X, y):
        pipe = Pipeline([("prep", preproc_factory()),
                         ("clf", RandomForestClassifier(n_estimators=300,
                                                        n_jobs=-1,
                                                        random_state=seed))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(pipe, X.iloc[te], y.iloc[te],
                                   scoring="roc_auc", n_repeats=n_repeats,
                                   random_state=seed, n_jobs=-1)
        per_fold.append(pd.Series(r.importances_mean, index=X.columns))

    imp = pd.concat(per_fold, axis=1)
    out = pd.DataFrame({"mean_auc_drop": imp.mean(axis=1),
                        "sd_auc_drop": imp.std(axis=1)})
    out = out.sort_values("mean_auc_drop", ascending=False)

    top = out.head(15)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(top.index[::-1], top["mean_auc_drop"].values[::-1],
            xerr=top["sd_auc_drop"].values[::-1], color="#6a1b9a",
            error_kw=dict(lw=0.8, ecolor="grey"))
    ax.set_xlabel("Mean drop in held-out AUC when the column is permuted")
    ax.set_title("Top 15 features - out-of-fold permutation importance")
    fig.tight_layout()
    fig.savefig(outdir / f"permutation_importance_{tag}.png", dpi=160)
    plt.close(fig)
    return out


def slug(s):
    return s.lower().replace(" ", "_").replace("(", "").replace(")", "")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/kidney_disease.csv")
    ap.add_argument("--experiment", choices=list(EXPERIMENTS),
                    default="no_outlier")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--tune", action="store_true",
                    help="nested CV: inner GridSearchCV inside every outer fold")
    ap.add_argument("--stability-repeats", type=int, default=10,
                    help="RepeatedStratifiedKFold repeats (0 disables)")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for confidence intervals (0 disables)")
    args = ap.parse_args()

    cfg = EXPERIMENTS[args.experiment]
    tag = args.experiment + ("_tuned" if args.tune else "")
    outdir = Path(args.outdir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    numeric, categorical = feature_sets(cfg["features"])

    print("=" * 78)
    print(f"CKD CLASSIFICATION  |  experiment = {tag}")
    print("=" * 78)
    print(f"    outlier capping   : {cfg['outliers']}")
    print(f"    feature set       : {cfg['features']} "
          f"({len(numeric)} numeric + {len(categorical)} categorical)")
    print(f"    missing indicators: {cfg['add_indicator']}")
    print(f"    nested CV tuning  : {args.tune}")

    # ---- STEP 1
    df = load_and_clean(args.csv)
    X = df[numeric + categorical]
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
    desc = X[numeric].describe().T
    q1, q3 = X[numeric].quantile(.25), X[numeric].quantile(.75)
    iqr = q3 - q1
    desc["outliers_IQR_1.5"] = (((X[numeric] < q1 - 1.5 * iqr) |
                                 (X[numeric] > q3 + 1.5 * iqr)).sum())
    desc.round(3).to_csv(outdir / "table_descriptive_stats.csv")
    print("\n[3] Numeric summary + IQR outlier counts:")
    print(desc[["mean", "std", "min", "max", "outliers_IQR_1.5"]]
          .round(2).to_string())

    # ---- STEPS 3-6
    def factory():
        return build_preprocessor(cfg["outliers"], numeric, categorical,
                                  cfg["add_indicator"])

    print(f"\n[4] Running {N_SPLITS}-fold Stratified CV ...")

    all_res, rows, thr_rows = {}, [], []
    for name, model in build_models().items():
        res = evaluate(name, model, X, y, factory,
                       tune=args.tune, param_grid=PARAM_GRIDS.get(name))
        all_res[name] = res
        res["fold_df"].round(4).to_csv(outdir / f"table_folds_{slug(name)}.csv")

        p = res["pooled"]
        row = {"Model": name,
               **{k: p[k] for k in SUMMARY_METRICS + COUNT_COLUMNS},
               "AUC_mean_folds": res["fold_df"]["AUC"].mean(),
               "AUC_sd_folds": res["fold_df"]["AUC"].std(),
               "Acc_mean_folds": res["fold_df"]["Accuracy"].mean(),
               "Acc_sd_folds": res["fold_df"]["Accuracy"].std()}

        # ---- bootstrap confidence intervals
        if args.n_boot:
            ci = bootstrap_ci(y, res["oof_pred"], res["oof_score"],
                              metrics_from_cm, SUMMARY_METRICS,
                              n_boot=args.n_boot, random_state=RANDOM_STATE)
            res["ci"] = ci
            for k, (lo, hi) in ci.items():
                row[f"{k}_CI_low"], row[f"{k}_CI_high"] = lo, hi
        rows.append(row)

        # ---- threshold analysis
        pt = res["pooled_thr"]
        thr_rows.append({"Model": name,
                         "mean_threshold": res["thresholds"]["threshold"].mean(),
                         "sd_threshold": res["thresholds"]["threshold"].std(),
                         **{k: pt[k] for k in SUMMARY_METRICS + COUNT_COLUMNS},
                         "screening_cost": FN_COST * pt["FN"] + FP_COST * pt["FP"],
                         "screening_cost_at_0.5": FN_COST * p["FN"] + FP_COST * p["FP"]})

        print(f"\n  --- {name} ---")
        print(res["fold_df"][COUNT_COLUMNS + SUMMARY_METRICS]
              .round(4).to_string())
        print(f"  Pooled OOF @0.5 : Acc={p['Accuracy']:.4f}  "
              f"Sens={p['Sensitivity']:.4f}  Spec={p['Specificity']:.4f}  "
              f"NPV={p['NPV']:.4f}  AUC={p['AUC']:.4f}  Brier={p['Brier']:.4f}")
        print(f"  Pooled OOF @thr : thr={res['thresholds']['threshold'].mean():.3f}  "
              f"Acc={pt['Accuracy']:.4f}  Sens={pt['Sensitivity']:.4f}  "
              f"Spec={pt['Specificity']:.4f}  FN={pt['FN']}  FP={pt['FP']}")

        if args.tune and res["best_params"]:
            bp = pd.DataFrame(res["best_params"]).set_index("Fold")
            bp.to_csv(outdir / f"table_tuned_params_{slug(name)}.csv")
            print("  Selected hyperparameters per outer fold:")
            print(bp.to_string())

        plot_confusion(p, name, tag, outdir)
        plot_confusion(pt, name, tag, outdir, suffix="_threshold")
        plot_roc(res, tag, outdir)
        plot_threshold_curve(res, y, tag, outdir)

    summary = pd.DataFrame(rows).set_index("Model")
    summary.round(4).to_csv(outdir / "table_model_summary.csv")
    plot_model_comparison(summary, tag, outdir)
    plot_calibration(all_res, y, tag, outdir)

    thr_tbl = pd.DataFrame(thr_rows).set_index("Model")
    thr_tbl.round(4).to_csv(outdir / "table_threshold_analysis.csv")

    print("\n[5] SUMMARY (pooled out-of-fold, threshold 0.5, CKD = positive)")
    print(summary[SUMMARY_METRICS + COUNT_COLUMNS].round(4).to_string())

    if args.n_boot:
        print(f"\n[5a] 95% bootstrap CIs ({args.n_boot} resamples)")
        for name in summary.index:
            ci = all_res[name]["ci"]
            bits = "  ".join(
                f"{k} {summary.loc[name, k]:.4f} "
                f"[{ci[k][0]:.4f}, {ci[k][1]:.4f}]"
                for k in ["Accuracy", "Sensitivity", "Specificity", "AUC"])
            print(f"  {name:<20} {bits}")

    # ---- paired significance testing -------------------------------------
    mcn = pairwise_mcnemar(y, {n: all_res[n]["oof_pred"] for n in all_res})
    mcn_df = pd.DataFrame(mcn)
    mcn_df.to_csv(outdir / "table_mcnemar.csv", index=False)
    print("\n[5b] Paired McNemar tests on pooled out-of-fold predictions "
          "(Holm-corrected)")
    print(mcn_df.round(4).to_string(index=False))
    n_sig = int(mcn_df["significant_at_0.05"].sum()) if len(mcn_df) else 0
    print(f"  -> {n_sig} of {len(mcn_df)} model pairs differ significantly "
          f"at alpha = 0.05.")

    print(f"\n[5c] THRESHOLD ANALYSIS (cost = {FN_COST:g}xFN + {FP_COST:g}xFP, "
          f"selected on inner CV of the training folds only)")
    print(thr_tbl[["mean_threshold", "Sensitivity", "Specificity", "NPV",
                   "FN", "FP", "screening_cost", "screening_cost_at_0.5"]]
          .round(4).to_string())

    # ---- stability --------------------------------------------------------
    stab_summary = None
    if args.stability_repeats:
        print(f"\n[5d] STABILITY: {args.stability_repeats} repeats x {N_SPLITS} "
              f"folds = {args.stability_repeats * N_SPLITS} test sets")
        stab = stability_run(build_models(), X, y, factory,
                             args.stability_repeats)
        stab.round(4).to_csv(outdir / "table_stability_folds.csv", index=False)
        stab_summary = (stab.groupby("Model")[SUMMARY_METRICS]
                        .agg(["mean", "std", "min", "max"]))
        stab_summary.columns = [f"{m}_{s}" for m, s in stab_summary.columns]
        stab_summary.round(4).to_csv(outdir / "table_stability_summary.csv")
        show = [f"{m}_{s}" for m in ("Accuracy", "Sensitivity", "AUC")
                for s in ("mean", "std", "min", "max")]
        print(stab_summary[show].round(4).to_string())
        plot_stability(stab, tag, outdir, "Accuracy")
        plot_stability(stab, tag, outdir, "Sensitivity")

    # ---- feature importance ----------------------------------------------
    imp = plot_feature_importance(X, y, factory, outdir, tag)
    imp.round(4).to_csv(outdir / "table_feature_importance.csv",
                        header=["importance"])
    print("\n[6] Top 10 features (Random Forest Gini, full data - descriptive):")
    print(imp.head(10).round(4).to_string())

    perm = permutation_importance_cv(X, y, factory, outdir, tag)
    perm.round(5).to_csv(outdir / "table_permutation_importance.csv")
    print("\n[6a] Top 10 features (out-of-fold permutation importance, AUC drop):")
    print(perm.head(10).round(5).to_string())

    best = summary["AUC"].idxmax()
    print(f"\n[7] Best model by pooled AUC: {best} "
          f"(AUC = {summary.loc[best, 'AUC']:.4f})")
    if n_sig == 0:
        print("    NOTE: no pairwise difference is statistically significant; "
              "'best' here is a ranking, not a demonstrated superiority.")

    with open(outdir / "summary.json", "w") as f:
        json.dump({"experiment": tag,
                   "config": cfg,
                   "n_rows": int(len(df)),
                   "n_ckd": int((y == 1).sum()),
                   "n_notckd": int((y == 0).sum()),
                   "n_features_numeric": len(numeric),
                   "n_features_categorical": len(categorical),
                   "tuned": bool(args.tune),
                   "cost_model": {"FN": FN_COST, "FP": FP_COST},
                   "best_model": best,
                   "n_significant_pairs": n_sig,
                   "environment": {
                       "python": platform.python_version(),
                       "scikit_learn": sklearn.__version__,
                       "pandas": pd.__version__,
                       "numpy": np.__version__,
                   },
                   "results": summary.round(6).to_dict(orient="index"),
                   "threshold_analysis": thr_tbl.round(6).to_dict(orient="index"),
                   "mcnemar": mcn,
                   "stability": (stab_summary.round(6).to_dict(orient="index")
                                 if stab_summary is not None else None)},
                  f, indent=2, default=str)
    print(f"\nAll tables + figures written to: {outdir}/")


if __name__ == "__main__":
    main()
