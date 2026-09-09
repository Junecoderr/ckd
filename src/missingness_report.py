"""Is the *pattern* of missing data itself predictive of CKD?

Motivation
----------
`rbc` is missing for 38% of patients, `rc` for 33%, `wc` for 27%. Values are
not missing at random in a hospital record: a test is ordered when a clinician
suspects something. If which tests were run correlates with who turned out to
have CKD, then part of any model's accuracy comes from the data-collection
process rather than from physiology -- and would not survive a prospective
deployment where every patient gets the same panel.

This script answers the question two ways, and neither uses a single measured
*value*:

1. Per column, the missingness rate in CKD patients vs. non-CKD patients, with
   a Fisher exact test and Holm correction across the 24 features.
2. A logistic regression trained on NOTHING BUT the 24 binary was-this-
   measured flags, evaluated with the same leak-free 5-fold stratified CV as
   the main pipeline. An AUC near 0.5 would mean missingness is uninformative;
   anything higher is the collection artefact made explicit.

Usage:
    python src/missingness_report.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckd_pipeline import (CATEGORICAL, NUMERIC, N_SPLITS, RANDOM_STATE, TARGET,
                          load_and_clean, metrics_from_cm)
from stats_tests import holm_adjust


def missingness_by_class(df):
    """Missingness rate per feature, split by outcome, with Fisher exact p."""
    features = NUMERIC + CATEGORICAL
    y = df[TARGET]
    rows = []
    for col in features:
        miss = df[col].isna()
        a = int((miss & (y == 1)).sum())      # missing, CKD
        b = int((~miss & (y == 1)).sum())     # present, CKD
        c = int((miss & (y == 0)).sum())      # missing, notckd
        d = int((~miss & (y == 0)).sum())     # present, notckd
        p = 1.0 if (a + c) == 0 else float(fisher_exact([[a, b], [c, d]])[1])
        rows.append({"feature": col,
                     "missing_total": a + c,
                     "pct_missing_ckd": 100 * a / max(a + b, 1),
                     "pct_missing_notckd": 100 * c / max(c + d, 1),
                     "p_fisher": p})

    out = pd.DataFrame(rows).set_index("feature")
    out["difference_pp"] = out["pct_missing_ckd"] - out["pct_missing_notckd"]
    out["p_holm"] = holm_adjust(out["p_fisher"].to_numpy())
    out["significant_at_0.05"] = out["p_holm"] < 0.05
    return out.sort_values("p_holm")


def missingness_only_model(df, seed=RANDOM_STATE):
    """Predict CKD from the was-this-measured flags alone. No values used."""
    features = NUMERIC + CATEGORICAL
    M = df[features].isna().astype(int)
    M = M.loc[:, M.sum() > 0]                 # drop never-missing columns
    y = df[TARGET]

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=5000, random_state=seed)
    score = cross_val_predict(clf, M, y, cv=cv, method="predict_proba")[:, 1]
    pred = (score >= 0.5).astype(int)
    return M.columns.tolist(), metrics_from_cm(y, pred, score)


def plot_missingness_by_class(tbl, outdir):
    top = tbl.sort_values("difference_pp", key=abs, ascending=False).head(15)
    idx = np.arange(len(top))[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.barh(idx + 0.19, top["pct_missing_ckd"], 0.38, label="CKD",
            color="#c0392b")
    ax.barh(idx - 0.19, top["pct_missing_notckd"], 0.38, label="not CKD",
            color="#1565c0")
    ax.set_yticks(idx, top.index)
    ax.set_xlabel("% of patients for whom this measurement is missing")
    ax.set_title("Missingness rate by outcome\n"
                 "(15 features with the largest gap)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "missingness_by_class.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/kidney_disease.csv")
    ap.add_argument("--outdir", default="results/missingness")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(args.csv)

    print("=" * 78)
    print("IS MISSINGNESS ITSELF PREDICTIVE?")
    print("=" * 78)

    tbl = missingness_by_class(df)
    tbl.round(4).to_csv(outdir / "table_missingness_by_class.csv")
    plot_missingness_by_class(tbl, outdir)

    n_sig = int(tbl["significant_at_0.05"].sum())
    print(f"\n[1] Missingness rate by outcome ({n_sig} of {len(tbl)} features "
          f"differ significantly, Holm-corrected):")
    print(tbl.head(12)[["missing_total", "pct_missing_ckd",
                        "pct_missing_notckd", "difference_pp", "p_holm",
                        "significant_at_0.05"]].round(4).to_string())

    cols, m = missingness_only_model(df)
    print(f"\n[2] Logistic regression on the {len(cols)} was-measured flags "
          f"ONLY (no measured values at all), 5-fold stratified CV:")
    print(f"    Accuracy    {m['Accuracy']:.4f}")
    print(f"    Sensitivity {m['Sensitivity']:.4f}")
    print(f"    Specificity {m['Specificity']:.4f}")
    print(f"    AUC         {m['AUC']:.4f}   (0.5 = missingness is uninformative)")
    print(f"    TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}")

    with open(outdir / "summary.json", "w") as f:
        json.dump({"n_features_tested": int(len(tbl)),
                   "n_significant": n_sig,
                   "missingness_only_model": {
                       "features": cols,
                       **{k: (float(v) if isinstance(v, float) else v)
                          for k, v in m.items()}}},
                  f, indent=2)
    print(f"\nWritten to: {outdir}/")


if __name__ == "__main__":
    main()
