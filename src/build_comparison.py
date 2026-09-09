"""Assemble the side-by-side table across every experiment that has been run.

Replaces the hand-maintained results/table_experiment_comparison.csv: it reads
whatever results/<experiment>/table_model_summary.csv files exist and joins
them, so the comparison can never silently disagree with the per-experiment
numbers it is summarising.

Usage:
    python src/build_comparison.py
"""

import argparse
from pathlib import Path

import pandas as pd

# Ordered so the primary experiment comes first in the output columns.
EXPERIMENT_LABELS = [
    ("no_outlier",        "Exp1_no_outlier"),
    ("outlier_capped",    "Exp2_iqr_capped"),
    ("ablation",          "Exp3_ablation"),
    ("missing_indicator", "Exp4_missing_indicator"),
    ("no_outlier_tuned",  "Exp5_nested_cv_tuned"),
]

METRICS = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "NPV",
           "AUC", "Brier"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)

    frames, found = [], []
    for folder, label in EXPERIMENT_LABELS:
        path = root / folder / "table_model_summary.csv"
        if not path.exists():
            print(f"  (skipping {folder}: not run yet)")
            continue
        df = pd.read_csv(path, index_col="Model")
        cols = [c for c in METRICS if c in df.columns]
        sub = df[cols].copy()
        sub.columns = pd.MultiIndex.from_product([[label], cols])
        frames.append(sub)
        found.append(label)

    if not frames:
        raise SystemExit("No experiment summaries found. Run the pipeline first.")

    out = pd.concat(frames, axis=1)
    dest = root / "table_experiment_comparison.csv"
    out.round(4).to_csv(dest)

    print(f"Comparison across {len(found)} experiments -> {dest}")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
