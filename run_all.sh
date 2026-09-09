#!/usr/bin/env bash
# Regenerate every table and figure in results/ from data/kidney_disease.csv.
# Equivalent to `make all`, for anyone without make.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CSV="${CSV:-data/kidney_disease.csv}"
mkdir -p logs

run () {
    local name="$1"; shift
    echo "==> $name"
    "$PYTHON" "$@" | tee "logs/run_${name}.txt" > /dev/null
}

run no_outlier        src/ckd_pipeline.py --csv "$CSV" --experiment no_outlier
run outlier_capped    src/ckd_pipeline.py --csv "$CSV" --experiment outlier_capped
run ablation          src/ckd_pipeline.py --csv "$CSV" --experiment ablation
run missing_indicator src/ckd_pipeline.py --csv "$CSV" --experiment missing_indicator
run no_outlier_tuned  src/ckd_pipeline.py --csv "$CSV" --experiment no_outlier \
                          --tune --stability-repeats 0
run missingness       src/missingness_report.py --csv "$CSV"

echo "==> comparison table"
"$PYTHON" src/build_comparison.py

echo
echo "Done. Tables and figures are in results/, console logs in logs/."
