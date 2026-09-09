"""Tests for the CKD pipeline.

Two things are worth testing in a project whose headline claim is "no data
leakage":

1. The structural cleaning really does repair the four documented defects.
2. Nothing learned from a held-out fold can reach the model -- asserted by
   fitting a preprocessor on a subset and checking that the statistics it
   learned come from that subset alone.

Run with:  PYTHONPATH=src pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_pipeline import (CATEGORICAL, DIAGNOSTIC_CRITERIA, NUMERIC,
                          RANDOM_STATE, TARGET, IQRWinsorizer,
                          build_preprocessor, feature_sets, load_and_clean,
                          metrics_from_cm, select_threshold)
from stats_tests import holm_adjust, mcnemar_exact

CSV = ROOT / "data" / "kidney_disease.csv"


@pytest.fixture(scope="module")
def df():
    return load_and_clean(CSV)


# ----------------------------------------------------------------------------
# Cleaning: the four defects documented in REPORT.md section 1.2
# ----------------------------------------------------------------------------
def test_shape_is_400_by_25(df):
    assert df.shape == (400, 25)          # 24 features + target, `id` dropped
    assert "id" not in df.columns


def test_target_is_binary_250_150(df):
    assert set(df[TARGET].unique()) == {0, 1}
    assert int((df[TARGET] == 1).sum()) == 250      # ckd  -> positive class
    assert int((df[TARGET] == 0).sum()) == 150      # notckd
    assert df[TARGET].dtype.kind in "iu"


def test_no_stray_whitespace_survives_cleaning(df):
    """Defect 1-3: 'ckd\\t', '\\tyes', ' yes', '\\tno' must all be collapsed."""
    for col in CATEGORICAL:
        vals = df[col].dropna().unique()
        for v in vals:
            assert v == v.strip(), f"{col}: {v!r} still has stray whitespace"
            assert v == v.lower(), f"{col}: {v!r} was not lower-cased"
    assert set(df["dm"].dropna().unique()) == {"yes", "no"}
    assert set(df["cad"].dropna().unique()) == {"yes", "no"}


def test_numeric_columns_are_numeric(df):
    """Defect 4: pcv, wc, rc arrive as text because of the '\\t?' cells."""
    for col in NUMERIC:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"
    for col in ("pcv", "wc", "rc"):
        assert df[col].notna().sum() > 0


def test_placeholders_became_missing(df):
    assert not (df.astype(str) == "?").to_numpy().any()
    assert df.isna().sum().sum() == 1012           # documented in REPORT.md


def test_no_duplicate_records(df):
    assert df.duplicated().sum() == 0


# ----------------------------------------------------------------------------
# Leakage control
# ----------------------------------------------------------------------------
def test_imputer_learns_from_training_rows_only(df):
    """The median used to fill a held-out row must come from the train split."""
    X = df[NUMERIC + CATEGORICAL]
    train, test = X.iloc[:300], X.iloc[300:]

    prep = build_preprocessor(False)
    prep.fit(train)

    learned = prep.named_transformers_["num"].named_steps["impute"].statistics_
    expected_train = train[NUMERIC].median().to_numpy()
    expected_full = X[NUMERIC].median().to_numpy()

    np.testing.assert_allclose(learned, expected_train)
    # And it must NOT be the full-data median, or the split bought us nothing.
    assert not np.allclose(learned, expected_full)
    prep.transform(test)                   # transform-only path must work


def test_winsorizer_fences_come_from_fit_data_only():
    rng = np.random.default_rng(RANDOM_STATE)
    train = rng.normal(0, 1, size=(200, 3))
    test = np.full((5, 3), 1e6)            # extreme, unseen values

    w = IQRWinsorizer(factor=1.5).fit(train)
    q1, q3 = np.percentile(train, [25, 75], axis=0)
    np.testing.assert_allclose(w.upper_, q3 + 1.5 * (q3 - q1))
    np.testing.assert_allclose(w.lower_, q1 - 1.5 * (q3 - q1))

    out = w.transform(test)
    # Held-out extremes are capped at the TRAINING fences, and fitting on the
    # test block would have produced completely different fences.
    np.testing.assert_allclose(out, np.tile(w.upper_, (5, 1)))
    assert not np.allclose(IQRWinsorizer().fit(test).upper_, w.upper_)


def test_winsorizer_does_not_change_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    assert IQRWinsorizer().fit(X).transform(X).shape == X.shape


def test_missing_indicator_adds_columns(df):
    X = df[NUMERIC + CATEGORICAL]
    plain = build_preprocessor(False).fit_transform(X)
    flagged = build_preprocessor(False, add_indicator=True).fit_transform(X)
    assert flagged.shape[1] > plain.shape[1]
    assert flagged.shape[0] == plain.shape[0]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def test_confusion_matrix_orientation():
    """CKD = 1 is positive; sensitivity must count caught CKD patients."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    m = metrics_from_cm(y_true, y_pred, y_pred.astype(float))

    assert (m["TP"], m["FN"], m["TN"], m["FP"]) == (3, 1, 3, 1)
    assert m["Sensitivity"] == pytest.approx(3 / 4)
    assert m["Specificity"] == pytest.approx(3 / 4)
    assert m["Precision"] == pytest.approx(3 / 4)
    assert m["NPV"] == pytest.approx(3 / 4)
    assert m["Accuracy"] == pytest.approx(6 / 8)


def test_npv_is_reported():
    """NPV was computed but silently dropped from the summary before; guard it."""
    from ckd_pipeline import SUMMARY_METRICS
    assert "NPV" in SUMMARY_METRICS
    assert "Brier" in SUMMARY_METRICS


def test_perfect_and_useless_classifiers():
    y = np.array([1, 1, 0, 0])
    perfect = metrics_from_cm(y, y, y.astype(float))
    assert perfect["Accuracy"] == 1.0 and perfect["AUC"] == 1.0
    flipped = metrics_from_cm(y, 1 - y, (1 - y).astype(float))
    assert flipped["Accuracy"] == 0.0 and flipped["AUC"] == 0.0


# ----------------------------------------------------------------------------
# Threshold selection
# ----------------------------------------------------------------------------
def test_threshold_moves_down_when_false_negatives_cost_more():
    y = np.array([1] * 10 + [0] * 10)
    score = np.concatenate([np.linspace(0.35, 0.95, 10),      # CKD patients
                            np.linspace(0.05, 0.45, 10)])     # healthy
    cheap_fn = select_threshold(y, score, fn_cost=1, fp_cost=1)
    dear_fn = select_threshold(y, score, fn_cost=20, fp_cost=1)
    assert dear_fn <= cheap_fn


def test_threshold_catches_every_case_when_misses_are_ruinous():
    y = np.array([1, 1, 0, 0])
    score = np.array([0.9, 0.2, 0.1, 0.05])
    thr = select_threshold(y, score, fn_cost=1000, fp_cost=1)
    assert np.sum((score >= thr) & (y == 1)) == 2      # no false negatives


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def test_mcnemar_identical_predictions_is_not_significant():
    y = np.array([1, 0, 1, 0, 1])
    pred = np.array([1, 0, 0, 0, 1])
    r = mcnemar_exact(y, pred, pred)
    assert r["n_discordant"] == 0 and r["p_value"] == 1.0


def test_mcnemar_counts_discordant_pairs_only():
    y = np.array([1, 1, 1, 1])
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 1, 1, 1])
    r = mcnemar_exact(y, a, b)
    assert r["n_discordant"] == 2
    assert r["n_only_B_correct"] == 2 and r["n_only_A_correct"] == 0


def test_holm_is_monotone_and_no_smaller_than_raw():
    raw = [0.001, 0.02, 0.03, 0.5]
    adj = holm_adjust(raw)
    assert np.all(adj >= np.array(raw))
    assert np.all(np.diff(adj) >= 0)
    assert np.all(adj <= 1.0)


# ----------------------------------------------------------------------------
# Ablation feature set
# ----------------------------------------------------------------------------
def test_ablation_removes_every_diagnostic_column():
    num, cat = feature_sets("non_diagnostic")
    assert not set(num + cat) & set(DIAGNOSTIC_CRITERIA)
    assert len(num + cat) == len(NUMERIC + CATEGORICAL) - len(DIAGNOSTIC_CRITERIA)


def test_all_feature_set_is_unchanged():
    num, cat = feature_sets("all")
    assert num == NUMERIC and cat == CATEGORICAL
