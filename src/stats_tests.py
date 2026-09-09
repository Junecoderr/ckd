"""Uncertainty quantification and paired significance testing.

The main pipeline reports point estimates. With n = 400 and only a handful of
misclassified patients, a difference of 0.005 in accuracy is two patients --
well inside sampling noise. This module supplies the two tools needed to say
so quantitatively:

* `bootstrap_ci`  -- percentile bootstrap confidence intervals for any metric
                     computed from pooled out-of-fold predictions.
* `mcnemar_exact` -- exact McNemar test for two classifiers evaluated on the
                     *same* patients (which is exactly our situation: every
                     model produces one out-of-fold prediction per patient).
* `holm_adjust`   -- Holm-Bonferroni step-down correction, because comparing
                     four models means six pairwise tests.

Only numpy/scipy are required; scipy is already an installation dependency of
scikit-learn, so this adds nothing to the environment.
"""

from itertools import combinations

import numpy as np
from scipy.stats import binomtest


# ----------------------------------------------------------------------------
# Bootstrap confidence intervals
# ----------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, y_score, metric_fn, keys,
                 n_boot=2000, alpha=0.05, random_state=42):
    """Percentile bootstrap CI for every metric returned by `metric_fn`.

    Patients are resampled with replacement (n out of n) and the metrics are
    recomputed on each resample. Resamples that end up single-class are
    discarded, since Sensitivity/Specificity/AUC are undefined there.

    Caveat, stated plainly: this quantifies the uncertainty of *evaluating*
    a fixed set of out-of-fold predictions on this cohort. It does not capture
    the extra variability that comes from refitting the models on a different
    sample of patients, so it is a lower bound on total uncertainty.

    Returns {metric_name: (low, high)}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    draws = {k: [] for k in keys}
    attempts = 0
    while len(draws[keys[0]]) < n_boot and attempts < n_boot * 10:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.min() == yt.max():          # degenerate single-class resample
            continue
        m = metric_fn(yt, y_pred[idx], y_score[idx])
        for k in keys:
            draws[k].append(m[k])

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q)))
            for k, v in draws.items()}


# ----------------------------------------------------------------------------
# Exact McNemar test for paired classifiers
# ----------------------------------------------------------------------------
def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact (binomial) McNemar test on two sets of paired predictions.

    Only the *discordant* patients carry information:
        n_ab = A wrong, B right
        n_ba = A right, B wrong
    Under H0 (equal error rates) each discordant patient is a fair coin, so
    n_ba ~ Binomial(n_ab + n_ba, 0.5). The exact binomial test is used rather
    than the chi-square approximation because the discordant counts here are
    tiny (single digits), where the approximation is unreliable.

    Returns dict with the discordant counts and the two-sided p-value.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true

    n_ab = int(np.sum(~a_ok & b_ok))      # only B got these right
    n_ba = int(np.sum(a_ok & ~b_ok))      # only A got these right
    n_disc = n_ab + n_ba

    if n_disc == 0:
        # The two models made identical predictions on every patient.
        p = 1.0
    else:
        p = float(binomtest(n_ba, n_disc, 0.5, alternative="two-sided").pvalue)

    return {"n_discordant": n_disc,
            "n_only_A_correct": n_ba,
            "n_only_B_correct": n_ab,
            "p_value": p}


def holm_adjust(pvals):
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)

    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def pairwise_mcnemar(y_true, predictions):
    """Run `mcnemar_exact` for every pair in {name: pred} and Holm-correct.

    Returns a list of dicts, one row per pair, ready for a DataFrame.
    """
    names = list(predictions)
    rows = []
    for a, b in combinations(names, 2):
        r = mcnemar_exact(y_true, predictions[a], predictions[b])
        rows.append({"Model_A": a, "Model_B": b, **r})

    if rows:
        adj = holm_adjust([r["p_value"] for r in rows])
        for r, q in zip(rows, adj):
            r["p_holm"] = float(q)
            r["significant_at_0.05"] = bool(q < 0.05)
    return rows
