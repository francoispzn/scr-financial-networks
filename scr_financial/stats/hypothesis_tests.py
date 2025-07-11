"""Hypothesis tests for model comparison."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Union


def welch_t_test(sample1, sample2) -> Dict[str, float]:
    """Welch's t-test for unequal variances."""
    s1, s2 = np.asarray(sample1), np.asarray(sample2)
    t_stat, p_value = stats.ttest_ind(s1, s2, equal_var=False)
    pooled_std = np.sqrt((np.var(s1, ddof=1) + np.var(s2, ddof=1)) / 2)
    cohens_d = (np.mean(s1) - np.mean(s2)) / pooled_std if pooled_std > 0 else 0.0
    diff = np.mean(s1) - np.mean(s2)
    se_diff = np.sqrt(np.var(s1, ddof=1)/len(s1) + np.var(s2, ddof=1)/len(s2))
    ci_lower = diff - 1.96 * se_diff
    ci_upper = diff + 1.96 * se_diff
    return {"t_stat": float(t_stat), "p_value": float(p_value),
            "cohens_d": float(cohens_d), "mean_diff": float(diff),
            "ci_95_diff": (float(ci_lower), float(ci_upper))}


def diebold_mariano_test(errors1, errors2, horizon=1, loss='mse') -> Dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy."""
    e1, e2 = np.asarray(errors1), np.asarray(errors2)
    if loss == 'mse':
        d = e1**2 - e2**2
    elif loss == 'mae':
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1 - e2

    n = len(d)
    d_mean = np.mean(d)

    # HAC variance (Newey-West for h > 1)
    if horizon <= 1:
        d_var = np.var(d, ddof=1) / n
    else:
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0.0
        for k in range(1, horizon):
            gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d) > k else 0
            gamma_sum += 2 * gamma_k
        d_var = (gamma_0 + gamma_sum) / n

    dm_stat = d_mean / np.sqrt(max(d_var, 1e-15))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value),
            "mean_loss_diff": float(d_mean), "n": n}


def paired_permutation_test(metric1, metric2, n_perms=10000, seed=42) -> Dict[str, float]:
    """Non-parametric paired permutation test."""
    m1, m2 = np.asarray(metric1), np.asarray(metric2)
    observed = np.mean(m1) - np.mean(m2)
    rng = np.random.default_rng(seed)
    count = 0
    diffs = m1 - m2
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed):
            count += 1
    p_value = count / n_perms
    return {"observed_diff": float(observed), "p_value": float(p_value), "n_perms": n_perms}


def multiple_testing_correction(p_values, method='holm', alpha=0.05) -> Dict:
    """Multiple testing correction."""
    pvals = np.asarray(p_values, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]

    if method == 'bonferroni':
        corrected = np.minimum(sorted_p * n, 1.0)
    elif method == 'holm':
        corrected = np.zeros(n)
        for i in range(n):
            corrected[i] = sorted_p[i] * (n - i)
        for i in range(1, n):
            corrected[i] = max(corrected[i], corrected[i-1])
        corrected = np.minimum(corrected, 1.0)
    elif method == 'bh':
        corrected = np.zeros(n)
        for i in range(n-1, -1, -1):
            corrected[i] = sorted_p[i] * n / (i + 1)
        for i in range(n-2, -1, -1):
            corrected[i] = min(corrected[i], corrected[i+1])
        corrected = np.minimum(corrected, 1.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Unsort
    result = np.zeros(n)
    result[order] = corrected
    rejected = result < alpha
    return {"corrected": result.tolist(), "rejected": rejected.tolist(), "method": method, "alpha": alpha}


def model_comparison_table(results_dict, baseline_key=None, alpha=0.05):
    """Generate comparison DataFrame with statistical tests."""
    rows = []
    models = list(results_dict.keys())
    if baseline_key is None:
        baseline_key = models[0]

    baseline_vals = np.asarray(results_dict[baseline_key])
    for model_name, values in results_dict.items():
        vals = np.asarray(values)
        row = {"model": model_name, "mean": np.mean(vals), "std": np.std(vals),
               "median": np.median(vals), "n": len(vals)}
        if model_name != baseline_key:
            wt = welch_t_test(vals, baseline_vals)
            row["vs_baseline_p"] = wt["p_value"]
            row["vs_baseline_d"] = wt["cohens_d"]
            row["significant"] = wt["p_value"] < alpha
        else:
            row["vs_baseline_p"] = None
            row["vs_baseline_d"] = None
            row["significant"] = None
        rows.append(row)
    return pd.DataFrame(rows)
