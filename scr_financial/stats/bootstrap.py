"""Bootstrap confidence interval methods."""

import numpy as np
from typing import Callable, Dict, Optional


def bootstrap_ci(data, stat_fn=None, n_boot=10000, alpha=0.05,
                 method='percentile', seed=42) -> Dict:
    """Bootstrap confidence interval for a statistic."""
    if stat_fn is None:
        stat_fn = np.mean
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    estimate = float(stat_fn(data))

    boot_stats = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])

    if method == 'bca':
        # Bias-corrected accelerated
        from scipy.stats import norm
        z0 = norm.ppf(np.mean(boot_stats < estimate))
        # Jackknife for acceleration
        n = len(data)
        jack_stats = np.array([stat_fn(np.delete(data, i)) for i in range(n)])
        jack_mean = np.mean(jack_stats)
        num = np.sum((jack_mean - jack_stats) ** 3)
        den = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
        a = num / den if den > 0 else 0

        z_alpha = norm.ppf(alpha / 2)
        z_1alpha = norm.ppf(1 - alpha / 2)
        alpha1 = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        alpha2 = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))
        ci_lower = float(np.percentile(boot_stats, 100 * alpha1))
        ci_upper = float(np.percentile(boot_stats, 100 * alpha2))
    else:
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {"estimate": estimate, "ci_lower": ci_lower, "ci_upper": ci_upper,
            "se": float(np.std(boot_stats)), "method": method, "n_boot": n_boot}


def block_bootstrap_ci(series, stat_fn=None, n_boot=5000,
                       block_size=None, alpha=0.05, seed=42) -> Dict:
    """Block bootstrap for time-series data."""
    if stat_fn is None:
        stat_fn = np.mean
    series = np.asarray(series)
    n = len(series)
    rng = np.random.default_rng(seed)

    if block_size is None:
        block_size = max(1, int(np.ceil((3 * n) ** (1/3))))

    estimate = float(stat_fn(series))
    boot_stats = []
    for _ in range(n_boot):
        indices = []
        while len(indices) < n:
            start = rng.integers(0, n - block_size + 1) if n > block_size else 0
            indices.extend(range(start, min(start + block_size, n)))
        indices = indices[:n]
        boot_stats.append(stat_fn(series[indices]))

    boot_stats = np.array(boot_stats)
    return {"estimate": estimate,
            "ci_lower": float(np.percentile(boot_stats, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
            "se": float(np.std(boot_stats)),
            "block_size": block_size, "n_boot": n_boot}


def bootstrap_r2_ci(y_true, y_pred, n_boot=10000, alpha=0.05, seed=42) -> Dict:
    """Bootstrap CI for R-squared."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    def r2(yt, yp):
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    estimate = r2(y_true, y_pred)
    boot_r2s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_r2s.append(r2(y_true[idx], y_pred[idx]))

    boot_r2s = np.array(boot_r2s)
    return {"estimate": float(estimate),
            "ci_lower": float(np.percentile(boot_r2s, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_r2s, 100 * (1 - alpha / 2))),
            "se": float(np.std(boot_r2s)), "n_boot": n_boot}


def bootstrap_correlation_ci(x, y, n_boot=10000, alpha=0.05, seed=42) -> Dict:
    """Bootstrap CI for Pearson correlation with permutation p-value."""
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    rng = np.random.default_rng(seed)

    r = float(np.corrcoef(x, y)[0, 1])

    # Bootstrap CI
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rs.append(np.corrcoef(x[idx], y[idx])[0, 1])
    boot_rs = np.array(boot_rs)

    # Permutation p-value
    n_perm = min(n_boot, 5000)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        r_perm = np.corrcoef(x, y_perm)[0, 1]
        if abs(r_perm) >= abs(r):
            count += 1

    return {"r": r,
            "ci_lower": float(np.percentile(boot_rs, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_rs, 100 * (1 - alpha / 2))),
            "se": float(np.std(boot_rs)),
            "p_value": float(count / n_perm), "n_boot": n_boot}
