"""Conditional spectral risk indicators."""

import numpy as np
from typing import Dict


def vol_regressed_lambda2(lambda2_series, vol_series) -> Dict:
    """Regress lambda_2 on volatility, return residuals.

    Residuals capture structural connectivity changes not explained by market conditions.
    """
    lam2 = np.asarray(lambda2_series)
    vol = np.asarray(vol_series)
    X = np.column_stack([np.ones(len(vol)), vol])
    try:
        coefs = np.linalg.lstsq(X, lam2, rcond=None)[0]
        predicted = X @ coefs
        residuals = lam2 - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((lam2 - np.mean(lam2)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    except Exception:
        coefs = np.array([np.mean(lam2), 0])
        residuals = lam2 - np.mean(lam2)
        r_squared = 0

    return {"residuals": residuals, "r_squared": float(r_squared),
            "intercept": float(coefs[0]), "beta": float(coefs[1])}


def marchenko_pastur_relative_connectivity(lambda2_series, T, N) -> np.ndarray:
    """Compute lambda_2 relative to MP upper bound.

    Values > 1 indicate genuine structural connectivity beyond noise.
    """
    from .rmt import marchenko_pastur_bounds
    _, lambda_plus = marchenko_pastur_bounds(T, N)
    return np.asarray(lambda2_series) / lambda_plus


def conditional_scg_risk(lambda2, rho, vol) -> np.ndarray:
    """Conditional spectral risk: high when both connectivity AND volatility are high.

    Standard SCG risk (1 - lambda_2/rho) fails because it captures fragmentation, not stress.
    This indicator captures correlated stress: high lambda_2 (banks moving together)
    combined with high volatility (banks moving a lot).
    """
    lam2 = np.asarray(lambda2)
    v = np.asarray(vol)
    v_norm = v / np.max(v) if np.max(v) > 0 else v
    lam2_norm = lam2 / np.max(lam2) if np.max(lam2) > 0 else lam2
    return lam2_norm * v_norm


def spectral_momentum(lambda2_series, window=20) -> Dict:
    """Rate of change of lambda_2 (z-scored rolling derivative).

    Sharp drops indicate rapid network fragmentation — potential stress signal.
    """
    lam2 = np.asarray(lambda2_series, dtype=float)
    n = len(lam2)

    if n < window + 1:
        return {"derivatives": np.zeros(n), "z_scores": np.zeros(n), "alerts": []}

    derivatives = np.zeros(n)
    derivatives[1:] = np.diff(lam2)

    # Rolling z-score
    z_scores = np.zeros(n)
    for i in range(window, n):
        w = derivatives[i - window:i]
        mu, sigma = np.mean(w), np.std(w)
        z_scores[i] = (derivatives[i] - mu) / sigma if sigma > 1e-10 else 0

    alerts = [int(i) for i in range(n) if abs(z_scores[i]) > 2.0]
    return {"derivatives": derivatives, "z_scores": z_scores, "alerts": alerts}
