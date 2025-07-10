"""
Systemic risk metrics: CoVaR, ΔCoVaR, MES, SRISK.

These implement the standard tail-risk measures used as benchmarks against
the Spectral Coarse-Graining (SCG) risk indicator.

References:
  - Adrian & Brunnermeier (2016) — CoVaR
  - Acharya et al. (2017) — MES / SRISK
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Value-at-Risk at level alpha (left tail)."""
    return float(np.percentile(returns, alpha * 100))


def compute_covar(
    bank_returns: np.ndarray,
    system_returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """CoVaR: VaR of a bank conditional on the system being at its VaR.

    Uses the historical simulation approach: subset bank returns to periods
    where system return <= system VaR, then take the alpha-quantile.
    """
    sys_var = compute_var(system_returns, alpha)
    # Periods when system is in distress
    mask = system_returns <= sys_var
    if mask.sum() < 3:
        return compute_var(bank_returns, alpha)
    return float(np.percentile(bank_returns[mask], alpha * 100))


def compute_delta_covar(
    bank_returns: np.ndarray,
    system_returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """ΔCoVaR: difference between CoVaR under stress and CoVaR at median.

    Positive ΔCoVaR means the bank contributes more risk when the system
    is stressed.
    """
    covar_stress = compute_covar(bank_returns, system_returns, alpha)
    # CoVaR at median state: system near its 50th percentile (±5pp)
    sys_med = np.median(system_returns)
    band = np.percentile(np.abs(system_returns - sys_med), 10)
    mask_median = np.abs(system_returns - sys_med) <= max(band, 1e-10)
    if mask_median.sum() < 3:
        covar_median = compute_var(bank_returns, 0.5)
    else:
        covar_median = float(np.percentile(bank_returns[mask_median], alpha * 100))
    return covar_stress - covar_median


def compute_mes(
    bank_returns: np.ndarray,
    system_returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Marginal Expected Shortfall: E[r_bank | r_system < VaR_alpha(system)].

    Measures a bank's expected loss conditional on a systemic tail event.
    """
    sys_var = compute_var(system_returns, alpha)
    mask = system_returns <= sys_var
    if mask.sum() < 1:
        return 0.0
    return float(np.mean(bank_returns[mask]))


def compute_srisk(
    mes: float,
    equity: float,
    liabilities: float,
    k: float = 0.08,
) -> float:
    """SRISK: expected capital shortfall of a bank in a systemic crisis.

    Following Brownlees & Engle (2017):
        SRISK = max(0, k*D - (1-k)*W*(1 - LRMES))

    where D = liabilities (book value of debt), W = equity (market cap),
    k = prudential capital ratio (default 8%), and LRMES is the long-run
    marginal expected shortfall. Since MES is typically negative for losses,
    we use (1 + MES) as the equity decline factor, giving:
        SRISK = max(0, k*D - (1-k)*W*(1 + MES))

    Reference:
        Brownlees, C. & Engle, R. (2017). SRISK: A Conditional Capital
        Shortfall Measure of Systemic Risk. Review of Financial Studies,
        30(1), 48-79.
    """
    return max(0.0, k * liabilities - (1.0 - k) * equity * (1.0 + mes))


def compute_system_risk_metrics(
    bank_returns_dict: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Compute all risk metrics for a set of banks.

    Parameters
    ----------
    bank_returns_dict : dict
        Maps bank_id → array of returns.
    alpha : float
        Tail probability.

    Returns
    -------
    dict mapping bank_id → {var, covar, delta_covar, mes}
    """
    # System return = equal-weighted average of all bank returns
    all_returns = np.array(list(bank_returns_dict.values()))
    system_returns = np.mean(all_returns, axis=0)

    results = {}
    for bank_id, returns in bank_returns_dict.items():
        results[bank_id] = {
            "var": compute_var(returns, alpha),
            "covar": compute_covar(returns, system_returns, alpha),
            "delta_covar": compute_delta_covar(returns, system_returns, alpha),
            "mes": compute_mes(returns, system_returns, alpha),
        }
    return results
