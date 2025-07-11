"""
Uncertainty quantification for GNN spectral predictions.

Provides MC Dropout, quantile regression, and conformal prediction
to produce prediction intervals alongside point estimates.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def mc_dropout_predict(model, graph_sequences, n_forward: int = 50,
                       target_mean=None, target_std=None):
    """Monte Carlo Dropout prediction intervals.

    Keep dropout active at inference time, run N forward passes,
    compute mean and variance of predictions.

    Args:
        model: TemporalGNN model (must have dropout layers).
        graph_sequences: List of graph sequences (same format as training).
        n_forward: Number of stochastic forward passes.
        target_mean, target_std: For denormalization.

    Returns:
        dict with 'mean', 'std', 'ci_lower', 'ci_upper' (95% CI), all shape (n_samples, n_targets)
    """
    model.train()  # Keep dropout active

    all_preds = []
    with torch.no_grad():
        for _ in range(n_forward):
            pred = model(graph_sequences).numpy()
            if target_std is not None and target_mean is not None:
                pred = pred * target_std + target_mean
            all_preds.append(pred)

    all_preds = np.stack(all_preds, axis=0)  # (n_forward, n_samples, n_targets)

    mean = np.mean(all_preds, axis=0)
    std = np.std(all_preds, axis=0)
    ci_lower = np.percentile(all_preds, 2.5, axis=0)
    ci_upper = np.percentile(all_preds, 97.5, axis=0)

    model.eval()

    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_forward": n_forward,
    }


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantile: float):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        error = target - pred
        return torch.mean(torch.max(self.quantile * error, (self.quantile - 1) * error))


def train_quantile_predictor(predictor_class, predictor_kwargs, snapshots,
                              quantiles=(0.05, 0.25, 0.50, 0.75, 0.95),
                              epochs=200, lr=3e-3, patience=30):
    """Train separate models for each quantile.

    Returns dict mapping quantile -> trained predictor.
    """
    results = {}
    for q in quantiles:
        logger.info("Training quantile %.2f predictor", q)
        predictor = predictor_class(**predictor_kwargs)
        # Override loss function for quantile regression
        predictor._quantile = q
        predictor.train(snapshots, epochs=epochs, lr=lr, patience=patience)
        results[q] = predictor
    return results


def conformal_prediction(y_cal_true, y_cal_pred, y_test_pred, alpha=0.05):
    """Split conformal prediction for guaranteed coverage.

    Uses calibration set residuals to construct prediction intervals
    with 1-alpha coverage guarantee (distribution-free).

    Args:
        y_cal_true: Calibration set true values (n_cal, n_targets).
        y_cal_pred: Calibration set predictions (n_cal, n_targets).
        y_test_pred: Test set predictions (n_test, n_targets).
        alpha: Miscoverage rate (default 5% → 95% intervals).

    Returns:
        dict with 'lower', 'upper', 'width', 'coverage_guarantee'
    """
    residuals = np.abs(y_cal_true - y_cal_pred)  # (n_cal, n_targets)

    # Quantile of calibration residuals
    n_cal = len(residuals)
    q_level = np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
    q_level = min(q_level, 1.0)

    # Per-target conformal quantile
    q_hat = np.quantile(residuals, q_level, axis=0)  # (n_targets,)

    lower = y_test_pred - q_hat
    upper = y_test_pred + q_hat
    width = upper - lower

    return {
        "lower": lower,
        "upper": upper,
        "width": width,
        "q_hat": q_hat,
        "coverage_guarantee": 1 - alpha,
        "n_calibration": n_cal,
    }


def calibration_metrics(y_true, lower, upper):
    """Compute prediction interval calibration metrics.

    Args:
        y_true: True values (n, d).
        lower: Lower bounds (n, d).
        upper: Upper bounds (n, d).

    Returns:
        dict with PICP (coverage), MPIW (width), CWC (coverage width criterion)
    """
    covered = (y_true >= lower) & (y_true <= upper)
    picp = np.mean(covered)  # Prediction Interval Coverage Probability
    mpiw = np.mean(upper - lower)  # Mean Prediction Interval Width

    # Coverage-Width Criterion (penalizes undercoverage)
    target_coverage = 0.95
    eta = 50  # Penalty strength
    if picp < target_coverage:
        cwc = mpiw * np.exp(-eta * (picp - target_coverage))
    else:
        cwc = mpiw

    # Per-target coverage
    if y_true.ndim == 2:
        per_target_picp = {f"target_{i}": float(np.mean(covered[:, i]))
                           for i in range(y_true.shape[1])}
    else:
        per_target_picp = {}

    return {
        "picp": float(picp),
        "mpiw": float(mpiw),
        "cwc": float(cwc),
        "per_target_picp": per_target_picp,
    }
