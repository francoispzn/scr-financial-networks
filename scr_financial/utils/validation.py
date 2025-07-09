"""
Validation utilities for financial network analysis.

This module provides functions for validating models and assessing
their performance.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit

logger = logging.getLogger(__name__)


def compute_spectral_error(
    original_eigenvalues: np.ndarray,
    cg_eigenvalues: np.ndarray,
    k: int,
) -> float:
    """Compute the spectral error between original and coarse-grained networks.

    Args:
        original_eigenvalues: Eigenvalues of the original network.
        cg_eigenvalues: Eigenvalues of the coarse-grained network.
        k: Number of eigenvalues to compare.

    Returns:
        Relative spectral error.
    """
    k = min(k, len(original_eigenvalues), len(cg_eigenvalues))
    error = np.linalg.norm(
        original_eigenvalues[:k] - cg_eigenvalues[:k]
    ) / np.linalg.norm(original_eigenvalues[:k])
    return float(error)


def validate_diffusion_dynamics(
    original_laplacian: np.ndarray,
    cg_laplacian: np.ndarray,
    time_steps: int = 10,
) -> Tuple[List[float], List[float]]:
    """Validate diffusion dynamics between original and coarse-grained networks.

    Args:
        original_laplacian: Laplacian matrix of the original network.
        cg_laplacian: Laplacian matrix of the coarse-grained network.
        time_steps: Number of time steps to simulate.

    Returns:
        Tuple containing (times, errors) where *times* is a list of time
        points and *errors* is a list of relative errors at each time point.
    """
    n = original_laplacian.shape[0]
    p0 = np.ones(n) / n

    times = np.linspace(0.1, 10, time_steps)
    errors: List[float] = []

    for t in times:
        pt_orig = np.exp(-t * original_laplacian) @ p0
        pt_cg = np.exp(-t * cg_laplacian) @ p0
        error = np.linalg.norm(pt_orig - pt_cg) / np.linalg.norm(pt_orig)
        errors.append(float(error))

    return list(times), errors


def compute_prediction_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for prediction evaluation.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary containing evaluation metrics (mse, rmse, mae, r2, mape).
    """
    metrics: Dict[str, float] = {}

    metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))

    mask = y_true != 0
    if np.any(mask):
        mape = (
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
            * 100
        )
        metrics["mape"] = float(mape)
    else:
        metrics["mape"] = float("nan")

    return metrics


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    time_series: bool = False,
    fit_params: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Perform cross-validation for a model.

    Args:
        model: Model with ``fit`` and ``predict`` methods.
        X: Feature matrix.
        y: Target values.
        cv: Number of folds.
        time_series: Whether to use time series split.
        fit_params: Parameters to pass to the fit method.
        verbose: Whether to log progress.

    Returns:
        Dictionary containing lists of metrics for each fold.
    """
    if fit_params is None:
        fit_params = {}

    kf: Union[TimeSeriesSplit, KFold] = (
        TimeSeriesSplit(n_splits=cv)
        if time_series
        else KFold(n_splits=cv, shuffle=True, random_state=42)
    )

    metrics: Dict[str, List[float]] = {
        "mse": [],
        "rmse": [],
        "mae": [],
        "r2": [],
        "mape": [],
    }

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train, **fit_params)
        y_pred = model.predict(X_test)

        fold_metrics = compute_prediction_metrics(y_test, y_pred)

        for key, value in fold_metrics.items():
            metrics[key].append(value)

        if verbose:
            logger.info(
                "Fold %d/%d - RMSE: %.4f, R²: %.4f",
                i + 1,
                cv,
                fold_metrics["rmse"],
                fold_metrics["r2"],
            )

    if verbose:
        logger.info("Average metrics:")
        for key, values in metrics.items():
            logger.info(
                "  %s: %.4f ± %.4f", key, np.mean(values), np.std(values)
            )

    return metrics


def validate_vae_reconstruction(
    original_data: np.ndarray, reconstructed_data: np.ndarray
) -> Dict[str, float]:
    """Validate VAE reconstruction quality.

    Args:
        original_data: Original data.
        reconstructed_data: Reconstructed data.

    Returns:
        Dictionary containing validation metrics.
    """
    metrics: Dict[str, float] = {}

    metrics["mse"] = float(mean_squared_error(original_data, reconstructed_data))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    metrics["mae"] = float(mean_absolute_error(original_data, reconstructed_data))

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    cos_sims = [
        cosine_similarity(original_data[i], reconstructed_data[i])
        for i in range(len(original_data))
    ]
    metrics["cosine_similarity"] = float(np.mean(cos_sims))

    return metrics


def compute_anomaly_scores(model: Any, data: np.ndarray) -> np.ndarray:
    """Compute anomaly scores using a VAE model.

    Args:
        model: VAE model with ``encode`` and ``decode`` methods.
        data: Data to compute anomaly scores for.

    Returns:
        Array of anomaly scores.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.FloatTensor(data)

    model.eval()
    with torch.no_grad():
        results = model(data)
        reconstructions = results["reconstruction"]

    mse = torch.mean((data - reconstructions) ** 2, dim=1)
    return mse.numpy()
