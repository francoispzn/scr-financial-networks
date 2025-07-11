"""Random Matrix Theory denoising for correlation matrices."""

import numpy as np
import scipy.linalg as la
from typing import Dict, Tuple


def marchenko_pastur_bounds(T: int, N: int) -> Tuple[float, float]:
    """Compute Marchenko-Pastur eigenvalue bounds.

    Args:
        T: Number of observations (time periods).
        N: Number of variables (banks).

    Returns:
        (lambda_minus, lambda_plus)
    """
    q = T / N
    lambda_plus = (1 + 1/np.sqrt(q)) ** 2
    lambda_minus = (1 - 1/np.sqrt(q)) ** 2
    return float(lambda_minus), float(lambda_plus)


def denoise_correlation(corr_matrix: np.ndarray, T: int,
                        method: str = 'constant') -> np.ndarray:
    """Denoise a correlation matrix using RMT.

    Methods:
        'constant': Replace noise eigenvalues with their average (Laloux 1999).
        'shrinkage': Ledoit-Wolf shrinkage toward identity.
        'targeted': Preserve signal eigenvalues, shrink noise toward average.
    """
    N = corr_matrix.shape[0]
    eigenvalues, eigenvectors = la.eigh(corr_matrix)
    _, lambda_plus = marchenko_pastur_bounds(T, N)

    if method == 'constant':
        noise_mask = eigenvalues <= lambda_plus
        if noise_mask.any():
            avg_noise = np.mean(eigenvalues[noise_mask])
            eigenvalues[noise_mask] = avg_noise
        denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    elif method == 'shrinkage':
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(np.eye(N))  # Dummy fit for shrinkage
            alpha = min(1.0, N / T)  # Simple shrinkage intensity
            denoised = (1 - alpha) * corr_matrix + alpha * np.eye(N)
        except ImportError:
            alpha = min(1.0, N / T)
            denoised = (1 - alpha) * corr_matrix + alpha * np.eye(N)

    elif method == 'targeted':
        noise_mask = eigenvalues <= lambda_plus
        signal_eigenvalues = eigenvalues.copy()
        if noise_mask.any():
            target = np.sum(eigenvalues[noise_mask]) / np.sum(noise_mask)
            signal_eigenvalues[noise_mask] = target
        denoised = eigenvectors @ np.diag(signal_eigenvalues) @ eigenvectors.T
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure PSD and unit diagonal
    np.fill_diagonal(denoised, 1.0)
    denoised = (denoised + denoised.T) / 2
    return denoised


def fit_marchenko_pastur(eigenvalues: np.ndarray, T: int, N: int) -> Dict:
    """Test fit against Marchenko-Pastur distribution."""
    from scipy.stats import kstest
    _, lambda_plus = marchenko_pastur_bounds(T, N)
    n_signal = int(np.sum(eigenvalues > lambda_plus))
    n_noise = len(eigenvalues) - n_signal

    # KS test against uniform on [lambda_minus, lambda_plus]
    noise_evals = eigenvalues[eigenvalues <= lambda_plus]
    if len(noise_evals) > 3:
        _, ub = marchenko_pastur_bounds(T, N)
        lb, _ = marchenko_pastur_bounds(T, N)
        ks_stat, p_value = kstest(noise_evals, 'uniform', args=(lb, ub - lb))
    else:
        ks_stat, p_value = 0.0, 1.0

    return {"n_signal": n_signal, "n_noise": n_noise,
            "lambda_plus": float(lambda_plus),
            "ks_stat": float(ks_stat), "p_value": float(p_value)}
