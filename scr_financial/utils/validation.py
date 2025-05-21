"""
Validation utilities for financial network analysis.

This module provides functions for validating models and assessing
their performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch

def compute_spectral_error(original_eigenvalues: np.ndarray, cg_eigenvalues: np.ndarray, k: int) -> float:
    """
    Compute the spectral error between original and coarse-grained networks.
    
    Parameters
    ----------
    original_eigenvalues : numpy.ndarray
        Eigenvalues of the original network
    cg_eigenvalues : numpy.ndarray
        Eigenvalues of the coarse-grained network
    k : int
        Number of eigenvalues to compare
        
    Returns
    -------
    float
        Relative spectral error
    """
    # Ensure we don't exceed array lengths
    k = min(k, len(original_eigenvalues), len(cg_eigenvalues))
    
    # Compare first k eigenvalues
    error = np.linalg.norm(original_eigenvalues[:k] - cg_eigenvalues[:k]) / np.linalg.norm(original_eigenvalues[:k])
    
    return error


def validate_diffusion_dynamics(original_laplacian: np.ndarray, cg_laplacian: np.ndarray, 
                              time_steps: int = 10) -> Tuple[List[float], List[float]]:
    """
    Validate diffusion dynamics between original and coarse-grained networks.
    
    Parameters
    ----------
    original_laplacian : numpy.ndarray
        Laplacian matrix of the original network
    cg_laplacian : numpy.ndarray
        Laplacian matrix of the coarse-grained network
    time_steps : int
        Number of time steps to simulate
        
    Returns
    -------
    tuple
        Tuple containing (times, errors) where times is a list of time points
        and errors is a list of relative errors at each time point
    """
    # Initial state (uniform distribution)
    n = original_laplacian.shape[0]
    p0 = np.ones(n) / n
    
    # Time points
    times = np.linspace(0.1, 10, time_steps)
    
    # Simulate diffusion and compute errors
    errors = []
    
    for t in times:
        # Original diffusion
        pt_orig = np.exp(-t * original_laplacian) @ p0
        
        # Coarse-grained diffusion
        pt_cg = np.exp(-t * cg_laplacian) @ p0
        
        # Compute error
        error = np.linalg.norm(pt_orig - pt_cg) / np.linalg.norm(pt_orig)
        errors.append(error)
    
    return list(times), errors


def compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for prediction evaluation.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
        
    Returns
    -------
    dict
        Dictionary containing evaluation metrics
    """
    metrics = {}
    
    # Mean Squared Error
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # R-squared
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    # Avoid division by zero
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics['mape'] = mape
    else:
        metrics['mape'] = np.nan
    
    return metrics


def cross_validate_model(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5, 
                       time_series: bool = False, fit_params: Optional[Dict] = None,
                       verbose: bool = True) -> Dict[str, List[float]]:
    """
    Perform cross-validation for a model.
    
    Parameters
    ----------
    model : object
        Model with fit and predict methods
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target values
    cv : int
        Number of folds
    time_series : bool
        Whether to use time series split
    fit_params : dict, optional
        Parameters to pass to the fit method
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    dict
        Dictionary containing lists of metrics for each fold
    """
    if fit_params is None:
        fit_params = {}
    
    # Choose cross-validation strategy
    if time_series:
        kf = TimeSeriesSplit(n_splits=cv)
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Initialize metrics
    metrics = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'mape': []
    }
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train, **fit_params)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        fold_metrics = compute_prediction_metrics(y_test, y_pred)
        
        # Store metrics
        for key, value in fold_metrics.items():
            metrics[key].append(value)
        
        if verbose:
            print(f"Fold {i+1}/{cv} - RMSE: {fold_metrics['rmse']:.4f}, R²: {fold_metrics['r2']:.4f}")
    
    # Compute average metrics
    if verbose:
        print("\nAverage metrics:")
        for key, values in metrics.items():
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return metrics


def validate_vae_reconstruction(original_data: np.ndarray, reconstructed_data: np.ndarray) -> Dict[str, float]:
    """
    Validate VAE reconstruction quality.
    
    Parameters
    ----------
    original_data : numpy.ndarray
        Original data
    reconstructed_data : numpy.ndarray
        Reconstructed data
        
    Returns
    -------
    dict
        Dictionary containing validation metrics
    """
    metrics = {}
    
    # Mean Squared Error
    metrics['mse'] = mean_squared_error(original_data, reconstructed_data)
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(original_data, reconstructed_data)
    
    # Cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Compute average cosine similarity across samples
    cos_sims = []
    for i in range(len(original_data)):
        cos_sim = cosine_similarity(original_data[i], reconstructed_data[i])
        cos_sims.append(cos_sim)
    
    metrics['cosine_similarity'] = np.mean(cos_sims)
    
    return metrics


def compute_anomaly_scores(model: Any, data: np.ndarray) -> np.ndarray:
    """
    Compute anomaly scores using a VAE model.
    
    Parameters
    ----------
    model : object
        VAE model with encode and decode methods
    data : numpy.ndarray
        Data to compute anomaly scores for
        
    Returns
    -------
    numpy.ndarray
        Array of anomaly scores
    """
    # Convert to tensor if needed
    if not isinstance(data, torch.Tensor):
        data = torch.FloatTensor(data)
    
    # Get reconstructions
    model.eval()
    with torch.no_grad():
        results = model(data)
        reconstructions = results['reconstruction']
    
    # Compute reconstruction error (MSE) for each sample
    mse = torch.mean((data - reconstructions) ** 2, dim=1)
    
    return mse.numpy()
