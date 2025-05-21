"""
Utility functions for data preprocessing and manipulation.

This module provides helper functions for data preprocessing, normalization,
and manipulation in the context of financial network analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime


def normalize_matrix(matrix: np.ndarray, method: str = 'degree') -> np.ndarray:
    """
    Normalize a matrix using various methods.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to normalize
    method : str, optional
        Normalization method, by default 'degree'
        
    Returns
    -------
    np.ndarray
        Normalized matrix
    """
    if method == 'degree':
        # Normalize by row and column sums
        row_sums = np.sum(matrix, axis=1)
        col_sums = np.sum(matrix, axis=0)
        
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        col_sums[col_sums == 0] = 1
        
        # Normalize
        D_row_inv_sqrt = np.diag(1.0 / np.sqrt(row_sums))
        D_col_inv_sqrt = np.diag(1.0 / np.sqrt(col_sums))
        
        normalized = D_row_inv_sqrt @ matrix @ D_col_inv_sqrt
        
    elif method == 'standardize':
        # Standardize to mean 0, std 1
        mean = np.mean(matrix)
        std = np.std(matrix)
        if std == 0:
            std = 1
        normalized = (matrix - mean) / std
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def filter_matrix(matrix: np.ndarray, method: str = 'threshold', threshold: float = 0.05) -> np.ndarray:
    """
    Filter a matrix to reduce noise.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to filter
    method : str, optional
        Filtering method, by default 'threshold'
    threshold : float, optional
        Threshold value for filtering, by default 0.05
        
    Returns
    -------
    np.ndarray
        Filtered matrix
    """
    if method == 'threshold':
        # Set values below threshold to zero
        filtered = matrix.copy()
        filtered[np.abs(filtered) < threshold] = 0
        
    else:
        raise ValueError(f"Unknown filtering method: {method}")
    
    return filtered


def align_time_series(series_dict: Dict[str, pd.DataFrame], freq: str = 'D') -> Dict[str, pd.DataFrame]:
    """
    Align multiple time series to a common frequency.
    
    Parameters
    ----------
    series_dict : dict
        Dictionary of time series DataFrames
    freq : str, optional
        Target frequency, by default 'D' (daily)
        
    Returns
    -------
    dict
        Dictionary of aligned time series DataFrames
    """
    # Find common date range
    start_dates = []
    end_dates = []
    
    for name, df in series_dict.items():
        if isinstance(df.index, pd.DatetimeIndex):
            start_dates.append(df.index.min())
            end_dates.append(df.index.max())
        elif 'date' in df.columns:
            start_dates.append(pd.to_datetime(df['date']).min())
            end_dates.append(pd.to_datetime(df['date']).max())
    
    if not start_dates or not end_dates:
        return series_dict
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    # Create common date range
    common_dates = pd.date_range(start=common_start, end=common_end, freq=freq)
    
    # Align each series
    aligned_dict = {}
    
    for name, df in series_dict.items():
        if isinstance(df.index, pd.DatetimeIndex):
            # Reindex to common dates
            aligned = df.reindex(common_dates, method='ffill')
            aligned_dict[name] = aligned
        elif 'date' in df.columns:
            # Set date as index
            df_copy = df.copy()
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy = df_copy.set_index('date')
            
            # Reindex to common dates
            aligned = df_copy.reindex(common_dates, method='ffill')
            
            # Reset index
            aligned = aligned.reset_index()
            aligned_dict[name] = aligned
        else:
            # If no date information, keep as is
            aligned_dict[name] = df
    
    return aligned_dict


def compute_rolling_correlation(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute rolling correlation matrix from returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of asset returns
    window : int, optional
        Rolling window size, by default 60
        
    Returns
    -------
    pd.DataFrame
        DataFrame of rolling correlation matrices
    """
    # Initialize empty DataFrame to store correlation matrices
    corr_matrices = {}
    
    # Compute rolling correlation
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i - window:i]
        corr_matrix = window_returns.corr()
        corr_matrices[returns.index[i - 1]] = corr_matrix
    
    return pd.Series(corr_matrices)


def compute_distance_matrix(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distance matrix from correlation matrix.
    
    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix
        
    Returns
    -------
    pd.DataFrame
        Distance matrix
    """
    # Convert correlation to distance: d_ij = sqrt(2 * (1 - rho_ij))
    distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
    
    return pd.DataFrame(distance_matrix, index=correlation_matrix.index, columns=correlation_matrix.columns)


def compute_minimum_spanning_tree(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute minimum spanning tree from distance matrix.
    
    Parameters
    ----------
    distance_matrix : pd.DataFrame
        Distance matrix
        
    Returns
    -------
    pd.DataFrame
        Adjacency matrix of minimum spanning tree
    """
    import networkx as nx
    
    # Create graph from distance matrix
    G = nx.Graph()
    
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            G.add_edge(
                distance_matrix.index[i],
                distance_matrix.index[j],
                weight=distance_matrix.iloc[i, j]
            )
    
    # Compute minimum spanning tree
    mst = nx.minimum_spanning_tree(G)
    
    # Convert to adjacency matrix
    adj_matrix = nx.to_pandas_adjacency(mst)
    
    return adj_matrix
