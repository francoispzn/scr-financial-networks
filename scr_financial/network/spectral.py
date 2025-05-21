"""
Spectral analysis tools for financial networks.

This module provides functions for spectral analysis of financial networks,
including eigendecomposition, spectral gap identification, and diffusion modes.
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from typing import Tuple, List, Optional

import networkx as nx


def compute_laplacian(adjacency_matrix: np.ndarray, normalized: bool = True) -> np.ndarray:
    """
    Compute the graph Laplacian from an adjacency matrix.
    
    Parameters
    ----------
    adjacency_matrix : numpy.ndarray
        Adjacency matrix of the graph
    normalized : bool, optional
        Whether to compute the normalized Laplacian, by default True
        
    Returns
    -------
    numpy.ndarray
        Graph Laplacian matrix
    """
    # Ensure the adjacency matrix is symmetric for undirected graphs
    if np.allclose(adjacency_matrix, adjacency_matrix.T):
        # Undirected graph
        if normalized:
            # Compute degree matrix
            degrees = np.sum(adjacency_matrix, axis=1)
            # Avoid division by zero
            degrees[degrees == 0] = 1
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            # Normalized Laplacian: I - D^(-1/2) A D^(-1/2)
            I = np.eye(adjacency_matrix.shape[0])
            L = I - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
        else:
            # Compute degree matrix
            D = np.diag(np.sum(adjacency_matrix, axis=1))
            # Unnormalized Laplacian: D - A
            L = D - adjacency_matrix
    else:
        # Directed graph - use networkx for proper handling
        G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
        if normalized:
            L = nx.normalized_laplacian_matrix(G).todense()
        else:
            L = nx.laplacian_matrix(G).todense()
    
    return L


def eigendecomposition(laplacian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform eigendecomposition of the Laplacian matrix.
    
    Parameters
    ----------
    laplacian : numpy.ndarray
        Graph Laplacian matrix
        
    Returns
    -------
    tuple
        Tuple containing eigenvalues and eigenvectors
    """
    # For symmetric matrices, use eigh which is faster and more accurate
    if np.allclose(laplacian, laplacian.T):
        eigenvalues, eigenvectors = la.eigh(laplacian)
    else:
        eigenvalues, eigenvectors = la.eig(laplacian)
        # Sort by eigenvalues
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
    
    return eigenvalues, eigenvectors


def find_spectral_gap(eigenvalues: np.ndarray, min_index: int = 1, max_index: Optional[int] = None) -> Tuple[int, float]:
    """
    Identify the spectral gap in the eigenvalue spectrum.
    
    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Array of eigenvalues
    min_index : int, optional
        Minimum index to consider, by default 1
    max_index : int, optional
        Maximum index to consider, by default None
        
    Returns
    -------
    tuple
        Tuple containing the index of the gap and the gap size
    """
    if max_index is None:
        max_index = len(eigenvalues) // 2
    
    # Compute differences between consecutive eigenvalues
    gaps = np.diff(eigenvalues)
    
    # Find the largest gap in the specified range
    search_range = gaps[min_index:max_index]
    if len(search_range) == 0:
        return min_index, 0
    
    max_gap_idx = np.argmax(search_range) + min_index
    max_gap = gaps[max_gap_idx]
    
    return max_gap_idx, max_gap


def compute_diffusion_modes(laplacian: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the first k diffusion modes of the network.
    
    Parameters
    ----------
    laplacian : numpy.ndarray
        Graph Laplacian matrix
    k : int
        Number of modes to compute
        
    Returns
    -------
    tuple
        Tuple containing eigenvalues and eigenvectors for the first k modes
    """
    eigenvalues, eigenvectors = eigendecomposition(laplacian)
    
    # Return the first k+1 modes (including the constant mode)
    return eigenvalues[:k+1], eigenvectors[:, :k+1]


def analyze_spectral_properties(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> dict:
    """
    Analyze spectral properties of the network.
    
    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Array of eigenvalues
    eigenvectors : numpy.ndarray
        Matrix of eigenvectors
        
    Returns
    -------
    dict
        Dictionary containing spectral properties
    """
    properties = {}
    
    # Spectral gap
    gap_idx, gap_size = find_spectral_gap(eigenvalues)
    properties['spectral_gap_index'] = gap_idx
    properties['spectral_gap_size'] = gap_size
    
    # Algebraic connectivity (second smallest eigenvalue)
    properties['algebraic_connectivity'] = eigenvalues[1]
    
    # Spectral radius (largest eigenvalue)
    properties['spectral_radius'] = eigenvalues[-1]
    
    # Participation ratio of eigenvectors
    participation_ratios = []
    for i in range(eigenvectors.shape[1]):
        v = eigenvectors[:, i]
        pr = np.sum(v**2)**2 / np.sum(v**4)
        participation_ratios.append(pr)
    
    properties['participation_ratios'] = np.array(participation_ratios)
    
    # Localization of eigenvectors
    properties['localization'] = 1.0 / properties['participation_ratios']
    
    return properties


def compute_spectral_embedding(laplacian: np.ndarray, dim: int = 2) -> np.ndarray:
    """
    Compute spectral embedding of the network.
    
    Parameters
    ----------
    laplacian : numpy.ndarray
        Graph Laplacian matrix
    dim : int, optional
        Embedding dimension, by default 2
        
    Returns
    -------
    numpy.ndarray
        Spectral embedding matrix of shape (n_nodes, dim)
    """
    eigenvalues, eigenvectors = eigendecomposition(laplacian)
    
    # Use the first non-trivial eigenvectors for embedding
    # Skip the first eigenvector for connected graphs (constant)
    embedding = eigenvectors[:, 1:dim+1]
    
    return embedding


def compute_diffusion_distance(laplacian: np.ndarray, t: float = 1.0, k: Optional[int] = None) -> np.ndarray:
    """
    Compute diffusion distance matrix at time t.
    
    Parameters
    ----------
    laplacian : numpy.ndarray
        Graph Laplacian matrix
    t : float, optional
        Diffusion time, by default 1.0
    k : int, optional
        Number of eigenmodes to use, by default None (use all)
        
    Returns
    -------
    numpy.ndarray
        Diffusion distance matrix
    """
    eigenvalues, eigenvectors = eigendecomposition(laplacian)
    
    # Determine number of eigenmodes to use
    if k is None:
        k = len(eigenvalues)
    else:
        k = min(k, len(eigenvalues))
    
    # Compute diffusion distance
    n_nodes = laplacian.shape[0]
    distance_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # Sum over eigenmodes
            distance_sq = 0
            for l in range(1, k):  # Skip the first eigenmode (constant)
                term = np.exp(-2 * t * eigenvalues[l]) * (eigenvectors[i, l] - eigenvectors[j, l])**2
                distance_sq += term
            
            distance = np.sqrt(distance_sq)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix
