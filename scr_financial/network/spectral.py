"""
Spectral analysis tools for financial networks.

This module provides functions for spectral analysis of financial networks,
including eigendecomposition, spectral gap identification, and diffusion modes.
"""

import logging

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional

import networkx as nx

logger = logging.getLogger(__name__)


def compute_laplacian(adjacency_matrix: np.ndarray, normalized: bool = True) -> np.ndarray:
    """
    Compute the graph Laplacian from an adjacency matrix.

    Args:
        adjacency_matrix: Adjacency matrix of the graph.
        normalized: Whether to compute the normalized Laplacian. Defaults to
            True.

    Returns:
        Graph Laplacian matrix as an ndarray.
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
            L = np.asarray(nx.normalized_laplacian_matrix(G).todense())
        else:
            L = np.asarray(nx.laplacian_matrix(G).todense())

    return L


def eigendecomposition(laplacian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform eigendecomposition of the Laplacian matrix.

    Args:
        laplacian: Graph Laplacian matrix.

    Returns:
        Tuple of (eigenvalues, eigenvectors).
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


def find_spectral_gap(
    eigenvalues: np.ndarray,
    min_index: int = 1,
    max_index: Optional[int] = None,
) -> Tuple[int, float]:
    """
    Identify the spectral gap in the eigenvalue spectrum.

    Args:
        eigenvalues: Array of eigenvalues.
        min_index: Minimum index to consider. Defaults to 1.
        max_index: Maximum index to consider. Defaults to None (half of
            spectrum).

    Returns:
        Tuple of (gap index, gap size).
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


def compute_diffusion_modes(
    laplacian: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the first k diffusion modes of the network.

    Args:
        laplacian: Graph Laplacian matrix.
        k: Number of modes to compute.

    Returns:
        Tuple of (eigenvalues, eigenvectors) for the first k modes.
    """
    eigenvalues, eigenvectors = eigendecomposition(laplacian)

    # Return the first k+1 modes (including the constant mode)
    return eigenvalues[: k + 1], eigenvectors[:, : k + 1]


def analyze_spectral_properties(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> dict:
    """
    Analyze spectral properties of the network.

    Args:
        eigenvalues: Array of eigenvalues.
        eigenvectors: Matrix of eigenvectors.

    Returns:
        Dictionary containing spectral properties.

    Raises:
        ValueError: If eigenvalues has fewer than 2 elements.
    """
    if len(eigenvalues) < 2:
        raise ValueError("Laplacian must have at least 2 eigenvalues")

    properties = {}

    # Spectral gap
    gap_idx, gap_size = find_spectral_gap(eigenvalues)
    properties["spectral_gap_index"] = gap_idx
    properties["spectral_gap_size"] = gap_size

    # Algebraic connectivity (second smallest eigenvalue)
    properties["algebraic_connectivity"] = eigenvalues[1]

    # Spectral radius (largest eigenvalue)
    properties["spectral_radius"] = eigenvalues[-1]

    # Participation ratio of eigenvectors
    participation_ratios = []
    for i in range(eigenvectors.shape[1]):
        v = eigenvectors[:, i]
        pr = np.sum(v**2) ** 2 / np.sum(v**4)
        participation_ratios.append(pr)

    properties["participation_ratios"] = np.array(participation_ratios)

    # Localization of eigenvectors
    properties["localization"] = 1.0 / properties["participation_ratios"]

    return properties


def compute_spectral_embedding(laplacian: np.ndarray, dim: int = 2) -> np.ndarray:
    """
    Compute spectral embedding of the network.

    Args:
        laplacian: Graph Laplacian matrix.
        dim: Embedding dimension. Defaults to 2.

    Returns:
        Spectral embedding matrix of shape (n_nodes, dim).
    """
    eigenvalues, eigenvectors = eigendecomposition(laplacian)

    # Use the first non-trivial eigenvectors for embedding
    # Skip the first eigenvector for connected graphs (constant)
    embedding = eigenvectors[:, 1 : dim + 1]

    return embedding


def compute_diffusion_distance(
    laplacian: np.ndarray, t: float = 1.0, k: Optional[int] = None
) -> np.ndarray:
    """
    Compute diffusion distance matrix at time t.

    Uses a fully vectorised NumPy/SciPy implementation. Memory complexity is
    O(n²·k) due to the pairwise squared-distance computation.

    Args:
        laplacian: Graph Laplacian matrix.
        t: Diffusion time. Defaults to 1.0.
        k: Number of eigenmodes to use. Defaults to None (use all).

    Returns:
        Symmetric diffusion distance matrix of shape (n_nodes, n_nodes) with
        zeros on the diagonal.
    """
    eigenvalues, eigenvectors = eigendecomposition(laplacian)

    # Determine number of eigenmodes to use
    if k is None:
        k = len(eigenvalues)
    else:
        k = min(k, len(eigenvalues))

    # Step 1: decay weights for modes 1..k-1 (skip constant mode 0)
    # shape: (k-1,)
    weights = np.exp(-2 * t * eigenvalues[1:k])

    # Step 2: weighted eigenvectors — shape: (n_nodes, k-1)
    phi = eigenvectors[:, 1:k] * np.sqrt(weights)

    # Step 3: pairwise squared Euclidean distances then sqrt
    # cdist returns shape (n_nodes, n_nodes); already symmetric
    distance_matrix = cdist(phi, phi, metric="sqeuclidean")
    np.sqrt(distance_matrix, out=distance_matrix)

    # Step 4: ensure exact symmetry (numerical safety)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

    # Step 5: zero diagonal
    np.fill_diagonal(distance_matrix, 0.0)

    return distance_matrix
