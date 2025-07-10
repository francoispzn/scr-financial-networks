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
    adjacency_matrix: Optional[np.ndarray] = None,
) -> Tuple[int, float]:
    """
    Identify the spectral gap in the eigenvalue spectrum using the
    Erdos-Renyi null model test from Schmidt, Caccioli & Aste (2025).

    For each gap index i, we compare the observed gap against a
    distribution of gaps from an ensemble of Erdos-Renyi random graphs
    with the same number of vertices and expected average degree. A gap
    is flagged as a candidate if its empirical one-tailed p-value is
    below alpha=0.01.

    If no statistically significant gap is found, falls back to the
    largest gap in [min_index, max_index].

    Args:
        eigenvalues: Array of eigenvalues (sorted ascending).
        min_index: Minimum gap index to consider. Defaults to 1.
        max_index: Maximum gap index to consider. Defaults to None (half
            of spectrum).
        adjacency_matrix: Optional adjacency matrix for computing exact
            average degree for the ER null model. If None, the degree is
            estimated from the eigenvalue spectrum.

    Returns:
        Tuple of (gap index, gap size).

    References:
        Schmidt, Caccioli & Aste, "Spectral coarse graining and rescaling
        for preserving structural and dynamical properties in graphs",
        Phys. Rev. E 112, 034303 (2025).
    """
    if max_index is None:
        max_index = len(eigenvalues) // 2

    gaps = np.diff(eigenvalues)

    if len(gaps) < 2:
        return min_index if min_index < len(gaps) else 0, float(gaps[0]) if len(gaps) > 0 else 0.0

    # Detect whether eigenvalues are from normalized or combinatorial Laplacian.
    # Normalized Laplacian has eigenvalues in [0, 2]; combinatorial can be much larger.
    # Schmidt uses combinatorial (D - A), so default to normalized=False unless
    # eigenvalues are clearly in normalized range.
    is_normalized = bool(len(eigenvalues) > 1 and eigenvalues[-1] <= 2.01)

    # ── Schmidt null model: Erdos-Renyi ensemble ─────────────────────
    n = len(eigenvalues)
    candidate = _find_gap_via_er_null_model(
        eigenvalues, n, min_index, max_index,
        adjacency_matrix=adjacency_matrix,
        normalized=is_normalized,
    )
    if candidate is not None:
        return candidate

    # ── Fallback: largest gap in search range ────────────────────────
    search_range = gaps[min_index:max_index]
    if len(search_range) == 0:
        return min_index, 0.0

    max_gap_idx = np.argmax(search_range) + min_index
    return max_gap_idx, float(gaps[max_gap_idx])


def _find_gap_via_er_null_model(
    eigenvalues: np.ndarray,
    n: int,
    min_index: int,
    max_index: int,
    n_ensemble: int = 1000,
    alpha: float = 0.01,
    seed: int = 42,
    adjacency_matrix: Optional[np.ndarray] = None,
    normalized: bool = False,
) -> Optional[Tuple[int, float]]:
    """Erdos-Renyi null model gap test (Schmidt et al. 2025, Section II).

    Generates an ensemble of connected ER graphs with same n and expected
    average degree, computes their Laplacian spectra, and flags gaps whose
    empirical p-value < alpha.

    Args:
        eigenvalues: Observed Laplacian eigenvalues.
        n: Number of vertices.
        min_index: Minimum gap index to test.
        max_index: Maximum gap index to test.
        n_ensemble: Number of ER realisations (Schmidt uses 10^4; we use
            10^3 for speed, adjustable).
        alpha: Significance threshold.
        seed: Random seed for reproducibility.
        adjacency_matrix: If provided, the exact average degree is computed
            from this matrix (p = k_avg / (n-1)). Otherwise estimated
            from the eigenvalue spectrum.
        normalized: If True, use normalized Laplacian (I - D^{-1/2}AD^{-1/2})
            for the ER ensemble. If False (default), use combinatorial
            Laplacian (D - A) matching Schmidt et al.

    Returns:
        (gap_index, gap_size) for the first significant gap, or None.
    """
    rng = np.random.default_rng(seed)
    gaps_obs = np.diff(eigenvalues)

    # Compute ER connection probability from adjacency if available
    if adjacency_matrix is not None:
        adj_bin = (adjacency_matrix > 0).astype(float)
        np.fill_diagonal(adj_bin, 0)
        k_avg = float(adj_bin.sum() / n) if n > 0 else 1.0
        p_est = min(0.95, max(0.05, k_avg / max(n - 1, 1)))
        logger.debug("ER null model: using exact density p=%.3f (k_avg=%.1f)", p_est, k_avg)
    else:
        # Heuristic from eigenvalue spectrum
        ev_var = float(np.var(eigenvalues[1:]))
        p_est = min(0.9, max(0.1, 1.0 / (1.0 + ev_var * n)))

    # Collect gap distributions from ER ensemble
    gap_distributions: List[List[float]] = [[] for _ in range(len(gaps_obs))]

    for _ in range(n_ensemble):
        # Generate connected ER graph
        for attempt in range(5):
            adj_er = (rng.random((n, n)) < p_est).astype(float)
            np.fill_diagonal(adj_er, 0)
            adj_er = np.maximum(adj_er, adj_er.T)  # Symmetric
            degrees = adj_er.sum(axis=1)
            if np.all(degrees > 0):
                break
        else:
            continue

        # Compute Laplacian eigenvalues (must match the type used for observed spectrum)
        if normalized:
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
            I = np.eye(n)
            L_er = I - D_inv_sqrt @ adj_er @ D_inv_sqrt
        else:
            # Combinatorial Laplacian (Schmidt et al.): L = D - A
            L_er = np.diag(degrees) - adj_er
        ev_er = la.eigh(L_er, eigvals_only=True)
        gaps_er = np.diff(ev_er)

        for i in range(min(len(gaps_er), len(gap_distributions))):
            gap_distributions[i].append(gaps_er[i])

    # Test each gap for significance
    candidates = []
    for i in range(min_index, min(max_index, len(gaps_obs))):
        if len(gap_distributions[i]) < 50:
            continue
        er_gaps = np.array(gap_distributions[i])
        # One-tailed p-value: fraction of ER gaps >= observed gap
        p_value = float(np.mean(er_gaps >= gaps_obs[i]))
        if p_value < alpha:
            candidates.append((i, float(gaps_obs[i]), p_value))

    if not candidates:
        return None

    # Return the first (smallest index) significant gap
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    logger.info(
        "Schmidt gap test: found significant gap at index %d (size=%.4f, p=%.4f)",
        best[0], best[1], best[2],
    )
    return best[0], best[1]


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
