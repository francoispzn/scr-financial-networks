"""
Tests for the spectral analysis module.

Uses pure pytest style (no unittest.TestCase).
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
import networkx as nx

from scr_financial.network.spectral import (
    compute_laplacian,
    eigendecomposition,
    find_spectral_gap,
    compute_diffusion_modes,
    analyze_spectral_properties,
    compute_spectral_embedding,
    compute_diffusion_distance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def symmetric_adj():
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
    ], dtype=float)
    return A


@pytest.fixture
def directed_adj():
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 0, 0],
    ], dtype=float)
    return A


# ---------------------------------------------------------------------------
# compute_laplacian tests
# ---------------------------------------------------------------------------

def test_laplacian_shape_matches_input(symmetric_adj):
    L = compute_laplacian(symmetric_adj)
    assert L.shape == (5, 5)


def test_laplacian_normalized_eigenvalues_nonneg(symmetric_adj):
    L = compute_laplacian(symmetric_adj, normalized=True)
    evals, _ = np.linalg.eig(L)
    assert np.all(evals.real >= -1e-10)


def test_laplacian_unnormalized_row_sums_zero(symmetric_adj):
    L = compute_laplacian(symmetric_adj, normalized=False)
    row_sums = L.sum(axis=1)
    assert_allclose(row_sums, 0, atol=1e-10)


def test_laplacian_returns_ndarray_not_matrix(symmetric_adj):
    L = compute_laplacian(symmetric_adj)
    assert type(L) is np.ndarray


def test_laplacian_directed_returns_ndarray(directed_adj):
    L = compute_laplacian(directed_adj)
    assert type(L) is np.ndarray


# ---------------------------------------------------------------------------
# eigendecomposition tests
# ---------------------------------------------------------------------------

def test_eigendecomposition_count(symmetric_adj):
    L = compute_laplacian(symmetric_adj)
    evals, evecs = eigendecomposition(L)
    n = symmetric_adj.shape[0]
    assert len(evals) == n
    assert evecs.shape == (n, n)


def test_eigendecomposition_sorted_ascending(symmetric_adj):
    L = compute_laplacian(symmetric_adj)
    evals, _ = eigendecomposition(L)
    assert np.all(np.diff(evals) >= -1e-10)


def test_eigendecomposition_eigenvectors_orthonormal(symmetric_adj):
    L = compute_laplacian(symmetric_adj, normalized=True)
    _, evecs = eigendecomposition(L)
    n = symmetric_adj.shape[0]
    assert_allclose(evecs.T @ evecs, np.eye(n), atol=1e-10)


# ---------------------------------------------------------------------------
# find_spectral_gap tests
# ---------------------------------------------------------------------------

def test_find_spectral_gap_known_spectrum():
    evals = np.array([0.0, 0.1, 0.2, 0.8, 0.9])
    # Pass max_index=len(evals)-1 to search the full spectrum.
    # Largest gap is between index 2 and 3: 0.8 - 0.2 = 0.6
    k, gap = find_spectral_gap(evals, max_index=len(evals) - 1)
    assert k == 2
    assert_allclose(gap, 0.6, atol=1e-10)


def test_find_spectral_gap_edge_case_empty_range():
    # When min_index >= max_index, search_range is empty, returns (min_index, 0)
    evals = np.array([0.0, 0.5, 1.0])
    k, gap = find_spectral_gap(evals, min_index=2, max_index=2)
    assert k == 2
    assert gap == 0


# ---------------------------------------------------------------------------
# compute_diffusion_modes tests
# ---------------------------------------------------------------------------

def test_diffusion_modes_returns_k_plus_1(symmetric_adj):
    L = compute_laplacian(symmetric_adj)
    evals, evecs = compute_diffusion_modes(L, k=2)
    # Should return first k+1 = 3 modes
    assert len(evals) == 3
    assert evecs.shape[1] == 3


# ---------------------------------------------------------------------------
# analyze_spectral_properties tests
# ---------------------------------------------------------------------------

def test_spectral_properties_all_keys_present(symmetric_adj):
    L = compute_laplacian(symmetric_adj)
    evals, evecs = eigendecomposition(L)
    props = analyze_spectral_properties(evals, evecs)
    for key in ('spectral_gap_index', 'spectral_gap_size', 'algebraic_connectivity',
                'spectral_radius', 'participation_ratios', 'localization'):
        assert key in props


def test_spectral_properties_algebraic_connectivity_positive(symmetric_adj):
    # symmetric_adj is a connected graph, so Fiedler value > 0
    L = compute_laplacian(symmetric_adj)
    evals, evecs = eigendecomposition(L)
    props = analyze_spectral_properties(evals, evecs)
    assert props['algebraic_connectivity'] > 0


def test_spectral_properties_too_few_eigenvalues_raises(symmetric_adj):
    L = compute_laplacian(symmetric_adj)
    evals, evecs = eigendecomposition(L)
    with pytest.raises(ValueError):
        analyze_spectral_properties(evals[:1], evecs[:, :1])


# ---------------------------------------------------------------------------
# compute_spectral_embedding tests
# ---------------------------------------------------------------------------

def test_spectral_embedding_shape(symmetric_adj):
    L = compute_laplacian(symmetric_adj)
    dim = 2
    embedding = compute_spectral_embedding(L, dim=dim)
    assert embedding.shape == (symmetric_adj.shape[0], dim)


# ---------------------------------------------------------------------------
# compute_diffusion_distance tests
# ---------------------------------------------------------------------------

@pytest.fixture
def diffusion_distance_matrix(symmetric_adj):
    L = compute_laplacian(symmetric_adj)
    return compute_diffusion_distance(L, t=1.0)


def test_diffusion_distance_shape(symmetric_adj, diffusion_distance_matrix):
    n = symmetric_adj.shape[0]
    assert diffusion_distance_matrix.shape == (n, n)


def test_diffusion_distance_symmetric(diffusion_distance_matrix):
    D = diffusion_distance_matrix
    assert_allclose(D, D.T, atol=1e-12)


def test_diffusion_distance_diagonal_zero(diffusion_distance_matrix):
    assert_allclose(np.diag(diffusion_distance_matrix), 0, atol=1e-12)


def test_diffusion_distance_nonneg(diffusion_distance_matrix):
    assert np.all(diffusion_distance_matrix >= 0)


def test_diffusion_distance_triangle_inequality(diffusion_distance_matrix):
    D = diffusion_distance_matrix
    # spot-check: D[0,2] <= D[0,1] + D[1,2]
    assert D[0, 2] <= D[0, 1] + D[1, 2] + 1e-12
