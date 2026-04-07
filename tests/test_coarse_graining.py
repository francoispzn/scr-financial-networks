"""
Tests for the spectral coarse-graining module.

Uses pure pytest style (no unittest.TestCase).
"""

import pytest
import numpy as np
import scipy.sparse as sp
import networkx as nx
from scipy.linalg import eigh
from numpy.testing import assert_allclose

from scr_financial.network.builder import FinancialNetworkBuilder
from scr_financial.network.coarse_graining import SpectralCoarseGraining


# ---------------------------------------------------------------------------
# Builder stub (no real DataPreprocessor needed)
# ---------------------------------------------------------------------------

def _make_builder(n=6):
    G = nx.complete_graph(n, create_using=nx.DiGraph)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    builder = object.__new__(FinancialNetworkBuilder)
    builder.G = G
    builder.adjacency_matrix = nx.to_scipy_sparse_array(G, weight='weight')
    L_dense = nx.normalized_laplacian_matrix(G.to_undirected()).toarray()
    builder.laplacian = sp.csr_matrix(L_dense)
    builder.eigenvalues, builder.eigenvectors = eigh(L_dense)
    builder.preprocessor = None
    return builder


@pytest.fixture
def builder():
    return _make_builder(n=6)


@pytest.fixture
def scg(builder):
    return SpectralCoarseGraining(builder)


# ---------------------------------------------------------------------------
# coarse_grain tests
# ---------------------------------------------------------------------------

def test_coarse_grain_returns_ndarray(scg):
    result = scg.coarse_grain()
    assert isinstance(result, np.ndarray)


def test_coarse_grain_shape(scg, builder):
    n = builder.G.number_of_nodes()
    result = scg.coarse_grain()
    assert result.shape == (n, n)


def test_coarse_grain_stores_adjacency(scg):
    scg.coarse_grain()
    assert scg.coarse_grained_adjacency is not None


# ---------------------------------------------------------------------------
# rescale tests
# ---------------------------------------------------------------------------

def test_rescale_raises_before_coarse_grain(scg):
    with pytest.raises(ValueError):
        scg.rescale()


def test_rescale_returns_ndarray_after_coarse_grain(scg):
    scg.coarse_grain()
    result = scg.rescale()
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# identify_clusters tests
# ---------------------------------------------------------------------------

def test_identify_clusters_all_nodes_assigned(scg, builder):
    n = builder.G.number_of_nodes()
    clusters = scg.identify_clusters(n_clusters=2)
    assert len(clusters) == n


def test_identify_clusters_count_matches_n_clusters(scg):
    n_clusters = 2
    clusters = scg.identify_clusters(n_clusters=n_clusters)
    assert len(set(clusters)) == n_clusters


# ---------------------------------------------------------------------------
# map_node_to_cluster tests
# ---------------------------------------------------------------------------

def test_map_node_to_cluster_raises_for_unknown(scg):
    scg.identify_clusters(n_clusters=2)
    with pytest.raises(ValueError):
        scg.map_node_to_cluster("nonexistent_node_xyz")


def test_map_node_to_cluster_returns_cluster_str(scg, builder):
    scg.identify_clusters(n_clusters=2)
    # Use the first node in the graph
    first_node = list(builder.G.nodes())[0]
    result = scg.map_node_to_cluster(first_node)
    assert result.startswith("cluster_")


# ---------------------------------------------------------------------------
# compute_coarse_graining_error tests
# ---------------------------------------------------------------------------

def test_cg_error_nonneg(scg):
    scg.coarse_grain()
    error = scg.compute_coarse_graining_error()
    assert error >= 0.0


# ---------------------------------------------------------------------------
# compare_diffusion_dynamics tests
# ---------------------------------------------------------------------------

def test_compare_diffusion_length(scg):
    scg.coarse_grain()
    time_steps = 5
    errors = scg.compare_diffusion_dynamics(time_steps=time_steps)
    assert len(errors) == time_steps


def test_compare_diffusion_errors_nonneg(scg):
    scg.coarse_grain()
    errors = scg.compare_diffusion_dynamics(time_steps=4)
    assert all(e >= 0.0 for e in errors)


# ---------------------------------------------------------------------------
# create_coarse_grained_graph tests
# ---------------------------------------------------------------------------

def test_create_cg_graph_node_count(scg):
    n_clusters = 2
    scg.identify_clusters(n_clusters=n_clusters)
    G_cg = scg.create_coarse_grained_graph()
    assert G_cg.number_of_nodes() == n_clusters
