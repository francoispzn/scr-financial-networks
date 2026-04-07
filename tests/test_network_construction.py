"""
Tests for the network construction module.

Uses pure pytest style (no unittest.TestCase).
"""

import pytest
import numpy as np
import pandas as pd
import networkx as nx
from unittest.mock import MagicMock
from numpy.testing import assert_allclose

from scr_financial.network.builder import FinancialNetworkBuilder

N_BANKS = 4
BANKS = [f"B{i}" for i in range(N_BANKS)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder():
    rows = [
        {
            'source': s,
            'target': t,
            'weight': float(i + 1),
            'date': pd.Timestamp('2022-01-01'),
        }
        for i, (s, t) in enumerate(
            (s, t) for s in BANKS for t in BANKS if s != t
        )
    ]
    edge_df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.get_data_for_timepoint.return_value = {
        'edge_data': {'interbank_exposures': edge_df},
        'node_data': {},
    }
    b = FinancialNetworkBuilder(mock)
    b.construct_network('2022-01-01')
    return b


# ---------------------------------------------------------------------------
# construct_network tests
# ---------------------------------------------------------------------------

def test_construct_network_returns_digraph(builder):
    assert isinstance(builder.G, nx.DiGraph)


def test_construct_network_correct_node_count(builder):
    assert builder.G.number_of_nodes() == N_BANKS


def test_construct_network_correct_edge_count(builder):
    assert builder.G.number_of_edges() == N_BANKS * (N_BANKS - 1)


def test_construct_network_unknown_edge_type_raises():
    mock = MagicMock()
    mock.get_data_for_timepoint.return_value = {
        'edge_data': {},       # missing 'unknown_type'
        'node_data': {},
    }
    b = FinancialNetworkBuilder(mock)
    with pytest.raises(ValueError):
        b.construct_network('2022-01-01', edge_weight_type='unknown_type')


# ---------------------------------------------------------------------------
# compute_laplacian tests
# ---------------------------------------------------------------------------

def test_compute_laplacian_normalized_shape(builder):
    L = builder.compute_laplacian(normalized=True)
    dense = np.asarray(L.todense())
    assert dense.shape == (N_BANKS, N_BANKS)


def test_compute_laplacian_raises_before_network():
    mock = MagicMock()
    b = FinancialNetworkBuilder(mock)
    with pytest.raises(ValueError):
        b.compute_laplacian()


# ---------------------------------------------------------------------------
# spectral_analysis tests
# ---------------------------------------------------------------------------

def test_spectral_analysis_eigenvalues_sorted(builder):
    builder.compute_laplacian()
    evals, _ = builder.spectral_analysis()
    assert np.all(np.diff(evals) >= -1e-10)


def test_spectral_analysis_raises_before_laplacian():
    mock = MagicMock()
    mock.get_data_for_timepoint.return_value = {
        'edge_data': {
            'interbank_exposures': pd.DataFrame([
                {'source': 'X', 'target': 'Y', 'weight': 1.0, 'date': pd.Timestamp('2022-01-01')}
            ])
        },
        'node_data': {},
    }
    b = FinancialNetworkBuilder(mock)
    b.construct_network('2022-01-01')
    # laplacian not computed yet
    with pytest.raises(ValueError):
        b.spectral_analysis()


# ---------------------------------------------------------------------------
# find_spectral_gap tests
# ---------------------------------------------------------------------------

def test_find_spectral_gap_returns_int_and_float(builder):
    builder.compute_laplacian()
    builder.spectral_analysis()
    k, gap = builder.find_spectral_gap()
    assert isinstance(k, (int, np.integer))
    assert isinstance(gap, (float, np.floating))


# ---------------------------------------------------------------------------
# get_node_attribute_matrix tests
# ---------------------------------------------------------------------------

def test_get_node_attribute_matrix_defaults_to_zero(builder):
    values = builder.get_node_attribute_matrix('nonexistent_attr')
    assert values.shape == (N_BANKS,)
    assert_allclose(values, 0.0)


# ---------------------------------------------------------------------------
# get_edge_weight_matrix tests
# ---------------------------------------------------------------------------

def test_get_edge_weight_matrix_shape(builder):
    matrix = builder.get_edge_weight_matrix()
    assert np.asarray(matrix).shape == (N_BANKS, N_BANKS)


# ---------------------------------------------------------------------------
# compute_centrality_measures tests
# ---------------------------------------------------------------------------

def test_compute_centrality_all_five_keys(builder):
    centrality = builder.compute_centrality_measures()
    for key in ('degree', 'eigenvector', 'betweenness', 'closeness', 'pagerank'):
        assert key in centrality


def test_compute_centrality_all_nodes_present(builder):
    centrality = builder.compute_centrality_measures()
    for key in ('degree', 'eigenvector', 'betweenness', 'closeness', 'pagerank'):
        for bank in BANKS:
            assert bank in centrality[key]


# ---------------------------------------------------------------------------
# compute_community_structure tests
# ---------------------------------------------------------------------------

def test_compute_community_structure_spectral_returns_dict(builder):
    builder.compute_laplacian()
    builder.spectral_analysis()
    communities = builder.compute_community_structure(method='spectral')
    assert isinstance(communities, dict)
    for node in builder.G.nodes():
        assert isinstance(communities[node], (int, np.integer))
