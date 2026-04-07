"""
Tests for the data preprocessing module.

Uses pure pytest style (no unittest.TestCase).
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from scr_financial.data.preprocessor import DataPreprocessor
from scr_financial.data.collectors.eba_collector import EBACollector
from scr_financial.data.collectors.ecb_collector import ECBCollector
from scr_financial.data.collectors.market_collector import MarketDataCollector
from scr_financial.data.utils import (
    normalize_matrix,
    filter_matrix,
    align_time_series,
    compute_rolling_correlation,
    compute_distance_matrix,
    compute_minimum_spanning_tree,
)

START, END = "2020-01-01", "2021-12-31"
BANKS = ["DE_DBK", "FR_BNP", "ES_SAN"]


# ---------------------------------------------------------------------------
# DataPreprocessor validation tests
# ---------------------------------------------------------------------------

def test_preprocessor_invalid_start_date_raises():
    with pytest.raises(ValueError):
        DataPreprocessor("not-a-date", "2021-01-01")


def test_preprocessor_end_before_start_raises():
    with pytest.raises(ValueError):
        DataPreprocessor("2020-01-01", "2019-01-01")


def test_preprocessor_valid_init_no_error():
    dp = DataPreprocessor(START, END, bank_list=BANKS)
    assert dp.start_date == START
    assert dp.end_date == END


# ---------------------------------------------------------------------------
# EBACollector tests
# ---------------------------------------------------------------------------

@pytest.fixture
def eba():
    return EBACollector()


def test_eba_transparency_returns_dataframe(eba):
    df = eba.collect_transparency_data(START, END, bank_list=BANKS)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_eba_transparency_required_columns(eba):
    df = eba.collect_transparency_data(START, END, bank_list=BANKS)
    for col in ('bank_id', 'CET1_ratio', 'date'):
        assert col in df.columns, f"Missing column: {col}"


def test_eba_transparency_filtered_by_bank_list(eba):
    df = eba.collect_transparency_data(START, END, bank_list=BANKS)
    returned_banks = set(df['bank_id'].unique())
    assert returned_banks.issubset(set(BANKS))


def test_eba_aggregated_required_columns(eba):
    df = eba.collect_aggregated_data(START, END, bank_list=BANKS)
    for col in ('LCR', 'NSFR'):
        assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# ECBCollector tests
# ---------------------------------------------------------------------------

@pytest.fixture
def ecb():
    return ECBCollector()


def test_ecb_target2_required_columns(ecb):
    df = ecb.collect_target2_data(START, END, bank_list=BANKS)
    for col in ('source', 'target', 'weight'):
        assert col in df.columns, f"Missing column: {col}"


def test_ecb_ciss_values_in_unit_interval(ecb):
    df = ecb.collect_ciss_data(START, END)
    assert (df['CISS'] >= 0).all() and (df['CISS'] <= 1).all()


def test_ecb_ciss_required_columns(ecb):
    df = ecb.collect_ciss_data(START, END)
    assert 'CISS' in df.columns


# ---------------------------------------------------------------------------
# MarketDataCollector tests
# ---------------------------------------------------------------------------

@pytest.fixture
def market():
    return MarketDataCollector()


def test_market_cds_positive_spreads(market):
    df = market.collect_cds_data(START, END, bank_list=BANKS)
    assert (df['CDS_5yr'] > 0).all()


def test_market_srisk_required_columns(market):
    df = market.collect_srisk_data(START, END, bank_list=BANKS)
    assert 'SRISK' in df.columns


# ---------------------------------------------------------------------------
# normalize_matrix tests
# ---------------------------------------------------------------------------

def test_normalize_matrix_degree_preserves_shape():
    M = np.random.default_rng(0).random((4, 4))
    result = normalize_matrix(M, method='degree')
    assert result.shape == M.shape


def test_normalize_matrix_standardize_mean_approx_zero():
    M = np.random.default_rng(1).random((5, 5))
    result = normalize_matrix(M, method='standardize')
    assert_allclose(np.mean(result), 0.0, atol=1e-10)


def test_normalize_matrix_unknown_method_raises():
    with pytest.raises(ValueError):
        normalize_matrix(np.eye(3), method='bogus')


# ---------------------------------------------------------------------------
# filter_matrix tests
# ---------------------------------------------------------------------------

def test_filter_matrix_threshold_zeros_small():
    M = np.array([[0.01, 0.5], [0.03, 0.8]])
    result = filter_matrix(M, method='threshold', threshold=0.05)
    assert result[0, 0] == 0.0
    assert result[1, 0] == 0.0


def test_filter_matrix_preserves_large_values():
    M = np.array([[0.01, 0.5], [0.03, 0.8]])
    result = filter_matrix(M, method='threshold', threshold=0.05)
    assert_allclose(result[0, 1], 0.5)
    assert_allclose(result[1, 1], 0.8)


# ---------------------------------------------------------------------------
# align_time_series tests
# ---------------------------------------------------------------------------

def test_align_time_series_empty_returns_empty():
    result = align_time_series({})
    assert result == {}


# ---------------------------------------------------------------------------
# compute_rolling_correlation tests
# ---------------------------------------------------------------------------

def test_compute_rolling_correlation_entry_count():
    rng = np.random.default_rng(42)
    n = 100
    window = 20
    returns = pd.DataFrame(
        rng.normal(0, 1, (n, 3)),
        columns=['A', 'B', 'C'],
        index=pd.date_range('2020-01-01', periods=n),
    )
    result = compute_rolling_correlation(returns, window=window)
    assert len(result) == n - window + 1


# ---------------------------------------------------------------------------
# compute_distance_matrix tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_corr_matrix():
    rng = np.random.default_rng(0)
    n = 4
    X = rng.normal(0, 1, (100, n))
    df = pd.DataFrame(X, columns=[f"B{i}" for i in range(n)])
    return df.corr()


def test_compute_distance_matrix_symmetric(sample_corr_matrix):
    D = compute_distance_matrix(sample_corr_matrix)
    assert_allclose(D.values, D.values.T, atol=1e-10)


def test_compute_distance_matrix_diagonal_zero(sample_corr_matrix):
    D = compute_distance_matrix(sample_corr_matrix)
    assert_allclose(np.diag(D.values), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_minimum_spanning_tree tests
# ---------------------------------------------------------------------------

def test_compute_minimum_spanning_tree_edge_count(sample_corr_matrix):
    D = compute_distance_matrix(sample_corr_matrix)
    mst_adj = compute_minimum_spanning_tree(D)
    n = len(D)
    # MST on n nodes has n-1 edges; adjacency counts each undirected edge twice
    # number of non-zero entries / 2 == n - 1
    nnz = np.count_nonzero(mst_adj.values)
    assert nnz // 2 == n - 1
