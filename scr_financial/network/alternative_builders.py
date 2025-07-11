"""Alternative network construction methods."""

import logging
import numpy as np
import networkx as nx
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def build_partial_correlation_network(returns: pd.DataFrame, threshold=0.1) -> np.ndarray:
    """Build network from partial correlations (precision matrix)."""
    try:
        from sklearn.covariance import GraphicalLassoCV
        model = GraphicalLassoCV(cv=5, max_iter=500).fit(returns.values)
        precision = model.precision_
        # Partial correlations from precision matrix
        d = np.sqrt(np.diag(precision))
        partial_corr = -precision / np.outer(d, d)
        np.fill_diagonal(partial_corr, 0)
        adj = np.abs(partial_corr)
        adj[adj < threshold] = 0
        return adj
    except Exception as e:
        logger.warning("GraphicalLasso failed: %s. Falling back to inverse correlation.", e)
        corr = returns.corr().values
        try:
            precision = np.linalg.inv(corr)
            d = np.sqrt(np.abs(np.diag(precision)))
            d[d < 1e-10] = 1
            partial_corr = -precision / np.outer(d, d)
            np.fill_diagonal(partial_corr, 0)
            adj = np.abs(partial_corr)
            adj[adj < threshold] = 0
            return adj
        except Exception:
            return np.zeros((len(returns.columns), len(returns.columns)))


def build_granger_causality_network(returns: pd.DataFrame, max_lag=5, alpha=0.05) -> np.ndarray:
    """Build directed network from pairwise Granger causality tests."""
    n = len(returns.columns)
    adj = np.zeros((n, n))

    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                try:
                    data = pd.DataFrame({
                        'y': returns.iloc[:, j].values,
                        'x': returns.iloc[:, i].values,
                    })
                    result = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)
                    min_p = min(result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))
                    if min_p < alpha:
                        adj[i, j] = 1 - min_p
                except Exception:
                    pass
    except ImportError:
        logger.warning("statsmodels not available for Granger causality. Using correlation fallback.")
        corr = np.abs(returns.corr().values)
        np.fill_diagonal(corr, 0)
        adj = corr

    return adj


def build_pmfg(corr_matrix: np.ndarray) -> np.ndarray:
    """Build Planar Maximally Filtered Graph.

    Greedily adds edges in decreasing correlation order while maintaining planarity.
    """
    n = corr_matrix.shape[0]
    max_edges = 3 * (n - 2)

    # Get all edges sorted by correlation (descending)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((corr_matrix[i, j], i, j))
    edges.sort(reverse=True)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    adj = np.zeros((n, n))

    for weight, i, j in edges:
        if G.number_of_edges() >= max_edges:
            break
        G.add_edge(i, j, weight=weight)
        if nx.check_planarity(G)[0]:
            adj[i, j] = weight
            adj[j, i] = weight
        else:
            G.remove_edge(i, j)

    return adj


def build_mst(corr_matrix: np.ndarray) -> np.ndarray:
    """Build Minimum Spanning Tree from correlation distance."""
    n = corr_matrix.shape[0]
    dist = np.sqrt(2 * (1 - np.clip(corr_matrix, -1, 1)))
    np.fill_diagonal(dist, 0)

    G = nx.from_numpy_array(dist)
    mst = nx.minimum_spanning_tree(G, weight='weight')

    adj = np.zeros((n, n))
    for u, v, d in mst.edges(data=True):
        w = corr_matrix[u, v]
        adj[u, v] = w
        adj[v, u] = w
    return adj


def compare_networks(returns: pd.DataFrame, corr_matrix: Optional[np.ndarray] = None,
                     threshold: float = 0.3) -> pd.DataFrame:
    """Compare network construction methods."""
    from .spectral import compute_laplacian, eigendecomposition, find_spectral_gap

    if corr_matrix is None:
        corr_matrix = returns.corr().values

    methods = {
        "Threshold": lambda: _threshold_adj(corr_matrix, threshold),
        "MST": lambda: build_mst(corr_matrix),
        "PMFG": lambda: build_pmfg(corr_matrix),
        "Partial Corr": lambda: build_partial_correlation_network(returns, threshold=0.05),
    }

    rows = []
    for name, builder in methods.items():
        try:
            adj = builder()
            n = adj.shape[0]
            n_edges = np.count_nonzero(adj) // 2
            density = n_edges / (n * (n - 1) / 2) if n > 1 else 0

            L = compute_laplacian(adj, normalized=True)
            eigenvalues, _ = eigendecomposition(L)
            gap_idx, gap_size = find_spectral_gap(eigenvalues)
            lam2 = eigenvalues[1] if len(eigenvalues) > 1 else 0

            rows.append({"method": name, "n_edges": n_edges, "density": round(density, 3),
                         "lambda_2": round(float(lam2), 4), "spectral_gap": round(float(gap_size), 4)})
        except Exception as e:
            logger.warning("Method %s failed: %s", name, e)
            rows.append({"method": name, "n_edges": 0, "density": 0, "lambda_2": 0, "spectral_gap": 0})

    return pd.DataFrame(rows)


def _threshold_adj(corr_matrix, threshold):
    adj = corr_matrix.copy()
    np.fill_diagonal(adj, 0)
    adj[adj < threshold] = 0
    return adj
