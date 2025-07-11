"""Principled threshold selection for network construction."""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def information_theoretic_threshold(corr_matrix: np.ndarray,
                                     thresholds: Optional[List[float]] = None) -> Dict:
    """Select threshold via Minimum Description Length (MDL).

    For each threshold, compute: L(G) = -log P(edges|G) + |E| * log(N)
    Pick the threshold minimizing total description length.
    """
    n = corr_matrix.shape[0]
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.8, 30).tolist()

    results = []
    for thr in thresholds:
        adj = (np.abs(corr_matrix) >= thr).astype(float)
        np.fill_diagonal(adj, 0)
        n_edges = np.count_nonzero(adj) // 2
        density = n_edges / max(n * (n - 1) / 2, 1)

        # MDL: encoding cost = edges * log(N) + residual error
        if density > 0 and density < 1:
            encoding_cost = n_edges * np.log(n)
            # Residual: sum of squared correlation below threshold
            residual = np.sum(corr_matrix[np.abs(corr_matrix) < thr] ** 2)
            mdl = encoding_cost + residual
        elif density == 0:
            mdl = np.sum(corr_matrix ** 2)  # All edges as residual
        else:
            mdl = n * (n - 1) / 2 * np.log(n)  # Full graph encoding

        results.append({"threshold": thr, "n_edges": n_edges,
                         "density": round(density, 4), "mdl": round(float(mdl), 2)})

    best = min(results, key=lambda r: r["mdl"])
    return {"optimal_threshold": best["threshold"], "mdl_scores": results, "best": best}


def cv_based_threshold(returns_matrix: np.ndarray, thresholds: Optional[List[float]] = None,
                       n_folds: int = 5, target: str = "spectral_gap_stability") -> Dict:
    """Cross-validate threshold choice by spectral stability across held-out windows.

    For each threshold, compute spectral gap across time windows and measure
    stability (lower CV = more stable = better threshold).
    """
    from .spectral import compute_laplacian, eigendecomposition, find_spectral_gap

    T, N = returns_matrix.shape
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.7, 15).tolist()

    fold_size = T // n_folds
    results = []

    for thr in thresholds:
        fold_gaps = []
        fold_lam2s = []

        for fold in range(n_folds):
            start = fold * fold_size
            end = min(start + fold_size, T)
            if end - start < N + 10:
                continue

            window_returns = returns_matrix[start:end]
            corr = np.corrcoef(window_returns.T)
            adj = corr.copy()
            np.fill_diagonal(adj, 0)
            adj[adj < thr] = 0

            try:
                L = compute_laplacian(adj, normalized=True)
                eigenvalues, _ = eigendecomposition(L)
                gap_idx, gap_size = find_spectral_gap(eigenvalues)
                lam2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
                fold_gaps.append(gap_size)
                fold_lam2s.append(lam2)
            except Exception:
                pass

        if len(fold_gaps) >= 2:
            gap_cv = np.std(fold_gaps) / max(np.mean(fold_gaps), 1e-10)
            lam2_cv = np.std(fold_lam2s) / max(np.mean(fold_lam2s), 1e-10)
        else:
            gap_cv = lam2_cv = float("inf")

        results.append({
            "threshold": thr,
            "gap_mean": round(float(np.mean(fold_gaps)), 4) if fold_gaps else 0,
            "gap_cv": round(float(gap_cv), 4),
            "lam2_mean": round(float(np.mean(fold_lam2s)), 4) if fold_lam2s else 0,
            "lam2_cv": round(float(lam2_cv), 4),
        })

    # Best = lowest coefficient of variation (most stable)
    valid = [r for r in results if r["gap_cv"] < float("inf")]
    if valid:
        best = min(valid, key=lambda r: r["gap_cv"])
    else:
        best = results[len(results) // 2] if results else {"threshold": 0.3}

    return {"optimal_threshold": best["threshold"], "cv_scores": results, "best": best}


def percolation_threshold(corr_matrix: np.ndarray, n_steps: int = 50) -> Dict:
    """Find the threshold at which the giant component emerges.

    Below this threshold, the network fragments into disconnected components.
    Above it, a giant connected component exists.
    """
    import networkx as nx
    n = corr_matrix.shape[0]
    thresholds = np.linspace(0.01, 0.95, n_steps)

    results = []
    for thr in thresholds:
        adj = corr_matrix.copy()
        np.fill_diagonal(adj, 0)
        adj[adj < thr] = 0

        G = nx.from_numpy_array(adj)
        components = list(nx.connected_components(G))
        largest = max(len(c) for c in components) if components else 0

        results.append({
            "threshold": round(float(thr), 3),
            "n_components": len(components),
            "largest_component": largest,
            "fraction_in_giant": round(largest / n, 3),
        })

    # Percolation threshold: where giant component drops below 90% of N
    percolation_thr = 0.3  # default
    for i in range(len(results) - 1):
        if results[i]["fraction_in_giant"] >= 0.9 and results[i + 1]["fraction_in_giant"] < 0.9:
            percolation_thr = results[i + 1]["threshold"]
            break

    return {"percolation_threshold": percolation_thr, "results": results}
