"""Multi-scale spectral coarse-graining."""

import logging
import numpy as np
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MultiScaleSCG:
    """Run SCG at multiple scales and record preservation metrics."""

    def __init__(self, adjacency: np.ndarray, node_ids: List[str]):
        self.adjacency = adjacency
        self.node_ids = node_ids
        self.results: List[Dict] = []

    def run_all_scales(self, max_k: Optional[int] = None) -> List[Dict]:
        """Coarse-grain at each scale k=1..max_k."""
        from .coarse_graining import SpectralCoarseGraining

        n = self.adjacency.shape[0]
        if max_k is None:
            max_k = n // 2

        self.results = []
        for k in range(1, max_k + 1):
            try:
                scg = SpectralCoarseGraining.from_adjacency(self.adjacency, self.node_ids)
                scg.coarse_grain(k=k)
                scg.identify_clusters()
                acc = scg.compute_reconstruction_accuracy(time_steps=10)
                n_clusters = len(set(scg.clusters)) if scg.clusters is not None else 0

                r2_mean = float(np.mean(acc.get("r2", [0])))
                rmse_mean = float(np.mean(acc.get("rmse", [0])))

                self.results.append({
                    "k": k, "n_clusters": n_clusters,
                    "r2_mean": round(r2_mean, 6),
                    "rmse_mean": round(rmse_mean, 8),
                    "cluster_sizes": [int(np.sum(scg.clusters == c))
                                      for c in range(n_clusters)] if scg.clusters is not None else [],
                })
            except Exception as e:
                logger.warning("Multi-scale SCG failed at k=%d: %s", k, e)
                self.results.append({"k": k, "error": str(e)})

        return self.results

    def optimal_scale(self) -> int:
        """Return k that maximizes R² while minimizing clusters."""
        valid = [r for r in self.results if "r2_mean" in r and r["r2_mean"] > 0.9]
        if not valid:
            return 1
        # Prefer fewest clusters among high-R² results
        return min(valid, key=lambda r: r["n_clusters"])["k"]


def villegas_lrg(adjacency: np.ndarray, n_steps: int = 5) -> List[Dict]:
    """Simplified Villegas et al. (2023) Laplacian Renormalization Group.

    At each step: compute diffusion at multiple timescales, cluster by
    diffusion similarity, contract clusters into super-nodes.
    """
    from .spectral import compute_laplacian, eigendecomposition
    from scipy.linalg import expm
    from sklearn.cluster import KMeans

    current_adj = adjacency.copy()
    results = []

    for step in range(n_steps):
        n = current_adj.shape[0]
        if n <= 2:
            break

        adj_sym = (current_adj + current_adj.T) / 2
        np.fill_diagonal(adj_sym, 0)

        degrees = adj_sym.sum(axis=1)
        if np.any(degrees < 1e-10):
            adj_sym += 0.01 * np.ones_like(adj_sym)
            np.fill_diagonal(adj_sym, 0)
            degrees = adj_sym.sum(axis=1)

        L = np.diag(degrees) - adj_sym
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        lam2 = float(eigenvalues[1]) if n > 1 else 0
        rho = float(eigenvalues[-1])
        n_edges = np.count_nonzero(adj_sym) // 2

        results.append({
            "step": step, "n_nodes": n, "n_edges": n_edges,
            "lambda_2": round(lam2, 4), "spectral_radius": round(rho, 4),
            "density": round(n_edges / max(n * (n-1) / 2, 1), 3),
        })

        # Diffusion-based clustering at optimal timescale
        tau = 1.0 / max(eigenvalues[1], 0.01) if n > 1 else 1.0
        H = expm(-tau * L)

        # Number of clusters: halve at each step (minimum 2)
        k = max(2, n // 2)
        if k >= n:
            break

        # Cluster by diffusion similarity
        embedding = eigenvectors[:, 1:min(k+1, n)]
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1
        embedding = embedding / norms

        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embedding)

        # Contract: build new adjacency
        new_adj = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                nodes_i = np.where(labels == i)[0]
                nodes_j = np.where(labels == j)[0]
                new_adj[i, j] = np.sum(adj_sym[np.ix_(nodes_i, nodes_j)])

        current_adj = new_adj

    return results


def compare_scg_vs_lrg(adjacency: np.ndarray, node_ids: List[str]) -> Dict:
    """Compare Schmidt SCG vs Villegas LRG."""
    import pandas as pd

    ms = MultiScaleSCG(adjacency, node_ids)
    scg_results = ms.run_all_scales()
    lrg_results = villegas_lrg(adjacency)

    return {
        "scg_results": scg_results,
        "lrg_results": lrg_results,
        "scg_optimal_k": ms.optimal_scale(),
        "comparison": pd.DataFrame([
            {"method": "SCG", "n_scales": len(scg_results), "best_r2": max(r.get("r2_mean", 0) for r in scg_results) if scg_results else 0},
            {"method": "LRG", "n_steps": len(lrg_results), "final_nodes": lrg_results[-1]["n_nodes"] if lrg_results else 0},
        ]).to_dict(orient="records"),
    }
