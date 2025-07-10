"""
Spectral coarse-graining for financial networks.

This module implements spectral coarse-graining techniques for simplifying
financial networks while preserving their essential structural and dynamic
properties.
"""

import logging

import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.linalg as la
from scipy.linalg import expm
from typing import Dict, List, Optional, Tuple, Union
from sklearn.cluster import KMeans, SpectralClustering

from .builder import FinancialNetworkBuilder

logger = logging.getLogger(__name__)


class _AdjacencyAdapter:
    """Lightweight adapter mimicking FinancialNetworkBuilder for SCG from a raw adjacency matrix."""

    def __init__(self, adj_matrix: np.ndarray, node_ids: List[str]) -> None:
        n = adj_matrix.shape[0]
        self.G = nx.DiGraph()
        for i, nid in enumerate(node_ids):
            self.G.add_node(nid)
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:
                    self.G.add_edge(node_ids[i], node_ids[j], weight=adj_matrix[i, j])

        adj_sym = (adj_matrix + adj_matrix.T) / 2.0
        self.adjacency_matrix = sp.csr_matrix(adj_sym)
        degree = np.sum(adj_sym, axis=1)
        self.laplacian = sp.csr_matrix(np.diag(degree) - adj_sym)
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None

    def spectral_analysis(self) -> Tuple[np.ndarray, np.ndarray]:
        L_dense = np.asarray(self.laplacian.todense())
        self.eigenvalues, self.eigenvectors = la.eigh(L_dense)
        return self.eigenvalues, self.eigenvectors

    def find_spectral_gap(self) -> Tuple[int, float]:
        if self.eigenvalues is None:
            self.spectral_analysis()
        from .spectral import find_spectral_gap as _find_gap
        adj_dense = np.asarray(self.adjacency_matrix.todense())
        return _find_gap(self.eigenvalues, adjacency_matrix=adj_dense)


class SpectralCoarseGraining:
    """
    Implements spectral coarse-graining for financial networks.

    Attributes:
        coarse_grained_laplacian: Coarse-grained Laplacian matrix.
        coarse_grained_adjacency: Coarse-grained adjacency matrix.
        rescaled_adjacency: Rescaled coarse-grained adjacency matrix.
        clusters: Cluster assignments for nodes.
    """

    def __init__(self, network_builder: FinancialNetworkBuilder) -> None:
        """
        Initialize the coarse-graining with a network builder.

        Args:
            network_builder: Network builder containing the network to
                coarse-grain.
        """
        self.network_builder = network_builder
        self.coarse_grained_laplacian: Optional[np.ndarray] = None
        self.coarse_grained_adjacency: Optional[np.ndarray] = None
        self.rescaled_adjacency: Optional[np.ndarray] = None
        self.clusters: Optional[np.ndarray] = None
        self.cluster_mapping: Optional[Dict] = None
        self.coarse_grained_graph: Optional[nx.Graph] = None

    @classmethod
    def from_adjacency(cls, adj_matrix: np.ndarray, node_ids: List[str]) -> "SpectralCoarseGraining":
        """Create an SCG instance directly from an adjacency matrix (no DataPreprocessor needed)."""
        adapter = _AdjacencyAdapter(adj_matrix, node_ids)
        instance = cls.__new__(cls)
        instance.network_builder = adapter
        instance.coarse_grained_laplacian = None
        instance.coarse_grained_adjacency = None
        instance.rescaled_adjacency = None
        instance.clusters = None
        instance.cluster_mapping = None
        instance.coarse_grained_graph = None
        return instance

    def coarse_grain(self, k: Optional[int] = None) -> np.ndarray:
        """
        Perform spectral coarse-graining using k eigenmodes.

        Args:
            k: Number of eigenmodes to use. If None, determined automatically
                using spectral gap.

        Returns:
            Coarse-grained Laplacian matrix as an ndarray.
        """
        if (
            self.network_builder.eigenvalues is None
            or self.network_builder.eigenvectors is None
        ):
            self.network_builder.spectral_analysis()

        eigenvalues = self.network_builder.eigenvalues
        eigenvectors = self.network_builder.eigenvectors

        if k is None:
            # Use spectral gap to determine k
            k, _ = self.network_builder.find_spectral_gap()

        # Construct coarse-grained Laplacian using first k eigenmodes.
        # Use np.zeros with the sparse matrix shape to avoid np.matrix.
        L_cg = np.zeros(self.network_builder.laplacian.shape)
        for alpha in range(k + 1):  # Include eigenmode 0 to k
            lambda_alpha = eigenvalues[alpha]
            u_alpha = eigenvectors[:, alpha].reshape(-1, 1)
            L_cg += lambda_alpha * (u_alpha @ u_alpha.T)

        self.coarse_grained_laplacian = L_cg

        # Extract adjacency A₁ from the reconstructed Laplacian (Eq. 4)
        D = np.diag(np.diag(L_cg))
        A1 = D - L_cg

        # Topology masking (Eq. 5): A₂ᵢⱼ = A₁ᵢⱼ if original Aᵢⱼ > 0, else 0
        if hasattr(self.network_builder, 'adjacency_matrix') and self.network_builder.adjacency_matrix is not None:
            orig_A = np.asarray(self.network_builder.adjacency_matrix.todense())
            mask = (orig_A > 0).astype(float)
            A1 *= mask

        self.coarse_grained_adjacency = A1

        return self.coarse_grained_laplacian

    def contract_vertices(self) -> np.ndarray:
        """Vertex contraction: merge nodes with negative off-diagonal in A₁.

        Nodes that are "similar" at the coarse scale (negative off-diagonal
        entries in the coarse-grained adjacency) are merged into super-nodes
        with summed edge weights.

        Returns:
            Contracted adjacency matrix (may be smaller than original).
        """
        if self.coarse_grained_adjacency is None:
            raise ValueError("Call coarse_grain() first.")

        A = self.coarse_grained_adjacency.copy()
        n = A.shape[0]

        # Build union-find for nodes that share a negative off-diagonal entry
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] < 0:
                    union(i, j)

        # Build mapping: root → list of members
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        group_list = sorted(groups.values(), key=lambda g: g[0])
        m = len(group_list)

        if m == n:
            # No contraction needed
            self.contracted_adjacency = np.maximum(A, 0)
            self._contraction_map = {i: i for i in range(n)}
            return self.contracted_adjacency

        # Build contracted adjacency by summing edges between super-nodes
        A_contracted = np.zeros((m, m))
        node_to_super = {}
        for si, members in enumerate(group_list):
            for node in members:
                node_to_super[node] = si

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                si, sj = node_to_super[i], node_to_super[j]
                if si != sj and A[i, j] > 0:
                    A_contracted[si, sj] += A[i, j]

        self.contracted_adjacency = A_contracted
        self._contraction_map = node_to_super
        self._contraction_groups = group_list
        logger.info("Vertex contraction: %d → %d super-nodes", n, m)
        return self.contracted_adjacency

    def rescale(self, lambda_k: Optional[float] = None) -> np.ndarray:
        """
        Rescale the coarse-grained adjacency matrix.

        Args:
            lambda_k: Eigenvalue to use for rescaling. If None, uses the k-th
                eigenvalue where k is from find_spectral_gap.

        Returns:
            Rescaled coarse-grained adjacency matrix.

        Raises:
            ValueError: If coarse-grained adjacency has not been computed.
        """
        if self.coarse_grained_adjacency is None:
            raise ValueError(
                "Coarse-grained adjacency not computed. Call coarse_grain first."
            )

        if lambda_k is None:
            k, _ = self.network_builder.find_spectral_gap()
            lambda_k = self.network_builder.eigenvalues[k]

        if lambda_k <= 0:
            raise ValueError(f"lambda_k must be positive, got {lambda_k}")

        # Use contracted adjacency if available, otherwise coarse-grained
        base = getattr(self, 'contracted_adjacency', None)
        if base is None:
            base = self.coarse_grained_adjacency

        # Rescale by 1/lambda_k to preserve diffusion dynamics (Eq. 6)
        self.rescaled_adjacency = (1 / lambda_k) * base

        return self.rescaled_adjacency

    def identify_clusters(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Identify clusters based on eigenvectors.

        Args:
            n_clusters: Number of clusters. If None, determined from spectral
                gap.

        Returns:
            Cluster assignments for nodes.
        """
        if self.network_builder.eigenvectors is None:
            self.network_builder.spectral_analysis()

        if n_clusters is None:
            k, _ = self.network_builder.find_spectral_gap()
            n_clusters = k + 1  # Use k+1 clusters

        # Extract relevant eigenvectors (skip first — trivial)
        eigenvectors = self.network_builder.eigenvectors[
            :, 1 : n_clusters + 1
        ]

        # Spectral partition: normalise rows and use sign structure
        # For 2 clusters use Fiedler vector sign; for k>2 use KMeans on
        # the normalised spectral embedding (Ng, Jordan & Weiss 2002).
        if n_clusters == 2:
            fiedler = eigenvectors[:, 0]
            self.clusters = (fiedler >= 0).astype(int)
        else:
            row_norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-12)
            embedding = eigenvectors / row_norms
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = kmeans.fit_predict(embedding)

        # Create mapping from nodes to clusters
        nodes = list(self.network_builder.G.nodes())
        self.cluster_mapping = {nodes[i]: self.clusters[i] for i in range(len(nodes))}

        return self.clusters

    def create_coarse_grained_graph(self) -> nx.Graph:
        """
        Create a coarse-grained graph based on identified clusters.

        Returns:
            Coarse-grained graph.
        """
        if self.clusters is None:
            self.identify_clusters()

        # Create a new graph with clusters as nodes
        G_cg = nx.DiGraph()

        # Add cluster nodes
        n_clusters = len(set(self.clusters))
        for i in range(n_clusters):
            G_cg.add_node(f"cluster_{i}")

        # Compute inter-cluster weights
        original_graph = self.network_builder.G
        nodes = list(original_graph.nodes())

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    # Find nodes in each cluster
                    nodes_i = [
                        nodes[k] for k in range(len(nodes)) if self.clusters[k] == i
                    ]
                    nodes_j = [
                        nodes[k] for k in range(len(nodes)) if self.clusters[k] == j
                    ]

                    # Compute total weight between clusters
                    total_weight = 0
                    for source in nodes_i:
                        for target in nodes_j:
                            if original_graph.has_edge(source, target):
                                total_weight += original_graph[source][target]["weight"]

                    # Add edge if there's a connection
                    if total_weight > 0:
                        G_cg.add_edge(
                            f"cluster_{i}", f"cluster_{j}", weight=total_weight
                        )

        # Compute aggregate node attributes
        for i in range(n_clusters):
            nodes_i = [
                nodes[k] for k in range(len(nodes)) if self.clusters[k] == i
            ]

            # Guard: skip attribute aggregation if cluster is empty
            if not nodes_i:
                return G_cg

            # Aggregate node attributes
            for attr in original_graph.nodes[nodes_i[0]]:
                # Skip non-numeric attributes
                if not isinstance(
                    original_graph.nodes[nodes_i[0]][attr], (int, float)
                ):
                    continue

                # Compute average
                avg_value = np.mean(
                    [
                        original_graph.nodes[node][attr]
                        for node in nodes_i
                        if attr in original_graph.nodes[node]
                    ]
                )

                G_cg.nodes[f"cluster_{i}"][attr] = avg_value

            # Store original nodes in this cluster
            G_cg.nodes[f"cluster_{i}"]["original_nodes"] = nodes_i
            G_cg.nodes[f"cluster_{i}"]["size"] = len(nodes_i)

        self.coarse_grained_graph = G_cg

        return G_cg

    def map_node_to_cluster(self, node: str) -> str:
        """
        Map an original node to its cluster.

        Args:
            node: Original node identifier.

        Returns:
            Cluster identifier.

        Raises:
            ValueError: If clusters have not been identified or node is not
                found.
        """
        if self.cluster_mapping is None:
            raise ValueError("Clusters not identified. Call identify_clusters first.")

        if node not in self.cluster_mapping:
            raise ValueError(f"Node {node} not found in the network")

        return f"cluster_{self.cluster_mapping[node]}"

    def compute_coarse_graining_error(self) -> float:
        """
        Compute the error introduced by coarse-graining.

        Returns:
            Relative error in eigenvalues.

        Raises:
            ValueError: If coarse-grained Laplacian has not been computed.
        """
        if self.coarse_grained_laplacian is None:
            raise ValueError(
                "Coarse-grained Laplacian not computed. Call coarse_grain first."
            )

        # Get original eigenvalues
        orig_eigenvalues = self.network_builder.eigenvalues

        # Compute eigenvalues of coarse-grained Laplacian
        cg_eigenvalues, _ = la.eigh(self.coarse_grained_laplacian)

        # Determine k
        k, _ = self.network_builder.find_spectral_gap()

        # Compare first k+1 eigenvalues
        error = np.linalg.norm(
            orig_eigenvalues[: k + 1] - cg_eigenvalues[: k + 1]
        ) / np.linalg.norm(orig_eigenvalues[: k + 1])

        return error

    def compute_reconstruction_accuracy(self, time_steps: int = 15) -> Dict[str, List[float]]:
        """Measure how well the CG Laplacian reconstructs the original diffusion signal.

        For each time step t, computes:
        - correlation between p_orig(t) and p_cg(t)
        - RMSE
        - R² (coefficient of determination)

        Returns dict with keys: 'time', 'correlation', 'rmse', 'r2'.
        """
        if self.coarse_grained_laplacian is None:
            raise ValueError("Call coarse_grain() first.")

        L_orig = np.asarray(self.network_builder.laplacian.todense())
        L_cg = self.coarse_grained_laplacian
        n = L_orig.shape[0]
        p0 = np.ones(n) / n

        times, corrs, rmses, r2s = [], [], [], []
        for t in range(1, time_steps + 1):
            pt_orig = expm(-t * L_orig) @ p0
            pt_cg = expm(-t * L_cg) @ p0

            corr = float(np.corrcoef(pt_orig, pt_cg)[0, 1]) if np.std(pt_orig) > 1e-15 else 1.0
            rmse = float(np.sqrt(np.mean((pt_orig - pt_cg) ** 2)))
            ss_res = np.sum((pt_orig - pt_cg) ** 2)
            ss_tot = np.sum((pt_orig - np.mean(pt_orig)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0

            times.append(t)
            corrs.append(corr)
            rmses.append(rmse)
            r2s.append(r2)

        return {"time": times, "correlation": corrs, "rmse": rmses, "r2": r2s}

    def compare_diffusion_dynamics(self, time_steps: int = 10) -> List[float]:
        """
        Compare diffusion dynamics between original and coarse-grained networks.

        Uses the matrix exponential (scipy.linalg.expm) to compute the exact
        diffusion operator e^{-tL}, avoiding element-wise exp.

        Args:
            time_steps: Number of time steps to simulate. Defaults to 10.

        Returns:
            List of relative errors at each time step.

        Raises:
            ValueError: If coarse-grained Laplacian has not been computed.
        """
        if self.coarse_grained_laplacian is None:
            raise ValueError(
                "Coarse-grained Laplacian not computed. Call coarse_grain first."
            )

        # Original Laplacian — convert from np.matrix to np.ndarray
        L_orig = np.asarray(self.network_builder.laplacian.todense())

        # Coarse-grained Laplacian (already ndarray)
        L_cg = self.coarse_grained_laplacian

        # Initial state (uniform distribution)
        p0 = np.ones(L_orig.shape[0]) / L_orig.shape[0]

        # Simulate diffusion on both networks using matrix exponential
        errors = []
        for t in range(1, time_steps + 1):
            # Original diffusion: e^{-t L_orig} p0
            pt_orig = expm(-t * np.asarray(L_orig)) @ p0

            # Coarse-grained diffusion: e^{-t L_cg} p0
            pt_cg = expm(-t * np.asarray(L_cg)) @ p0

            # Compute relative error
            error = np.linalg.norm(pt_orig - pt_cg) / np.linalg.norm(pt_orig)
            errors.append(error)

        return errors
